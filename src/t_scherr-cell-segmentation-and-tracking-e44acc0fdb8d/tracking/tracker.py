"""Functionalities to track cells."""
import numpy as np
from tifffile import imread

from tracking.extract_data import get_mask_positions
from tracking.flow import compute_fft_displacement
from tracking.graph import graph_tracking


class MultiCellTracker:
    """Tracks multiple cells in 2D+t and 3D+t using
    a coupled minimum-cost flow as matching strategy."""
    def __init__(self, config):
        """
        Initialises the tracker.
        Args:
            config: an instance of type TrackingConfig containing the parametrisation
             for the tracking algorithm (e.g. location of the raw and segmentation images,
             tracking objective - track all objects or a few marked cells only)
        """
        self.config = config
        self.cell_rois = {}
        self.tracks = {}
        self.segmentation_masks = {}
        self.img_shape = None
        self.delta_t = self.config.delta_t
        if self.config.seeds is None:
            self.sparse_tracking = False
        else:
            self.sparse_tracking = True

    def __call__(self):
        """Tracks the cells in the provided data set."""
        time_steps = self.config.time_steps[:]
        for i, time in enumerate(time_steps):
            print('#'*20)
            print('timestep:', time)
            print('#'*20)
            if i == 0:
                self.map_seeds_to_segmentation(time)
            self.tracking_step(i, time)
        return self.tracks

    def tracking_step(self, i, time):
        """
        Applies a single tracking step.
        Args:
            i: index of time point
            time: time point

        Returns:

        """
        img = imread(self.config.get_image_file(time))
        if self.img_shape is None:
            self.img_shape = img.shape
        # estimate position of each tracked object
        for track_id, cell_roi in self.cell_rois.items():
            cell_roi(time, img)
        if i > 0:
            # extract neighbors of each tracked objects
            segmented_candidates, all_masks = self.extract_candidates(time)
            # graph-based matching of objects
            self.match_objects(time, segmented_candidates, all_masks, img)
        # remove roi for each track, that has successors
        rois_to_remove = [track_id for track_id, track in self.tracks.items() if len(track.successors) > 0]
        for track_id in rois_to_remove:
            if track_id in self.cell_rois:
                self.cell_rois.pop(track_id)

    def _get_features(self, candidates, track):
        """
        Extracts for each track its features and the features of its neighbors at the next step.
        """
        candidate_features = {m_id: [np.median(positions, axis=1), len(positions[0])]
                              for m_id, positions in candidates.items()}
        estimated_track_position = self.cell_rois[track.track_id].last_roi().center
        track_features = [estimated_track_position, len(track.masks[track.get_last_time()][0])]
        return track_features, candidate_features

    def fill_in_dummy_masks(self, track_id, time, new_mask):
        """Creates dummy masks for missing objects."""
        prev_mask = self.tracks[track_id].masks[self.tracks[track_id].get_last_time()]
        displacement = (np.median(new_mask, axis=1) - np.median(prev_mask, axis=1)).astype(np.int)
        t_last = self.tracks[track_id].get_last_time()
        n_missing_steps = time - t_last
        for step in range(1, n_missing_steps):
            dummy_mask = np.array(prev_mask) + (step / n_missing_steps) * displacement.reshape(-1, 1)
            dummy_mask = dummy_mask.astype(np.uint16)
            out_of_img = np.any(dummy_mask < 0, axis=0) | \
                         np.any(dummy_mask >= np.array(self.img_shape).reshape(-1, 1), axis=0)
            dummy_mask = tuple(dummy_mask[:, ~out_of_img])
            self.cell_rois[track_id].active = True
            self.tracks[track_id].active = True
            self.tracks[track_id].add_time_step(t_last + step, dummy_mask)

    def match_objects(self, time, matching_candidates, all_masks, image):
        """Matches tracks from previous time points to objects at current time point."""

        # extract features of tracks and their potential matched_objects candidates
        track_features = {}
        candidate_features = {}
        for track_id, candidates in matching_candidates.items():
            track = self.tracks[track_id]
            if len(candidates.keys()) > 0:
                features_track, features_candidates = self._get_features(candidates, track)
                track_features[track.track_id] = features_track
                candidate_features[track.track_id] = features_candidates
            else:
                track.active = False
                self.cell_rois[track.track_id].active = False
        # match objects using coupled minimum-cost flow tracking
        matched_objects = graph_tracking(track_features, candidate_features,
                                         cutoff_distance=self.config.cut_off_distance,
                                         allow_cell_division=self.config.allow_cell_division)
        # add matched objects to tracks
        for track_id, matched_ids in matched_objects.items():
            # no match
            if len(matched_ids) == 0:
                self.tracks[track_id].active = False
                self.cell_rois[track_id].active = False
            # exactly one match -> append to track
            elif len(matched_ids) == 1:
                new_mask = matching_candidates[track_id][matched_ids[0]]
                if not self.tracks[track_id].active:
                    assert len(self.tracks[track_id].successors) == 0, 'object has successors'
                    self.fill_in_dummy_masks(track_id, time, new_mask)

                self.tracks[track_id].add_time_step(time, new_mask)
                self.cell_rois[track_id](time, image)
                new_center = np.median(new_mask, axis=1)
                self.cell_rois[track_id].correct_last_roi(new_center, image)

            # track has more than one successor -> assume cell division
            elif len(matched_ids) == 2:
                self.tracks[track_id].active = False
                self.cell_rois[track_id].active = False
                self.tracks[track_id].successors = tuple(matched_ids)
                for mask_id in matched_ids:
                    self.init_new_track(time, track_id, matching_candidates[track_id][mask_id])
                    new_track_id = max(list(self.tracks.keys()))
                    # inherit mask\img crop from parent
                    self.cell_rois[new_track_id](time, image)

        # new candidates without predecessor -> initialize new tracks
        if not self.sparse_tracking:
            matched_ids = []
            for m_id in matched_objects.values():
                matched_ids.extend(m_id)
            mask_id = all_masks.index.values
            new_ids = mask_id[~np.isin(mask_id, matched_ids)]
            for mask_id in new_ids:
                self.init_new_track(time, 0, all_masks[mask_id])
                new_track_id = max(list(self.tracks.keys()))
                self.cell_rois[new_track_id](time, image)

    def init_new_track(self, time, pred_track_id, mask):
        """Initalises a new track."""
        if self.tracks.keys():
            track_id = max(list(self.tracks.keys())) + 1
        else:
            track_id = 1  # start with 1 as background is 0, no predecessor is 0
        self.tracks[track_id] = CellTrack(track_id, pred_track_id)
        self.tracks[track_id].add_time_step(time, mask)
        self.cell_rois[track_id] = CellROI(track_id, np.median(mask, axis=1), self.config.roi_box_size)

    def extract_candidates(self, time):
        """Selects for each track based on its ROI a set of potential matching candidates."""
        segmentation, mask_indices = self.config.get_segmentation_masks(time)
        img_shape = segmentation.shape
        matching_candidates = {}
        for cell_roi in self.cell_rois.values():
            # extract for all tracks existing at t-n...t-1 possible matching candidates at t
            if (time - self.tracks[cell_roi.track_id].get_last_time() < self.delta_t+1) \
                    and len(self.tracks[cell_roi.track_id].successors) == 0:
                coords = cell_roi.last_roi_crop_box(img_shape)
                mask_ids = segmentation[coords]
                mask_ids = np.unique(mask_ids[mask_ids > 0])
                if len(mask_ids) > 0:
                    matching_candidates[cell_roi.track_id] = mask_indices[np.isin(mask_indices.index, mask_ids)]
                else:
                    matching_candidates[cell_roi.track_id] = {}

        return matching_candidates, mask_indices

    def map_seeds_to_segmentation(self, time):
        """Maps initial tracking seeds to segmentation masks."""
        segmentation_mask, mask_indices = self.config.get_segmentation_masks(time)
        # map initial tracking seeds with segmentation
        taken_masks = []
        if self.sparse_tracking:
            for track_id, seed_mask in self.config.seeds.items():
                mask_ids = segmentation_mask[seed_mask]
                mask_id = np.unique(mask_ids[mask_ids > 0])
                if len(mask_id) == 0 or np.any(np.isin(mask_id, taken_masks)):
                    mask = seed_mask
                else:
                    taken_masks.extend(mask_id)
                    if len(mask_id) > 1:
                        mask = tuple(np.stack([np.hstack(m)
                                               for m in list(zip(*mask_indices.loc[mask_id].values))]))
                    else:
                        mask = tuple(*np.stack(mask_indices.loc[mask_id].values))  # * to unpack [[]] -> []
                if track_id not in self.tracks:
                    self.init_new_track(time, 0, mask)
        else:
            for mask_id, coordinates in mask_indices.items():
                self.init_new_track(time, 0, coordinates)


class CellTrack:
    """Contains the object position of an object and its predecessor/successors over time."""
    def __init__(self, track_id, pred_track_id=0):
        """
        Initialises a track.
        Args:
            track_id: a unique track id
            pred_track_id: id of the predecessor track, 0: no predecessor
        """
        self.track_id = track_id
        self.pred_track_id = pred_track_id
        self.successors = tuple()
        self.masks = {}
        self.active = True

    def add_time_step(self, time, mask_indices):
        """Adds a segmentation mask to the track."""
        if self.active:
            self.masks[time] = mask_indices

    def get_last_position(self):
        """Returns the last position of the tracked object."""
        last_time = sorted(list(self.masks.keys()))[-1]
        return np.median(np.stack(self.masks[last_time]), axis=-1)

    def get_last_time(self):
        """Returns the last time point a segmentation mask has been added to the track."""
        return sorted(list(self.masks.keys()))[-1]


class CellROI:
    """Defines for each tracked object a region of interest (ROI) in which to expect the
    object at a time step."""
    def __init__(self, track_id, init_position, roi_size):
        """
        Initialises a ROI for an object.
        Args:
            track_id: tracking id of the track associated with this ROI
            init_position: initial position of the tracked object
            roi_size: a tuple providing the ROI size (rectangular ROI)
        """
        self.track_id = track_id
        self.init_position = init_position
        self.roi_size = roi_size
        self.roi = {}  # frame_id: ROI (center,box size)
        self.img_patches = {}
        self.active = True
        self.displacement = {}

    def __call__(self, time, image):
        """Propagates the ROI to a new time step."""
        self._add_time_step(time, image)

    def _add_time_step(self, time, image):
        """Propagates the ROI based on the estimated displacement between last and current image crop."""
        if self.roi:
            last_img_patch = self.get_last_img_patch()
            last_roi = self.last_roi()
            img_patch = image[last_roi.crop_box(image.shape)]
            # compute shift between img patches:
            # motion model: p_1 = p_0 + displacement -> p_1: new ROI center
            self.displacement[time] = compute_displacement(last_img_patch, img_patch)
            new_center_pos = last_roi.center + self.displacement[time]
        else:
            new_center_pos = self.init_position
        new_center_pos[new_center_pos < 0] = 0
        new_center_pos[new_center_pos > image.shape] = np.array(image.shape)[new_center_pos > image.shape]
        new_roi = self.create_roi(new_center_pos)
        self.roi[time] = new_roi
        # drop all patches from before to save memory
        self.img_patches = dict()
        self.img_patches[time] = image[new_roi.crop_box(image.shape)].copy()  # avoid references on large images

    def get_last_displacement(self):
        """Returns the last estimated displacement."""
        keys = sorted(list(self.roi.keys()))
        return self.displacement[keys[-1]]

    def last_roi(self):
        """Returns the last added ROI instance."""
        keys = sorted(list(self.roi.keys()))
        return self.roi[keys[-1]]

    def last_roi_crop_box(self, img_shape):
        """Returns of the last added ROI instance the slice."""
        keys = sorted(list(self.roi.keys()))
        return self.roi[keys[-1]].crop_box(img_shape)

    def get_last_img_patch(self):
        """Returns the last added image crop."""
        keys = sorted(list(self.img_patches.keys()))
        return self.img_patches[keys[-1]]

    def create_roi(self, center_point):
        """Creates a new ROI instance."""
        return ROI(center_point, self.roi_size)

    def correct_last_roi(self, center, image):
        """Adapts the position of the last added ROI."""
        last_roi = self.last_roi()
        last_roi.adapt_roi(center)
        time = sorted(list(self.img_patches.keys()))[-1]
        self.img_patches = dict()
        self.img_patches[time] = image[last_roi.crop_box(image.shape)].copy()

    def get_last_time(self):
        """Returns the last time point a ROI has been added."""
        return sorted(list(self.roi.keys()))[-1]


def compute_displacement(patch_1, patch_2):
    """Computes the shift between 2 images"""
    #assert patch_1.shape == patch_2.shape, 'mismatching shapes {} {}'.format(patch_1.shape, patch_2.shape)

    if patch_1.shape == patch_2.shape:
        displacement = compute_fft_displacement(patch_1, patch_2)
    else:
        displacement = 0
    return displacement


class TrackingConfig:
    """Provides the configuration for the MultiCellTracker."""
    def __init__(self, img_files, segm_files, seeds, roi_box_size, delta_t=2,
                 cut_off_distance=None, allow_cell_division=True):
        """
        Initialises the configuration.
        Args:
            img_files: a dict containing the image files {time_point: img_file}
            segm_files: a dict containing the segmentation image files {time_point: img_file}
            seeds: if seeds are provided the tracking objective is to track only the seeded objects, otherwise
                    all segmented object are tracked, seeds are provided in a dict {seed_id: seed_positions},
                    where seed_positions is a tuple containing the indices of the marked pixels
            roi_box_size: a tuple providing the size of a ROI
            delta_t: the maximum time span tracks with no match will be tracked
            cut_off_distance: a distance threshold for the graph-based matching providing
                              a cost for objects to appear/disappear
            allow_cell_division: a boolean for the graph-based matching strategy,
                                 if True: cell divisions are modelled
                                 otherwise no cell divisions are modelled in the graph-based matching strategy
        """
        self.img_files = img_files
        self.segm_files = segm_files
        self.time_steps = sorted(list(self.img_files.keys()))
        self.seeds = seeds
        self.roi_box_size = roi_box_size
        self.cut_off_distance = cut_off_distance
        self.delta_t = delta_t
        self.allow_cell_division = allow_cell_division
        if self.cut_off_distance is None:
            self.cut_off_distance = max(roi_box_size)

    def get_image_file(self, time_point):
        """Returns the image file at a time point."""
        return self.img_files[time_point]

    def get_segmentation_masks(self, time_step):
        """Returns the segmentation mask image and the indices of the pixels of each segmentation mask."""
        segmentation = imread(self.segm_files[time_step])
        segmentation = np.squeeze(segmentation)
        return segmentation, get_mask_positions(segmentation)


class ROI:
    """Defines a rectangular shaped region of interest (ROI)."""
    def __init__(self, center_position, box_size):
        """
        Initialises a ROI.
        Args:
            center_position: center of the ROI
            box_size: size of the ROI
        """
        self.center = np.array(center_position)
        self.box_size = tuple(box_size)
        self.top_left = np.array(np.array(center_position) - np.array(box_size) // 2, np.int)
        self.bottom_right = np.array(np.array(center_position) + np.array(box_size) // 2
                                     + np.array(box_size) % 2, np.int)

    def crop_box(self, img_shape):
        """Returns a slice operator to crop a np.array."""
        # this is basically a[0]:b[0],a[1],b[1],... to select a set of indices from an array
        return tuple([slice(min(max(0, a), shape), min(max(0, b), shape))
                      for a, b, shape in zip(self.top_left, self.bottom_right, img_shape)])

    def adapt_roi(self, center):
        """Adapts the ROI to a new center position."""
        self.center = np.array(center)
        self.top_left = np.array(np.array(self.center) - np.array(self.box_size) // 2, np.int)
        self.bottom_right = np.array(np.array(self.center) + np.array(self.box_size) // 2
                                     + np.array(self.box_size) % 2, np.int)
