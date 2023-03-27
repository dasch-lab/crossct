"""Utilities to export tracking results to ctc metrics readable format
 (tracking masks + txt file with lineage)"""
import os

import numpy as np
import pandas as pd
from tifffile import imsave


class ExportResults:
    """Exports tracking results in a ctc tracking metrics readable format."""
    def __init__(self):
        self.img_file_name = 'mask'
        self.img_file_ending = '.tif'
        self.track_file_name = 'res_track.txt'
        self.time_steps = None

    def __call__(self, tracks, export_dir, img_shape, time_steps):
        """
        Exports tracks to a given export directory.
        Args:
            tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to
            img_shape: a tuple providing the img shape of the original data
            time_steps: a list of time steps the tracking was applied to

        Returns:

        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        tracks = remove_short_tracks(tracks)
        tracks = fill_empty_tracking_images(tracks, time_steps)
        self.time_steps = time_steps
        self.create_track_file(tracks, export_dir)
        self.create_segm_masks(tracks, export_dir, img_shape)

    def create_track_file(self, all_tracks, export_dir):
        """
        Creates a res_track.txt file readable by TRA measure.
        Args:
            all_tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to

        Returns:

        """
        track_info = {'track_id': [], 't_start': [], 't_end': [], 'predecessor_id': []}
        for track in all_tracks.values():

            track_info['track_id'].append(track.track_id)
            frame_ids = sorted(list(track.masks.keys()))
            track_info['t_start'].append(frame_ids[0])
            track_info['t_end'].append(frame_ids[-1])
            track_info['predecessor_id'].append(track.pred_track_id)
        df = pd.DataFrame.from_dict(track_info)
        df.to_csv(os.path.join(export_dir, self.track_file_name),
                  columns=["track_id", "t_start", "t_end", 'predecessor_id'],
                  sep=' ', index=False, header=False)

    def create_segm_masks(self, all_tracks, export_dir, img_shape):
        """
        Creates for each time step a tracking image with masks
        corresponding to the segmented and tracked objects.
        Args:
             all_tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to
            img_shape: a tuple providing the img shape of the original data

        Returns:

        """
        tracks_in_frame = {time: [] for time in self.time_steps}

        for track_data in all_tracks.values():
            time_steps = sorted(list(track_data.masks.keys()))
            for time in time_steps:
                if time not in tracks_in_frame:
                    tracks_in_frame[time] = []
                tracks_in_frame[time].append(track_data.track_id)

        t_max = sorted(list(tracks_in_frame.keys()))[-1]
        z_fill = np.int(np.ceil(max(np.log10(t_max), 3)))  # either 3 or 4 digits
        all_tracking_masks = np.zeros(img_shape, dtype=np.uint16)
        for time, track_ids in tracks_in_frame.items():
            all_tracking_masks *= 0
            for t_id in track_ids:
                track = all_tracks[t_id]
                mask = track.masks[time]
                if not isinstance(mask, tuple):
                    mask = tuple(*mask)
                conflicting_masks = np.unique(all_tracking_masks[mask])
                conflicting_masks = conflicting_masks[conflicting_masks > 0]
                # resolve mask conflicts (due to adding segmentation masks for missing objects)
                if len(conflicting_masks) > 0:
                    sub_mask = mask[:]
                    for c_mask in conflicting_masks:
                        other_mask_coords = np.where(all_tracking_masks == c_mask)
                        mask_flat = np.ravel_multi_index(np.array(sub_mask), all_tracking_masks.shape)
                        other_mask_flat = np.ravel_multi_index(np.array(other_mask_coords),
                                                               all_tracking_masks.shape)
                        # mask fully covered by other mask -> overwrite over mask
                        # mask overwrites other mask -> reduce mask size
                        if np.all(np.isin(other_mask_flat, mask_flat)):
                            flat_ids = mask_flat[~np.isin(mask_flat, other_mask_flat)]
                            sub_mask = np.unravel_index(flat_ids, all_tracking_masks.shape)
                            sub_mask = tuple(sub_mask)
                        # else: # partial overlap
                    all_tracking_masks[sub_mask] = t_id
                else:
                    all_tracking_masks[mask] = t_id

            file_name = ''.join([self.img_file_name, str(time).zfill(z_fill), self.img_file_ending])
            imsave(os.path.join(export_dir, file_name),
                   all_tracking_masks, compress=3,bigtiff=True)


def remove_short_tracks(all_tracks):
    """
    Removes single time tracks without predecessor+successor
    Args:
        all_tracks:  a dictionary containing the trajectories

    Returns:
        a dictionary containing the edited trajectories
    """
    predecessor = [track.pred_track_id for track in all_tracks.values()]
    temp = all_tracks.copy()
    for track_id, track in all_tracks.items():
        frame_ids = sorted(list(track.masks.keys()))
        if (len(frame_ids) == 1) and (track.pred_track_id == 0) and (track_id not in predecessor):
            temp.pop(track_id)
    return temp


def fill_empty_tracking_images(all_tracks, time_steps):
    """
    Fills missing tracking frames with the temporally closest, filled tracking frame
    Args:
        all_tracks:  a dictionary containing the trajectories
        time_steps: a list of time steps the tracking was run on

    Returns:
        a dictionary containing the edited trajectories

    """
    tracks_in_frame = {}
    for track_data in all_tracks.values():
        track_timesteps = sorted(list(track_data.masks.keys()))
        for time in track_timesteps:
            if time not in tracks_in_frame:
                tracks_in_frame[time] = []
            tracks_in_frame[time].append(track_data.track_id)
    if sorted(time_steps) != sorted(list(tracks_in_frame.keys())):
        empty_timesteps = sorted(np.array(time_steps)[~np.isin(time_steps, list(tracks_in_frame.keys()))])
        filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
        for empty_frame in empty_timesteps:
            nearest_filled_frame = filled_timesteps[np.argmin(abs(filled_timesteps-empty_frame))]
            track_ids = tracks_in_frame[nearest_filled_frame]
            for track_id in track_ids:
                all_tracks[track_id].masks[empty_frame] = all_tracks[track_id].masks[nearest_filled_frame]
            tracks_in_frame[empty_frame] = track_ids
            filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
    return all_tracks
