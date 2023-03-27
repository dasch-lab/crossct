"""Runs the tracking with same parametrization as in cell tracking challenge."""
import os
import re
from pathlib import Path

import numpy as np
from tifffile import imread

from tracking.export import ExportResults
from tracking.extract_data import get_img_files
from tracking.extract_data import get_mask_positions
from tracking.tracker import TrackingConfig, MultiCellTracker


def track(img_path, seg_path, res_path):
    img_path = Path(img_path)
    seg_path = Path(seg_path)
    res_path = Path(res_path)
    gt_path = Path(img_path.as_posix()+'_GT/TRA')

    img_files = get_img_files(img_path)
    segm_files = get_img_files(seg_path)

    # set roi size
    dummy = np.squeeze(imread(segm_files[0]))
    shape = dummy.shape
    if len(shape) == 2:
        roi_size = (150, 150)
    else:
        roi_size = (100, 100, 100)

    if img_path.parent.name in ['Fluo-N3DL-DRO', 'Fluo-N3DL-TRIC', 'Fluo-N3DL-TRIF']:
        seed_file = [Path(gt_path) / file for file in os.listdir(gt_path) if file.endswith('.tif')]
        seed_file.sort(key=lambda x: int(re.match(r'(\D+)(\d*)(\D+)', x.name).groups()[1]))
        seed_file = seed_file[0]
        seeds = get_mask_positions(imread(seed_file.as_posix()))
        roi_size = (60, 60, 60)
    else:
        seeds = None

    config = TrackingConfig(img_files, segm_files, seeds, roi_size, delta_t=3)
    tracker = MultiCellTracker(config)
    tracks = tracker()
    exporter = ExportResults()
    exporter(tracks, res_path, tracker.img_shape, time_steps=sorted(img_files.keys()))


if __name__ == '__main__':
    from argparse import ArgumentParser

    PARSER = ArgumentParser(description='Cell Tracking')
    PARSER.add_argument('img_path')
    PARSER.add_argument('segm_path')
    PARSER.add_argument('res_path')
    ARGS = PARSER.parse_args()
    track(ARGS.img_path, ARGS.segm_path, ARGS.res_path)
