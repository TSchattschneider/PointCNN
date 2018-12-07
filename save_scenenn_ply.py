#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import importlib
import os
import sys

from matplotlib import cm
import numpy as np

import data_utils


class AttrDict(dict):
    """Allows to address keys as if they were attributes."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


if __name__ == '__main__':
    args = AttrDict()
    args.model = 'pointcnn_seg'
    args.setting = "scenenn_x8_2048_fps"
    args.load_ckpt = "../models/pointcnn_seg_scenenn_x8_2048_fps_2018-12-04-15-20-55_23790/ckpts/iter-108000"
    args.file_name = "scenenn_seg_032.hdf5"  # os.listdir(args.data_folder)

    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    import pickle

    with open(os.path.join(os.path.dirname(args.load_ckpt), os.pardir, 'data.pickle'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(os.path.dirname(args.load_ckpt), os.pardir, 'labels_pred.pickle'), 'rb') as f:
        labels_pred = pickle.load(f)

    print('{}-Saving ply of {}...'.format(datetime.now(), args.file_name))
    scene_name = os.path.splitext(os.path.basename(args.file_name))[0]  # Get filename without extension
    # Create subfolder
    predictions_folder = os.path.join(os.path.dirname(args.load_ckpt), os.pardir, 'predictions', 'test')
    if not os.path.exists(os.path.dirname(predictions_folder)):
        os.makedirs(predictions_folder)

    # Create point colors according to labels
    cmap = cm.get_cmap('tab20')
    label_max = setting.num_class
    cmap_LUT = [cmap(label / label_max)[:3] for label in range(label_max)]
    cmap_LUT[0] = (0.0, 0.0, 0.0)
    colors = np.array([cmap_LUT[label] for label in labels_pred.ravel()])
    filename = os.path.join(predictions_folder, scene_name + '_nudelholz.ply')
    data_utils.save_ply(data.reshape(-1, 3), filename, colors)

    print('{}-Done!'.format(datetime.now()))
