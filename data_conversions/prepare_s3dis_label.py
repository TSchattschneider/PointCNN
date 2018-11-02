#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # Enable imports from project root directory
from meta_definitions import DATA_DIR

BASE_DIR = path.join(DATA_DIR, 'S3DIS')

object_dict = {
            'clutter':   0,
            'ceiling':   1,
            'floor':     2,
            'wall':      3,
            'beam':      4,
            'column':    5,
            'door':      6,
            'window':    7,
            'table':     8,
            'chair':     9,
            'sofa':     10,
            'bookcase': 11,
            'board':    12}

path_dir_areas =  [entry for entry in os.listdir(BASE_DIR)
                   if path.isdir(path.join(BASE_DIR, entry))]

if "prepare_label_rgb" in path_dir_areas:
    path_dir_areas.remove("prepare_label_rgb")

for area in path_dir_areas:
    print("Area:", area)
    path_dir_rooms = os.listdir(path.join(BASE_DIR, area))
    for room in path_dir_rooms:
        print("Room:", room)
        # make store directories
        path_prepare_label = path.join(DATA_DIR, "S3DIS", "prepare_label_rgb", area, room)
        if not os.path.exists(path_prepare_label):
            os.makedirs(path_prepare_label)
        elif len(os.listdir(path_prepare_label)) != 0:
            print("Room data already exists, skipping.")
            continue
        #############################
        xyz_room_list = list()
        label_room_list = list()
        path_annotations = path.join(BASE_DIR, area, room, "Annotations")
        path_items = os.listdir(path_annotations)
        for item in path_items:
            label = item.split("_", 1)[0]
            if label in object_dict:
                xyz_object = np.loadtxt(path.join(path_annotations, item)) # (N,6)
                label_object = np.full((xyz_object.shape[0], 1), object_dict[label])  # (N,1)
            else:
                continue

            xyz_room_list.append(xyz_object)
            label_room_list.append(label_object)

        xyz_room = np.vstack(xyz_room_list)
        label_room = np.vstack(label_room_list)

        np.save(path.join(path_prepare_label, "xyzrgb.npy"), xyz_room)
        np.save(path.join(path_prepare_label, "label.npy"), label_room)

    print(area, "done.\n")