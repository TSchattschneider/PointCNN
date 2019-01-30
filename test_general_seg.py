#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import math
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tqdm import tqdm, trange

import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--experiment_descr',
                        help='Description of the dataset used for testing, used as directory name for saving results',
                        type=str, default='', required=True)
    parser.add_argument('--max_point_num', '-p', help='Max point number of each sample', type=int, default=8192)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    args = parser.parse_args()

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    max_point_num = args.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    ######################################################################
    # First, create designated output folder. Abort if it already exists to prevent accidental overwriting.
    model_dir = os.path.abspath(os.path.join(args.load_ckpt, '../..'))  # Get this model's specific file directory
    preds_folder = os.path.join(model_dir, args.experiment_descr + '_preds')
    try:
        os.mkdir(preds_folder)
    except FileExistsError:
        print('{}-Output folder {} already exists. Aborting.'.format(datetime.now(), preds_folder), file=sys.stderr)
        sys.exit(1)
    ######################################################################

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    print('{}-Creating network architecture...'.format(datetime.now()))
    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        tqdm.write('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        folder = os.path.dirname(args.filelist)
        filenames = [os.path.join(folder, line.strip()) for line in open(args.filelist)]
        for filename in tqdm(filenames, ncols=80):
            # tqdm.write('{}-Reading {}...'.format(datetime.now(), filename))
            data_h5 = h5py.File(filename)
            data = data_h5['data'][...].astype(np.float32)
            data_num = data_h5['data_num'][...].astype(np.int32)
            batch_num = data.shape[0]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            # tqdm.write('{}-{:d} testing batches. Creating predictions...'.format(datetime.now(), batch_num))
            for batch_idx in range(batch_num):
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = math.ceil((sample_num * batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

                seg_probs = sess.run([seg_probs_op],
                                     feed_dict={
                                         pts_fts: points_batch,
                                         indices: indices_batch,
                                         is_training: False,
                                     })
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])

            h5_name = os.path.splitext(os.path.split(filename)[1])[0] + '_pred.h5'  # Create a new file name
            filepath_pred = os.path.join(preds_folder, h5_name)
            # tqdm.write('{}-Saving {}...'.format(datetime.now(), filepath_pred))
            file = h5py.File(filepath_pred, 'w')
            file.create_dataset('data_num', data=data_num)
            file.create_dataset('label_seg', data=labels_pred)
            file.create_dataset('confidence', data=confidences_pred)
            has_indices = 'indices_split_to_full' in data_h5
            if has_indices:
                file.create_dataset('indices_split_to_full', data=data_h5['indices_split_to_full'][...])
            file.close()

            if args.save_ply:
                # tqdm.write('{}-Saving ply of {}...'.format(datetime.now(), os.path.basename(filepath_pred)))
                export_ply_monolithic(data, data_num, filepath_pred, labels_pred, setting)
            ######################################################################
        tqdm.write('{}-Done!'.format(datetime.now()))


def export_ply_blocks(batched_data, data_num, filepath_pred, labels, setting):
    folder = os.path.join(os.path.dirname(filepath_pred), 'PLY')
    filename = os.path.splitext(os.path.basename(filepath_pred))[0]
    filepath_label_ply = os.path.join(folder, filename)
    data_utils.save_ply_property_batch(batched_data[:, :, 0:3], labels[...],
                                       filepath_label_ply, data_num[...], setting.num_class)


def export_ply_monolithic(batched_data, data_num, filepath_pred, batched_labels, setting):
    assert len(batched_data) == len(data_num)
    assert batched_data.shape[0:2] == batched_labels.shape

    # Take the the predefined valid number of points out of the batches and create a contiguous arrays.
    tmp_data = []
    tmp_labels = []
    for idx, (data_batch, label_batch) in enumerate(zip(batched_data, batched_labels)):
        num_points = data_num[idx]
        tmp_data.append(data_batch[:num_points])
        tmp_labels.append(label_batch[:num_points])
    data = np.concatenate(tmp_data)
    labels = np.concatenate(tmp_labels).reshape(-1, 1)

    folder = Path(filepath_pred).parent / 'PLY'
    filename = Path(filepath_pred).with_suffix('.ply').name
    filepath_label_ply = folder / filename
    data_utils.save_ply(data, str(filepath_label_ply), labels=labels)


def export_ply_monolithic_with_RGB_labels(batched_data, data_num, filepath_pred, batched_labels, setting):
    assert len(batched_data) == len(data_num)
    assert batched_data.shape[0:2] == batched_labels.shape

    # Take the the predefined valid number of points out of the batches and create a contiguous arrays.
    tmp_data = []
    tmp_labels = []
    for idx, (data_batch, label_batch) in enumerate(zip(batched_data, batched_labels)):
        num_points = data_num[idx]
        tmp_data.append(data_batch[:num_points])
        tmp_labels.append(label_batch[:num_points])
    data = np.concatenate(tmp_data)
    labels = np.concatenate(tmp_labels)

    # Create a lookup table (LUT) for efficient label-to-color mapping
    cmap = cm.get_cmap('tab20')
    label_max = setting.num_class
    cmap_LUT = np.array([cmap(label / label_max)[:3] for label in range(label_max)])
    cmap_LUT[0] = (0.0, 0.0, 0.0)

    # Create segment colors according to the labels, using the color LUT
    rgb_labels = cmap_LUT[labels]

    folder = Path(filepath_pred).parent / 'PLY'
    filename = Path(filepath_pred).with_suffix('.ply').name
    filepath_label_ply = folder / filename
    data_utils.save_ply(data, str(filepath_label_ply), colors=rgb_labels)


if __name__ == '__main__':
    main()
