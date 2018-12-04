#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime


class AttrDict(dict):
    """Allows to address keys as if they were attributes."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


if __name__ == '__main__':
    args = AttrDict()
    args.model = 'pointcnn_seg'
    args.setting = "scenenn_x8_2048_fps"
    args.load_ckpt = "../models/pointcnn_seg_scenenn_x8_2048_fps_2018-11-30-17-12-21_29945/ckpts/iter-2000"
    args.data_folder = "data/SceneNN/preprocessed"
    args.file_names = ["scenenn_seg_237.hdf5"]
    args.max_point_num = 4096
    args.repeat_num = 4
    args.save_ply = True

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    max_point_num = args.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    points_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_sampled = None

    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        filepaths = [os.path.join(args.data_folder, filename) for filename in args.file_names]
        for filepath in filepaths:
            print('{}-Reading {}...'.format(datetime.now(), filepath))
            data_h5 = h5py.File(filepath)
            data = data_h5['data'][...].astype(np.float32)
            data_num = data_h5['data_num'][...].astype(np.int32)
            batch_num = data.shape[0]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    print('{}-Processing {} of {} batches.'.format(datetime.now(), batch_idx, batch_num))
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]  # 4096

                tile_num = math.ceil((sample_num * batch_size) / point_num)  # 8192 / 4096 = 2
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

            filename_pred = filepath[:-5] + '_pred.h5'
            # print('{}-Saving {}...'.format(datetime.now(), filename_pred))
            # file = h5py.File(filename_pred, 'w')
            # file.create_dataset('data_num', data=data_num)
            # file.create_dataset('label_seg', data=labels_pred)
            # file.create_dataset('confidence', data=confidences_pred)
            # has_indices = 'indices_split_to_full' in data_h5
            # if has_indices:
            #     file.create_dataset('indices_split_to_full', data=data_h5['indices_split_to_full'][...])
            # file.close()

            if args.save_ply:
                print('{}-Saving ply of {}...'.format(datetime.now(), filename_pred))
                scene_name = os.path.splitext(os.path.basename(filepath))[0]  # Get filename without extension
                ply_folder = os.path.join(os.path.dirname(filepath), scene_name + '_pred_ply')  # Create subfolder
                if not os.path.exists(os.path.dirname(ply_folder)):
                    os.makedirs(os.path.dirname(ply_folder))
                filepath_label_ply = os.path.join(ply_folder, scene_name + '_ply_label')
                data_utils.save_ply_property_batch(data[:, :, 0:3], labels_pred[...],
                                                   filepath_label_ply, data_num[...], setting.num_class)
            ######################################################################
        print('{}-Done!'.format(datetime.now()))