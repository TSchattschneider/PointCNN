#!/usr/bin/python3
# coding: utf-8
'''
Merge the segmentation predictions of differently offset point cloud sub-blocks.
The merging strategy is to use the prediction with higher confidence for each point.
'''

import argparse
import collections
from pathlib import Path

import h5py
import numpy as np


def main(args):
    datafolder = Path(args.datafolder)
    output_filepath = datafolder / 'preds.npy'

    file_list = list(datafolder.glob('*.h5'))

    zero_offset_data = []
    half_offset_data = []

    total_data_num = 0

    for filepath in file_list:
        with h5py.File(filepath, 'r') as h5file:
            labels_seg = np.array(h5file['label_seg'])
            indices = h5file['indices_split_to_full'][:, :, 1]
            confidence = np.array(h5file['confidence'])
            data_num = np.array(h5file['data_num'])

            total_data_num += data_num.sum()

            filename = filepath.name
            if 'zero' in filename:
                zero_offset_data.append((labels_seg, indices, confidence, data_num))
            else:
                half_offset_data.append((labels_seg, indices, confidence, data_num))

    print(f"Total number of points: {total_data_num:,}")

    merged_label_zero = np.zeros(total_data_num, dtype=int)
    merged_confidence_zero = np.zeros(total_data_num, dtype=float)
    merged_label_half = np.zeros(total_data_num, dtype=int)
    merged_confidence_half = np.zeros(total_data_num, dtype=float)

    final_preds = np.zeros(total_data_num, dtype=int)

    for data_tuple in zero_offset_data:
        labels_seg, indices, confidence, data_num = data_tuple  # Unpack data

        for i in range(labels_seg.shape[0]):
            # XXX Error: There are bigger indices than there is
            #   data as indicated by the data_num
            merged_label_zero[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
            merged_confidence_zero[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]

    for data_tuple in half_offset_data:
        labels_seg, indices, confidence, data_num = data_tuple  # Unpack data

        for i in range(labels_seg.shape[0]):
            merged_label_half[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
            merged_confidence_half[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]

    final_preds[merged_confidence_zero >= merged_confidence_half] = \
        merged_label_zero[merged_confidence_zero >= merged_confidence_half]
    final_preds[merged_confidence_zero < merged_confidence_half] = \
        merged_label_half[merged_confidence_zero < merged_confidence_half]

    np.save(output_filepath, final_preds)
    print(f"Merged predictions saved to {output_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    args = parser.parse_args()
    main(args)
    print("Finished.")
