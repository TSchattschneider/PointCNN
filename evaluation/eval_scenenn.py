#!/usr/bin/env python3
'''
Script to run evaluation metrics on the SceneNN dataset.
'''
import argparse

from meta_definitions import DATA_DIR


def main(args):
    # TODO Provide preds via argument, compare with ground truth via DATA_DIR
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    args = parser.parse_args()
    main(args)
    print("Finished.")
