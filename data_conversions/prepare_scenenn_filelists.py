#!/usr/bin/env python3
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

data_dir = Path.home() / 'Thesis/data/SceneNN/preprocessed_rgb'
train_dir = data_dir / 'train'
val_dir = data_dir / 'validation'
test_dir = data_dir / 'test'
filelist_dir = data_dir / 'filelists'

train_filelist = data_dir / 'train_files.txt'
validation_filelist = data_dir / 'validation_files.txt'
test_filelist = Path(data_dir / 'test_files.txt')


def random_split(args):
    h5files = list(data_dir.glob('*.hdf5'))

    # Split data into subdirectories
    if any(h5files):
        # Split data into train, validation and test set (60 / 20 / 20)
        files_train, files_test = train_test_split(h5files, test_size=0.2)
        files_train, files_val = train_test_split(files_train, test_size=0.25)

        # Move the sets to seperate subdirectories.
        # the replace command moves a file.
        train_dir.mkdir()
        for file in files_train:
            destination = train_dir / file.name
            file.replace(destination)

        val_dir.mkdir()
        for file in files_val:
            destination = val_dir / file.name
            file.replace(destination)

        test_dir.mkdir()
        for file in files_test:
            destination = test_dir / file.name
            file.replace(destination)

    # Split list of training files into chunks
    if not filelist_dir.exists():
        files_train = list(train_dir.glob('*.hdf5'))
        n = args.chunk_size
        chunks_train = [files_train[i:i + n] for i in range(0, len(files_train), n)]

        # Write lists of chunked files to a text file
        filelist_dir.mkdir()
        for idx, chunk in enumerate(chunks_train):
            chunk_textfile = filelist_dir / f'train_chunk_{idx}.txt'
            with chunk_textfile.open(mode='w') as txt:
                for filepath in chunk:
                    txt.write(f'{filepath}\n')

        # Group the chunk text files into a single text file
        with train_filelist.open(mode='w') as txt:
            chunk_textfiles = list(filelist_dir.glob('*chunk*.txt'))
            for filepath in chunk_textfiles:
                txt.write(f'{filepath}\n')

    # Write list of validation files
    with validation_filelist.open(mode='w') as txt:
        files_val = list(val_dir.glob('*.hdf5'))
        for filepath in files_val:
            txt.write(f'{filepath}\n')

    # Write list of test files
    with test_filelist.open(mode='w') as txt:
        files_test = list(test_dir.glob('*.hdf5'))
        for filepath in files_test:
            txt.write(f'{filepath}\n')


def filelist_split(args):
    h5files = list(data_dir.glob('*.hdf5'))

    # Split data into subdirectories
    if any(h5files):

        # Read existing splits from filelists

        files_train = list()
        with train_filelist.open(mode='r') as txt:
            chunks_paths = [Path(line) for line in txt.read().splitlines()]

            for chunk_path in chunks_paths:
                with chunk_path.open(mode='r') as c:
                    files_train.extend([Path(line).name for line in c.read().splitlines()])

        with validation_filelist.open(mode='r') as txt:
            files_val = [Path(line).name for line in txt.read().splitlines()]

        with test_filelist.open(mode='r') as txt:
            files_test = [Path(line).name for line in txt.read().splitlines()]

        # Move the sets to seperate subdirectories.
        # the replace command moves a file.

        train_dir.mkdir()
        for file in files_train:
            filepath = data_dir / file
            destination = train_dir / file
            filepath.replace(destination)

        val_dir.mkdir()
        for file in files_val:
            filepath = data_dir / file
            destination = val_dir / file
            filepath.replace(destination)

        test_dir.mkdir()
        for file in files_test:
            filepath = data_dir / file
            destination = test_dir / file
            filepath.replace(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Group SceneNN HDF5 files into datasets.")
    subparsers = parser.add_subparsers()

    parser_random = subparsers.add_parser('random_split')
    parser_random.add_argument('--chunk_size', type=int, default=8,
                               help='Size of training file chunks,'
                                    'specifies how many h5 files will be loaded at training time.')
    parser_random.set_defaults(func=random_split)

    parser_filelist = subparsers.add_parser('filelist_split')
    parser_filelist.set_defaults(func=filelist_split)

    args = parser.parse_args()
    args.func(args)
