import argparse
import math
from pathlib import Path

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size',
                        help='Size of training file chunks, '
                             'specifies how many h5 files will be loaded at a time.',
                        type=int, default=8)
    args = parser.parse_args()

    data_dir = Path.home() / 'Thesis/data/SceneNN/preprocessed'
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'validation'
    test_dir = data_dir / 'test'
    h5files = list(data_dir.glob('*.hdf5'))

    # Split data into subdirectories
    if any(h5files):
        # Split data into train, validation and test set (60 / 20 / 20)
        files_dev, files_test = train_test_split(h5files, test_size=0.2)
        files_train, files_val = train_test_split(files_dev, test_size=0.25)

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
    files_train = list(train_dir.iterdir())
    n = args.chunk_size
    chunks_train = [files_train[i:i + n] for i in range(0, len(files_train), n)]

    # Write lists of chunked files to txt
    for idx, chunk in enumerate(chunks_train):
        with Path(data_dir / f'train_chunk_{idx}.txt').open(mode='w') as f:
            for path in chunk:
                f.write(f'{path}\n')


if __name__ == '__main__':
    main()
