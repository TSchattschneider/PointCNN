import logging
import os
import sys
from os import path

import h5py
import numpy as np
from tqdm import tqdm

from meta_definitions import DATA_DIR


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt='%d.%m.%y - %H:%M:%S',
                        stream=sys.stdout)

    sceneNN_folder = path.join(DATA_DIR, 'SceneNN', 'untouched')
    output_folder = path.join(DATA_DIR, 'SceneNN', 'preprocessed')
    logging.debug(f"Reading from folder: {sceneNN_folder}")

    h5_filenames = [filename for filename in os.listdir(sceneNN_folder)
                    if 'seg' in filename and filename.endswith('.hdf5')]

    for idx, h5_filename in enumerate(tqdm(h5_filenames, ncols=80)):
        logging.debug(f"Loading {h5_filename}. Scene {idx} / {len(h5_filenames)}...")
        filepath = path.join(sceneNN_folder, h5_filename)
        with h5py.File(filepath, 'r') as h5file:
            data = np.array(h5file['data'])
            segmentation_labels = np.array(h5file['label'])

            # Only keep the XYZ dimensions, discard color for now
            data = data[:, :, 9:12]

            # Fields required by 'load_seg' function of PointCNN
            data_num = np.full(data.shape[0], data.shape[1])
            labels = np.zeros(data.shape[0])

            output_filepath = path.join(output_folder, h5_filename)
            if not path.exists(os.path.dirname(output_filepath)):
                os.makedirs(path.dirname(output_filepath))

            with h5py.File(output_filepath, 'w') as output:
                output.create_dataset('data', data=data)
                output.create_dataset('data_num', data=data_num)
                output.create_dataset('label', data=labels)
                output.create_dataset('label_seg', data=segmentation_labels)


if __name__ == "__main__":
    main()
    logging.info("Finished.")
