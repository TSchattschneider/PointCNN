import logging
from pathlib import Path
import sys
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

from meta_definitions import DATA_DIR


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt='%d.%m.%y - %H:%M:%S',
                        stream=sys.stdout)

    sceneNN_folder = Path(DATA_DIR, 'SceneNN/untouched')
    output_folder = Path(DATA_DIR, 'SceneNN/preprocessed_rgb')
    logging.debug(f"Reading from folder: {sceneNN_folder}")

    h5_filenames = list(sceneNN_folder.glob('*seg*.hdf5'))

    h5_filename: Path
    for idx, h5_filename in enumerate(tqdm(h5_filenames, ncols=80)):
        logging.debug(f"Loading {h5_filename}. Scene {idx} / {len(h5_filenames)}...")
        target_filepath = output_folder / h5_filename.name
        if target_filepath.exists():
            logging.debug(f"Skipping {h5_filename.name}. File already exists.")
        else:
            with h5py.File(h5_filename, 'r') as h5file:
                data = np.array(h5file['data'])
                segmentation_labels = np.array(h5file['label'])

                # Only keep the XYZ dimensions, discard color for now
                # data = data[:, :, 9:12]

                # Keep XYZ[9:12] and color[6:9] data, discard the rest
                data = data[:, :, 6:12]

                # Fields required by 'load_seg' function of PointCNN
                data_num = np.full(data.shape[0], data.shape[1])
                labels = np.zeros(data.shape[0])

                if not output_folder.exists():
                    Path.mkdir(output_folder)

                with h5py.File(target_filepath, 'w') as output:
                    output.create_dataset('data', data=data)
                    output.create_dataset('data_num', data=data_num)
                    output.create_dataset('label', data=labels)
                    output.create_dataset('label_seg', data=segmentation_labels)


if __name__ == "__main__":
    main()
    logging.info("Finished.")
