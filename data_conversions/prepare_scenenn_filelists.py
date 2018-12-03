from pathlib import Path

from sklearn.model_selection import train_test_split


def move_to_subdirectory(files, directory):
    directory.mkdir()
    for file in files:
        destination = directory / file.name
        file.replace(destination)  # The replace command moves the file


root_dir = Path.home() / 'Thesis/data/SceneNN'
data_dir = root_dir / 'preprocessed'
train_dir = data_dir / 'train'
val_dir = data_dir / 'validation'
test_dir = data_dir / 'test'

h5files = list(data_dir.glob('*.hdf5'))

# Split data into subdirectories
if any(h5files):
    # Split data into train, validation and test set (60 / 20 / 20)
    files_dev, files_test = train_test_split(h5files, test_size=0.2)
    files_train, files_val = train_test_split(files_dev, test_size=0.25)

    move_to_subdirectory(files_train, train_dir)
    move_to_subdirectory(files_val, val_dir)
    move_to_subdirectory(files_test, test_dir)
