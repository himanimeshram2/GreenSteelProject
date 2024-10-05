import unittest
import os
import h5py
from hdf5_utilities import load_hdf5_data, get_hdf5_id_map

class TestHDF5Utilities(unittest.TestCase):

    def setUp(self):
        self.filepath = 'test_file.hdf5'
        self.dataset_name = 'test_dataset'
        with h5py.File(self.filepath, 'w') as f:
            f.create_dataset(self.dataset_name, data=[1, 2, 3, 4, 5])

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_load_hdf5_data(self):
        """Test loading data from an HDF5 dataset."""
        data = load_hdf5_data(self.filepath, self.dataset_name)
        self.assertEqual(list(data), [1, 2, 3, 4, 5])

    def test_empty_dataset(self):
        """Test loading an empty dataset from an HDF5 file."""
        with h5py.File(self.filepath, 'w') as f:
            f.create_dataset('empty_dataset', data=[])

        data = load_hdf5_data(self.filepath, 'empty_dataset')
        self.assertEqual(len(data), 0)

    def test_missing_dataset(self):
        """Test behavior when the dataset is missing."""
        data = load_hdf5_data(self.filepath, 'non_existent_dataset')
        self.assertIsNone(data)

    def test_get_hdf5_id_map(self):
        """Test retrieving ID map from an HDF5 file."""
        id_map = get_hdf5_id_map(self.filepath)
        self.assertIn(self.dataset_name, id_map)

if __name__ == '__main__':
    unittest.main()
