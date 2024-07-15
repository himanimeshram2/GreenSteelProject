import unittest
import h5py
import numpy as np
import os

from IO.hdf5_utilities import load_hdf5_data, get_hdf5_id_map  # Updated import if in the IO directory

class TestHdf5Utilities(unittest.TestCase):
    def setUp(self):
        """Setup a temporary HDF5 file for testing."""
        self.filepath = 'test_file.h5'
        self.data = np.arange(10)
        with h5py.File(self.filepath, 'w') as f:
            f.create_dataset('test_dataset', data=self.data)

    def test_load_hdf5_data(self):
        """Test loading data from an HDF5 file."""
        try:
            loaded_data = load_hdf5_data(self.filepath, 'test_dataset')
            np.testing.assert_array_equal(loaded_data, self.data)
        except Exception as e:
            self.fail(f"Loading data failed with an unexpected error: {e}")

    def test_get_hdf5_id_map(self):
        """Test getting ID map from an HDF5 file."""
        try:
            id_map = get_hdf5_id_map(self.filepath)
            self.assertIn('test_dataset', id_map.keys())
        except Exception as e:
            self.fail(f"Getting ID map failed with an unexpected error: {e}")

    def test_file_not_found(self):
        """Test behavior when the file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_hdf5_data('nonexistent_file.h5', 'dataset')

    def tearDown(self):
        """Clean up by removing the temporary file after each test."""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass  # File was not created or was already deleted, ignore

if __name__ == '__main__':
    unittest.main()
