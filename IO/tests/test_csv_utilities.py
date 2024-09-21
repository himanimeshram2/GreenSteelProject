import unittest
import os
import csv
from IO.csv_utilities import load_csv_file, save_data_to_csv_file

class TestCsvUtilities(unittest.TestCase):
    def setUp(self):
        """Set up a temporary CSV file for testing."""
        self.test_data = [['header1', 'header2'], ['row1col1', 'row1col2'], ['row2col1', 'row2col2']]
        self.filepath = 'test_file.csv'
        save_data_to_csv_file(self.test_data, self.filepath)

    def test_load_csv_file(self):
        """Test loading data from a CSV file."""
        data = load_csv_file(self.filepath)
        self.assertEqual(data, self.test_data)

    def test_save_data_to_csv_file(self):
        """Test saving data to a CSV file."""
        new_data = [['header1', 'header2'], ['newrow1col1', 'newrow1col2']]
        save_data_to_csv_file(new_data, self.filepath)
        loaded_data = load_csv_file(self.filepath)
        self.assertEqual(loaded_data, new_data)

    def tearDown(self):
        """Clean up by removing the temporary file after each test."""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()

