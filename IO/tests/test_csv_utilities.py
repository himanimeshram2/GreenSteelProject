import unittest
import os
from csv_utilities import load_csv_file, save_data_to_csv_file

class TestCSVUtilities(unittest.TestCase):

    def setUp(self):
        self.filepath = 'test_file.csv'
        self.empty_filepath = 'empty_test_file.csv'
        self.data = [['Name', 'Age'], ['John Doe', '30'], ['Jane Smith', '25']]

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        if os.path.exists(self.empty_filepath):
            os.remove(self.empty_filepath)

    def test_load_csv_file(self):
        """Test loading a CSV file."""
        save_data_to_csv_file(self.data, self.filepath)
        loaded_data = load_csv_file(self.filepath)
        self.assertEqual(loaded_data, self.data)

    def test_empty_csv_file(self):
        """Test behavior when the CSV file is empty."""
        with open(self.empty_filepath, 'w', newline='') as file:
            pass  # Create an empty file

        data = load_csv_file(self.empty_filepath)
        self.assertEqual(data, [])  # Expect an empty list for an empty CSV

    def test_save_data_to_csv_file(self):
        """Test saving data to a CSV file."""
        save_data_to_csv_file(self.data, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_malformed_csv_file(self):
        """Test behavior with a malformed CSV file."""
        with open(self.filepath, 'w', newline='') as file:
            file.write("Name,Age\nJohn Doe,30\nJane Smith")  # Malformed: Missing age for Jane

        data = load_csv_file(self.filepath)
        self.assertEqual(len(data), 2)  # It should still read the two lines
        self.assertEqual(data[1], ['Jane Smith'])  # The malformed line is partially read

if __name__ == '__main__':
    unittest.main()
