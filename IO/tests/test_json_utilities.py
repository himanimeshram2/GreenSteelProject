import unittest
import json
import os
from IO.json_utilities import load_json_file, save_data_to_json_file  # Adjusted import path

class TestJsonUtilities(unittest.TestCase):
    def setUp(self):
        """Setup a temporary JSON file for testing."""
        self.test_data = {'key': 'value'}
        self.filepath = 'test_file.json'
        with open(self.filepath, 'w') as f:
            json.dump(self.test_data, f)

    def test_load_json_file(self):
        """Test loading JSON data from a file."""
        data = load_json_file(self.filepath)
        self.assertEqual(data, self.test_data)

    def test_save_data_to_json_file(self):
        """Test saving JSON data to a file."""
        new_data = {'new_key': 'new_value'}
        save_data_to_json_file(new_data, self.filepath)
        with open(self.filepath, 'r') as f:
            data_loaded = json.load(f)
        self.assertEqual(data_loaded, new_data)

    def test_load_nonexistent_file(self):
        """Test loading JSON data from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_json_file('nonexistent.json')

    def test_corrupted_json_file(self):
        """Test behavior when the JSON data is corrupted."""
        with open(self.filepath, 'w') as f:
            f.write("{bad json")  # Write corrupted JSON
        with self.assertRaises(json.JSONDecodeError):
            load_json_file(self.filepath)

    def tearDown(self):
        """Clean up by removing the temporary file after each test."""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()
