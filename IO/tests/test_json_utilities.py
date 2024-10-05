import unittest
import os
from json_utilities import load_json_file, save_data_to_json_file

class TestJSONUtilities(unittest.TestCase):

    def setUp(self):
        self.filepath = 'test_file.json'
        self.data = {"name": "John Doe", "age": 30}

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_load_json_file(self):
        """Test loading data from a JSON file."""
        save_data_to_json_file(self.data, self.filepath)
        loaded_data = load_json_file(self.filepath)
        self.assertEqual(loaded_data, self.data)

    def test_large_json_file(self):
        """Test loading a large JSON file."""
        large_data = {'key': ['value'] * 1000000}  # Large JSON structure
        save_data_to_json_file(large_data, self.filepath)
        
        loaded_data = load_json_file(self.filepath)
        self.assertEqual(loaded_data['key'][0], 'value')
        self.assertEqual(len(loaded_data['key']), 1000000)

    def test_save_data_to_json_file(self):
        """Test saving data to a JSON file."""
        save_data_to_json_file(self.data, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_malformed_json(self):
        """Test behavior with a malformed JSON file."""
        with open(self.filepath, 'w') as file:
            file.write('{"name": "John Doe", "age": 30,}')  # Malformed JSON (trailing comma)
        
        loaded_data = load_json_file(self.filepath)
        self.assertIsNone(loaded_data)  # Should return None due to JSONDecodeError

if __name__ == '__main__':
    unittest.main()
