import h5py

def get_hdf5_id_map(filepath):
    try:
        with h5py.File(filepath, 'r') as file:
            return {name: file[name].id for name in file.keys()}
    except Exception as e:
        print(f"Failed to read HDF5 file {filepath}: {e}")
        return None

def load_hdf5_data(filepath, dataset_name):
    try:
        with h5py.File(filepath, 'r') as file:
            return file[dataset_name][:]
    except Exception as e:
        print(f"Failed to load dataset {dataset_name} from {filepath}: {e}")
        return None
