import h5py
import pprint

def pprint_h5_file_contents(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print("Contents of HDF5 file:", file_path)
            pprint.pprint(list(f.keys()))
            for key in f.keys():
                print("---------------------------------------------------------")
                print("Dataset:", key)
                print("---------------------------------------------------------")
                print("Attributes:")
                for attr_key, attr_value in f[key].attrs.items():
                    print(f"{attr_key}: {attr_value}")
                print("---------------------------------------------------------")
                print("Data:")
                print(f[key][...])  # Access data
                print("---------------------------------------------------------")
    except Exception as e:
        print("Error:", e)

# Example usage:
file_path = "./prepro_data/demo/resnet14x14.h5"
pprint_h5_file_contents(file_path)
