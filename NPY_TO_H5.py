# NPY_TO_H5.py is a script to convert numpy arrays to hdf5 format.

import h5py
import numpy as np

# Define the path of the numpy array file and the hdf5 file to be created
npy_file = '/home/wzn/projects/senceiver/data/npy/train_data.npy'
h5_file = '/home/wzn/projects/senceiver/data/h5/train_data.h5'

# Load the numpy array
data = np.load(npy_file)

# Create the hdf5 file
with h5py.File(h5_file, 'w') as f:
    # Create a dataset in the hdf5 file
    dset = f.create_dataset('data', data=data)

# Print the shape of the numpy array and the hdf5 file
print('Shape of numpy array:', data.shape)
with h5py.File(h5_file, 'r') as f:
    print('Shape of hdf5 file:', f['data'].shape)