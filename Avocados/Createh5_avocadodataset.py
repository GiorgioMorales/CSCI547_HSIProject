from random import shuffle
import glob
import os
import numpy as np
import h5py
import pickle
import gdal

#####################################################################
#   Create list of image addresses and labels
#####################################################################

# Get dataset foler
basepath = os.getcwd()[:-13 - 14]
orig_path_train = basepath + '\\KochiaHyperspectralImages\\DatasetMerge\\*'

# Get the list of addresses
addri = sorted(glob.glob(orig_path_train))
labels = []

# Set classes
for addr in addri:
    yc = 0
    if 'Fresh' in addr:
        yc = 0
    elif 'NotF' in addr:
        yc = 1
    labels.append(yc)

# Shuffle data
c = list(zip(addri, labels))
shuffle(c)
addri, labels = zip(*c)

# Split training and validation sets
train_addrs = addri[0:int(1*len(addri))]
train_labels = labels[0:int(1*len(addri))]

#####################################################################
#   CREATE HDF5 FILE
#####################################################################

# Set image size
train_shape = (len(train_addrs), 64, 64, 150)

# Create .hdf5
hdf5_path = 'avocado_dataset_w64.hdf5'  # address to where you want to save the hdf5 file
hdf5_file = h5py.File(hdf5_path, mode='w')

# Create "train" and "test" fields and their sizes
hdf5_file.create_dataset("train_img", train_shape, np.float16)
hdf5_file.create_dataset("train_labels", (len(train_labels),), np.int8)

# Save labels
hdf5_file["train_labels"][...] = train_labels

#####################################################################
#   READ AND SAVE IMAGES
#####################################################################

for i in range(len(train_addrs)):

    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    # Read images

    addr = train_addrs[i]
    # Read image
    ds = gdal.Open(addr)
    rb = ds.GetRasterBand(1)
    img_array = rb.ReadAsArray()
    img = np.zeros((img_array.shape[0], img_array.shape[1], 300))
    img[:, :, 0] = img_array
    for ind in range(1, 300):
        rb = ds.GetRasterBand(ind + 1)
        img_array = rb.ReadAsArray()
        img[:, :, ind] = img_array

    # I am averaging consecutive bands, so instead of 300 we will have 150 (avg(1,2), avg (3,4), avg(4,5), ...)
    img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 2)))
    for n in range(0, img.shape[0]):
        for band in range(0, img.shape[3], 2):
            img2[n, :, :, int(band / 2)] = (img[n, :, :, band] + img[n, :, :, band + 1]) / 2.
    img = img2

    # Reshape image to 64 x 64 x 150
    #reshaped_image =

    hdf5_file["train_img"][i, ...] = reshaped_image

hdf5_file.close()
