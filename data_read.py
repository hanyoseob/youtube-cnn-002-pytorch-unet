##
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

##
dir_dataset = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_dataset, name_label))
img_input = Image.open(os.path.join(dir_dataset, name_input))

ny, nx = img_label.size
nframe = img_input.n_frames

##
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_dataset, 'train')
dir_save_val = os.path.join(dir_dataset, 'val')
dir_save_test = os.path.join(dir_dataset, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(i + offset_nframe)
    img_input.seek(i + offset_nframe)

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(i + offset_nframe)
    img_input.seek(i + offset_nframe)

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(i + offset_nframe)
    img_input.seek(i + offset_nframe)

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
