import os
import cv2
import glob

import numpy as np
import tensorflow as tf

from flow_utils import read_flow
from absl import app, flags, logging

# FLAGS = flags.FLAGS

# flags.DEFINE_string('data_dir', '/mnt/WindowsDev/DataSets/FlyingChairs/data/', 'Link to dataset directory.')
# flags.DEFINE_string('train_list',  "/mnt/WindowsDev/DataSets/FlyingChairs/train_0.03split.txt", 'Link to validation list.')
# flags.DEFINE_string("val_list", "/mnt/WindowsDev/DataSets/FlyingChairs/val_0.03split.txt", "Link to validation list.")

# flags.DEFINE_list('losses_weight', [0.32, 0.08, 0.02, 0.01, 0.005], 'Loss weights for 6th to 2nd flow predictions, as described in the original paper.')
# flags.DEFINE_float('gamma', 0.0004, None)
# flags.DEFINE_integer('batch_size', 1, None)
# flags.DEFINE_enum('dataset', 'mixed', ['mixed', 'chairs', 'things3d_ft'], None)

# # Learing schedule for training from scratch. Train on the mixed dataset of FlyingChairs and FlyingThings3D.
# flags.DEFINE_list('lr_boundaries', [400000, 600000, 800000, 1000000], None)
# flags.DEFINE_float('lr', 0.0001, None)
# flags.DEFINE_integer('num_steps', 1500000, None)
# flags.DEFINE_list('crop_size', [240, 240], None)

# flags.DEFINE_integer('steps_per_save', 10000, None)
# flags.DEFINE_integer('steps_per_eval', 1000, None)
# flags.DEFINE_integer('log_freq', 50, None)

# flags.DEFINE_boolean('random_scale', False, 'Random scale.')
# flags.DEFINE_boolean('random_flip', False, 'Random flip.') 


_FLYINGCHAIRS = 0
_FLYINGTHINGS3D = 1

_FLYINGCHAIRS_WIDTH = 512
_FLYINGCHAIRS_HEIGHT = 384

_FLYINGTHINGS3D_WIDTH = 480
_FLYINGTHINGS3D_HEIGHT = 270

def read_list(data_dir, data_list):
    fp = open(data_list, 'r')
    line = fp.readline()

    im1_filenames, im2_filenames, flo_filenames = [], [], []
    while line:
        for chunck in line.replace('\n', '').split('###'):
            chunck_s = chunck.split(".")
            if chunck_s[0].endswith("img1"):
                im1_fn = chunck
            elif chunck_s[0].endswith("img2"):
                im2_fn = chunck
            else:
                flo_fn = chunck
                
        im1_filenames.append(os.path.join(data_dir, im1_fn))
        im2_filenames.append(os.path.join(data_dir, im2_fn))
        flo_filenames.append(os.path.join(data_dir, flo_fn))
        
        line = fp.readline()
        
    fp.close()

    return im1_filenames, im2_filenames, flo_filenames

def _parse_function(im1_filename, im2_filename, flo_filename):
    im1_fn_decoded = im1_filename.numpy().decode("utf-8")
    im2_fn_decoded = im2_filename.numpy().decode("utf-8")
    flo_fn_decoded = flo_filename.numpy().decode("utf-8")
    
    im1 = cv2.imread(im1_fn_decoded)
    im2 = cv2.imread(im2_fn_decoded)
    flo = read_flow(flo_fn_decoded)

    mode = _FLYINGCHAIRS
    if 'FlyingThings3D' in im1_fn_decoded: 
        '''
        Use half-resolution, as suggested by Philferrier, 2018. 
        Please refer to the details in - 'https://github.com/NVlabs/PWC-Net/issues/44'
                                       - 'https://github.com/philferriere/tfoptflow#multisteps-learning-rate-schedule-'
        '''
        new_w, new_h = int(im1.shape[1] * 0.5), int(im1.shape[0] * 0.5)
        im1 = cv2.resize(im1, (new_w, new_h))
        im2 = cv2.resize(im2, (new_w, new_h))
        flo = cv2.resize(flo, (new_w, new_h)) * 0.5

        mode = _FLYINGTHINGS3D

    return im1, im2, flo, mode

def tf_parse_function(im1_filename, im2_filename, flo_filename):
    [im1, im2, flo, mode] = tf.py_function(_parse_function, 
                                    inp=[im1_filename, im2_filename, flo_filename], 
                                    Tout=[tf.uint8, tf.uint8, tf.float32, tf.uint8])
    
    im1 = tf.cast(im1, tf.float32) / 255.0
    im2 = tf.cast(im2, tf.float32) / 255.0

    return im1, im2, flo, mode

def tf_image_flip_ud(im1, im2, flo):
    distort_up_down_random = tf.random.uniform([1], 0, 1.0, dtype=tf.float32)[0]
    flip = tf.less(distort_up_down_random, 0.5)
    flip_mask = tf.stack([flip, False, False])
    flip_axis = tf.boolean_mask([0, 1, 2], flip_mask)
    
    im1 = tf.reverse(im1, flip_axis)
    im2 = tf.reverse(im2, flip_axis)
    flo = tf.reverse(flo, flip_axis)
    
    if flip: 
        v = flo[:, :, 1:] * -1
    else: 
        v = flo[:, :, 1:]
        
    u = flo[:, :, :1] 
    flo = tf.concat([u, v], axis=2)
        
    return im1, im2, flo

def tf_image_flip_lr(im1, im2, flo):
    distort_left_right_random = tf.random.uniform([1], 0, 1.0, dtype=tf.float32)[0]
    flip = tf.less(distort_left_right_random, 0.5)
    flip_mask = tf.stack([False, flip, False])
    flip_axis = tf.boolean_mask([0, 1, 2], flip_mask)
    
    im1 = tf.reverse(im1, flip_axis)
    im2 = tf.reverse(im2, flip_axis)
    flo = tf.reverse(flo, flip_axis)
    
    if flip: 
        u = flo[:, :, :1] * -1
    else: 
        u = flo[:, :, :1]
        
    v = flo[:, :, 1:] 
    flo = tf.concat([u, v], axis=2)
        
    return im1, im2, flo

def tf_image_scale_and_crop(im1, im2, flo, mode, flags):    
    im_concat = tf.concat([im1, im2, flo], axis=2)
    
    scale = tf.random.uniform([1], minval=0.955, maxval=1.05, dtype=tf.float32, seed=None)

    h_new_chairs = tf.multiply(tf.cast(_FLYINGCHAIRS_HEIGHT, tf.float32), scale)[0]
    w_new_chairs = tf.multiply(tf.cast(_FLYINGCHAIRS_WIDTH, tf.float32), scale)[0]

    h_new_things =  tf.multiply(tf.cast(_FLYINGTHINGS3D_HEIGHT, tf.float32), scale)[0]
    w_new_things =  tf.multiply(tf.cast(_FLYINGTHINGS3D_WIDTH, tf.float32), scale)[0]

    new_shape_chairs = tf.cast([h_new_chairs, w_new_chairs], tf.int32)
    new_shape_things = tf.cast([h_new_things, w_new_things], tf.int32)

    crop_size = flags.crop_size
    if flags.dataset == 'mixed':
        im_padded = tf.image.pad_to_bounding_box(
                                im_concat,
                                0,
                                0,
                                _FLYINGCHAIRS_HEIGHT,
                                _FLYINGCHAIRS_WIDTH)

        im_resized = tf.image.resize(im_padded, new_shape_chairs, method=tf.image.ResizeMethod.BILINEAR)

        if tf.equal(mode, _FLYINGTHINGS3D): # Avoid to crop on the padded area.
            h_border = tf.cast(h_new_things, tf.int32)
            w_border = tf.cast(w_new_things, tf.int32)

            h_offset = tf.random.uniform([1], minval=0, maxval=h_border-crop_size[0], dtype=tf.int32, seed=None)[0]
            w_offset = tf.random.uniform([1], minval=0, maxval=w_border-crop_size[1], dtype=tf.int32, seed=None)[0]

            im_cropped = tf.image.crop_to_bounding_box(im_resized, h_offset, w_offset, crop_size[0], crop_size[1])
        else:
            im_cropped = tf.image.random_crop(im_resized, [crop_size[0], crop_size[1], 8])

    elif flags.dataset == 'things3d_ft':
        im_concat.set_shape((_FLYINGTHINGS3D_HEIGHT, _FLYINGTHINGS3D_WIDTH, 8))

        im_resized = tf.image.resize(im_concat, new_shape_things, method=tf.image.ResizeMethod.BILINEAR)
        im_cropped = tf.image.random_crop(im_resized, [crop_size[0], crop_size[1], 8])

    im1 = im_cropped[:, :, :3]
    im2 = im_cropped[:, :, 3:6]
    flo = im_cropped[:, :, 6:]
    flo = flo * scale

    return im1, im2, flo

def tf_image_crop(im1, im2, flo, mode, crop_size):
    im_concat = tf.concat([im1, im2, flo], axis=2)
    im_cropped = tf.image.random_crop(im_concat, [crop_size[0], crop_size[1], 8]) # RGB + RGB + UV = 8 channels
    
    im1 = im_cropped[:, :, :3]
    im2 = im_cropped[:, :, 3:6]
    flo = im_cropped[:, :, 6:]

    return im1, im2, flo

class DataLoader(object):
    '''
    Generic data loader which reads images and corresponding flow ground truth (.flo file)
    from the disk, and enqueues them into a TensorFlow queue using tf.Dataset API.
    '''

    def __init__(self, data_dir, train_list=None, val_list=None):
        self.train_list = train_list
        self.val_list = val_list

        if train_list:
            self.train_im1_ls, self.train_im2_ls, self.train_flo_ls = read_list(data_dir, train_list)
            self.train_size = len(self.train_im1_ls)

        if val_list:
            self.val_im1_ls, self.val_im2_ls, self.val_flo_ls = read_list(data_dir, val_list)
            self.val_size = len(self.val_im1_ls)
    
    def create_tf_dataset(self, flags):
        train_dataset, val_dataset = None, None

        if self.train_list:
            '''Prepare for training dataset'''
            train_dataset = tf.data.Dataset.from_tensor_slices((self.train_im1_ls, self.train_im2_ls, self.train_flo_ls))
            train_dataset = train_dataset.map(lambda x, y, z: 
                                            tf_parse_function(x, y, z), num_parallel_calls=8)
            
            # Preprocessing part
            if flags.random_scale:
                train_dataset = train_dataset.map(lambda x, y, z, v: 
                                                    tf_image_scale_and_crop(x, y, z, v, flags), num_parallel_calls=8)
            else:
                train_dataset = train_dataset.map(lambda x, y, z, v: 
                                                tf_image_crop(x, y, z, v, flags.crop_size), num_parallel_calls=8)

            if flags.random_flip:
                train_dataset = train_dataset.map(tf_image_flip_ud, num_parallel_calls=8)
                train_dataset = train_dataset.map(tf_image_flip_lr, num_parallel_calls=8)

            train_dataset = train_dataset.map(lambda x, y, z: [tf.concat([x, y], axis=2), z], num_parallel_calls=8)

            # train_dataset = train_dataset.shuffle(buffer_size=5000)
            train_dataset = train_dataset.batch(flags.batch_size, drop_remainder=True)
        
        if self.val_list:
            '''Prepare for validation dataset'''
            val_dataset = tf.data.Dataset.from_tensor_slices((self.val_im1_ls, self.val_im2_ls, self.val_flo_ls))
            val_dataset = val_dataset.map(lambda x, y, z: 
                                            tf_parse_function(x, y, z), num_parallel_calls=8)
            val_dataset = val_dataset.map(lambda x, y, z, v: [tf.concat([x, y], axis=2), z], num_parallel_calls=8)
            val_dataset = val_dataset.batch(1)
        
        return train_dataset, val_dataset


# def main(argv):
#     import matplotlib.pyplot as plt
#     data_gen = DataLoader(FLAGS.data_dir, FLAGS.train_list, FLAGS.val_list)
#     train_dataset, val_dataset = data_gen.create_tf_dataset(FLAGS)
#     for im_pairs, flo_gt in train_dataset:
#         im1 = im_pairs[:, :, :, :3]
#         im2 = im_pairs[:, :, :, 3:]
#         plt.imshow(im1[0].numpy())
#         plt.show()
#         break

# if __name__ == '__main__':
#     app.run(main)

    

    
    
