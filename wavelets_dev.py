
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from json import load
import cv2

import tensorflow_wavelets.Layers.DMWT as DMWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.utils.helpers as helper
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from OpenDVCW import OpenDVC, OpticalFlowLoss, OpticalFlow
from tensorflow.keras.layers import AveragePooling2D, Conv2D
import OpenDVCW


def split_wt_to_lllhhlhh(data):
    ll = tf.split(tf.split(data, 2, axis=1)[0], 2, axis=2)[0]
    lh = tf.split(tf.split(data, 2, axis=1)[0], 2, axis=2)[1]
    hl = tf.split(tf.split(data, 2, axis=1)[1], 2, axis=2)[0]
    hh = tf.split(tf.split(data, 2, axis=1)[1], 2, axis=2)[1]
    return [ll, lh, hl, hh]


class WaveletsOpticalFlow(tf.keras.layers.Layer):
    """ 
    """
    def __init__(self, batch_size, width, height,  **kwargs):
        super(WaveletsOpticalFlow, self).__init__(**kwargs)
        self.optic_loss = OpticalFlowLoss()
        self.optic_flow = OpticalFlow(batch_size, width, height)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.dwt_db2 = DWT.DWT("haar", concat=1)

    def call(self, inputs, training=None, mask=None):
        
        im1 = inputs[0]
        im2 = inputs[1]

        im1_dwt_l1 = self.dwt_db2(im1)
        im2_dwt_l1 = self.dwt_db2(im2)
 
        [im1_3, lh1_l1, hl1_l1, hh1_l1] = split_wt_to_lllhhlhh(im1_dwt_l1)
        [im2_3, lh2_l1, hl2_l1, hh2_l1] = split_wt_to_lllhhlhh(im2_dwt_l1)

        im1_dwt_l2 = self.dwt_db2(im1_3)
        im2_dwt_l2 = self.dwt_db2(im2_3)

        [im1_2, lh1_l2, hl1_l2, hh1_l2] = split_wt_to_lllhhlhh(im1_dwt_l2)
        [im2_2, lh2_l2, hl2_l2, hh2_l2] = split_wt_to_lllhhlhh(im2_dwt_l2)

        im1_dwt_l3 = self.dwt_db2(im1_2)
        im2_dwt_l3 = self.dwt_db2(im2_2)

        [im1_1, lh1_l2, hl1_l2, hh1_l2] = split_wt_to_lllhhlhh(im1_dwt_l3)
        [im2_1, lh2_l2, hl2_l2, hh2_l2] = split_wt_to_lllhhlhh(im2_dwt_l3)

        im1_dwt_l4 = self.dwt_db2(im1_1)
        im2_dwt_l4 = self.dwt_db2(im2_1)

        [im1_0, lh1_l2, hl1_l2, hh1_l2] = split_wt_to_lllhhlhh(im1_dwt_l4)
        [im2_0, lh2_l2, hl2_l2, hh2_l2] = split_wt_to_lllhhlhh(im2_dwt_l4)

        flow_zero = tf.zeros((self.batch_size, self.width//2, self.height//2, 2), dtype=tf.float32)

        loss_0, flow_0 = self.optic_loss([flow_zero, im1_0, im2_0])
        loss_1, flow_1 = self.optic_loss([flow_0, im1_1, im2_1])
        loss_2, flow_2 = self.optic_loss([flow_1, im1_2, im2_2])
        loss_3, flow_3 = self.optic_loss([flow_2, im1_3, im2_3])

        return flow_3



class OF_Test(tf.keras.Model):
    """Main model class."""

    def __init__(self, width=240, height=240, batch_size=1):
        super(OF_Test, self).__init__()
        self.optical_flow = WaveletsOpticalFlow(batch_size, width, height)

    def call(self, x, training):
        """Computes rate and distortion losses."""
        
        # Reference frame frame
        Y0_com = tf.cast(x[0], dtype=tf.float32)
        # current frame
        Y1_raw = tf.cast(x[1], dtype=tf.float32)

        return self.optical_flow([Y0_com, Y1_raw])



if __name__ == "__main__":

    # lena = cv2.imread("tensorflow-wavelets/Development/input/Lenna_orig.png")
    # lena_res = cv2.resize(lena, (240, 240),  interpolation=cv2.INTER_AREA)
    # lena_res = np.float32(np.expand_dims(lena_res, axis=0))
    # x = DMWT.DMWT("ghm")
    # y = x(lena_res)
    # y = tf.image.convert_image_dtype(y[0, ... ], dtype=tf.float32)
    # y = y.numpy()
    # data = y.astype(np.uint8)
    # cv2.imshow("orig", data)
    # cv2.waitKey(0)

    # paths = np.load("/workspaces/OpenDVCW/folder_cloud_test.npy") 
    # path = paths[np.random.randint(len(paths))] + '/'
    # print(path)
    Height = 240
    Width = 240
    inp1 = "/mnt/WindowsDev/DataSets/vimeo_septuplet/sequences/00012/0126/im1.png"
    inp2 = "/mnt/WindowsDev/DataSets/vimeo_septuplet/sequences/00012/0126/im2.png"

    # img1 = cv2.imread(inp1)
    # img2 = cv2.imread(inp2)
    # img1 = cv2.resize(img1, (Height, Width))
    # img2 = cv2.resize(img2, (Height, Width))

    img1 = tf.expand_dims(OpenDVCW.read_png_crop(inp1, Height, Width), 0)
    img2 = tf.expand_dims(OpenDVCW.read_png_crop(inp2, Height, Width), 0)

    model = OF_Test(width=Width, height=Height, batch_size=1)
    loss_0  = model([img1, img2])
    print(loss_0.shape)



