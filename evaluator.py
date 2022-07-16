import os
import cv2
import numpy as np
import tensorflow as tf
from math import log10, sqrt
import OpenDVCW


class Evaluator(object):
    def __init__(self, bs=1, height=240, width=240, channels=3,
     input_seq_path="", workdir="", model_list=[], num_of_p_frames=7,
      prefix="img", suffix=".png", bin_suffix=".bin", decom_prefix="decom") -> None:
        # bach size
        self.bs = bs
        self.height = height
        self.width = width
        self.channels = channels
        # input video sequence frames
        self.input_seq_path = input_seq_path
        self.workdir = workdir
        # tensorflow models paths
        self.model_list = model_list
        self.num_of_p_frames = num_of_p_frames

        self.suffix = suffix
        self.prefix = prefix

        self.bin_suffix = bin_suffix
        self.decom_prefix = decom_prefix

        self.res = dict()

    def PSNR(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def compare(self, original_path, compressed_path, binary_path):
        original = cv2.imread(original_path)
        compressed = cv2.imread(compressed_path)
        bin_size = os.path.getsize(binary_path)
        value = self.PSNR(original, compressed)
        return {"bin_size": bin_size , "psnr" : value , "BPP" : bin_size/( self.height*self.width*self.channels)}

    def test_tf_model(self, model, iframe, p_on_test, out_decom, p_frame_out_bin):
        OpenDVCW.compress(model, iframe, p_on_test, p_frame_out_bin, self.height, self.width)
        OpenDVCW.decompress(model, iframe, p_frame_out_bin, out_decom,  self.height, self.width)
        return self.compare(p_on_test, out_decom, p_frame_out_bin)
    
    def eval(self):
        print("Evaluation Started ...")
        i_frame = os.path.join(self.input_seq_path, self.prefix + str(0) + self.suffix)
        for model in self.model_list:
            print("working on model", model)
            iter_res = []
            model =  tf.keras.models.load_model(model)
            for idx in np.arange(1, self.num_of_p_frames+1):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_bin = os.path.join(self.workdir, self.prefix + str(idx) + self.bin_suffix)
                p_out_decom = os.path.join(self.workdir, self.decom_prefix + str(idx) + self.suffix)
                iter_res.append(self.test_tf_model(model, i_frame, p_frame, p_out_decom, p_frame_out_bin))   

            self.res[model] = iter_res             
