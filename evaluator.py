import os
import cv2
import numpy as np
import tensorflow as tf
from math import log10, sqrt
import OpenDVCW
import subprocess
import matplotlib

import matplotlib.pyplot as plt

def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Evaluator(object):
    def __init__(self, bs=1, height=240, width=240, channels=3,
                input_seq_path="", workdir="", model_list=[], num_of_p_frames=5,
                prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom",
                tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave",
                h264_workdir="", h264_bitrate_list=[50e3, 100e3, 200e3, 400e3, 1e5],
                h265_workdir="", h265_bitrate_list=[50e3, 100e3, 200e3, 400e3, 1e5]) -> None:
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

        self.tave_path = tave_path
        self.h264_workdir = h264_workdir
        self.h264_bitrate_list = h264_bitrate_list

        self.h265_workdir = h265_workdir
        self.h265_bitrate_list = h265_bitrate_list

        path_exists(h264_workdir)
        path_exists(h265_workdir)
        path_exists(workdir)

        self.encoded_h264 = r"/output_joined.h264"
        self.encoded_h265 = r"/output_joined.h265"
        self.res = dict()
        self.h264_res = dict()
        self.h265_res = dict()

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
        for model_name in self.model_list:
            print("working on model", model_name)
            iter_res = []

            model =  tf.keras.models.load_model(model_name)
            for idx in np.arange(1, self.num_of_p_frames+1):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_bin = os.path.join(self.workdir, self.prefix + str(idx) + self.bin_suffix)
                p_out_decom = os.path.join(self.workdir, self.decom_prefix + str(idx) + self.suffix)
                iter_res.append(self.test_tf_model(model, i_frame, p_frame, p_out_decom, p_frame_out_bin))   

            self.res[model_name] = iter_res   

        print("H264 Evaluation ...")          
        for bit_rate in self.h264_bitrate_list:
            self.h264_test(bit_rate)

        print("H265 Evaluation ...")          
        for bit_rate in self.h265_bitrate_list:
            self.h265_test(bit_rate)

    def h264_test(self, bit_rate):
        command = [self.tave_path, "libx264", str(bit_rate), self.input_seq_path, self.prefix, self.suffix, self.h264_workdir, str(self.num_of_p_frames)]
        print("Command: ", command)
        subprocess.run(command)

        command = ["ffmpeg", "-i", self.h264_workdir + self.encoded_h264, self.h264_workdir + r"/decoded_%04d.png"]

        subprocess.run(command)
        res = []
        for idx in np.arange(1, self.num_of_p_frames):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_decom = os.path.join(self.h264_workdir, "decoded_" + str(idx).zfill(4) + self.suffix)
                p_frame_out_bin = os.path.join(self.h264_workdir, str(idx)+".h264")
                res.append(self.compare(p_frame, p_frame_out_decom, p_frame_out_bin))

        self.h264_res[bit_rate] = res

    def h265_test(self, bit_rate):
        command = [self.tave_path, "libx265", str(bit_rate), self.input_seq_path, self.prefix, self.suffix, self.h265_workdir, str(self.num_of_p_frames)]
        print("Command: ", command)
        subprocess.run(command)

        command = ["ffmpeg", "-i", self.h265_workdir + self.encoded_h265, self.h265_workdir + r"/decoded_%04d.png"]

        subprocess.run(command)
        res = []
        for idx in np.arange(1, self.num_of_p_frames):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_decom = os.path.join(self.h265_workdir, "decoded_" + str(idx).zfill(4) + self.suffix)
                p_frame_out_bin = os.path.join(self.h265_workdir, str(idx)+".h265")
                res.append(self.compare(p_frame, p_frame_out_decom, p_frame_out_bin))

        self.h265_res[bit_rate] = res

    def plot_graph(self, fig_name="", proposed_labhel="Proposed-Haar"):
        font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
        matplotlib.rc('font', **font)
        LineWidth = 3

        our_bpp = []
        our_psnr = []
        
        for model_name in self.model_list:
            avg_bpp = 0
            avg_psnr = 0
            for iter in range(self.num_of_p_frames):
                avg_bpp += self.res[model_name][iter]["BPP"]
                avg_psnr += self.res[model_name][iter]["psnr"]
            our_bpp.append(avg_bpp/self.num_of_p_frames)
            our_psnr.append(avg_psnr/self.num_of_p_frames)

            
        ours, = plt.plot(our_bpp, our_psnr, "k-o", linewidth=LineWidth, label=proposed_labhel)

        # H.264 
        h264_bpp = []
        h264_psnr = []
        for bit_rate in self.h264_bitrate_list:
            avg_bpp = 0
            avg_psnr = 0
            for iter in range(self.num_of_p_frames-1):
                avg_bpp += self.h264_res[bit_rate][iter]["BPP"]
                avg_psnr += self.h264_res[bit_rate][iter]["psnr"]
            h264_bpp.append(avg_bpp/self.num_of_p_frames)
            h264_psnr.append(avg_bpp/self.num_of_p_frames)
        
        h265_bpp = []
        h265_psnr = []
        for bit_rate in self.h265_bitrate_list:
            avg_bpp = 0
            avg_psnr = 0
            for iter in range(self.num_of_p_frames-1):
                avg_bpp += self.h265_res[bit_rate][iter]["BPP"]
                avg_psnr += self.h265_res[bit_rate][iter]["psnr"]
            h265_bpp.append(avg_bpp/self.num_of_p_frames)
            h265_psnr.append(avg_bpp/self.num_of_p_frames)

        h264, = plt.plot(h264_bpp, h264_psnr, "m--s", linewidth=LineWidth, label='H.264')
        h265, = plt.plot(h265_bpp, h265_psnr, "r--v", linewidth=LineWidth, label='H.265')

        plt.legend(handles=[h264, h265, ours], loc=4)
        plt.grid()
        plt.xlabel('BPP')
        plt.ylabel('PSNR(dB)')
        plt.title('')
        if fig_name != "":
            plt.savefig(fig_name, format='eps', dpi=600, bbox_inches='tight')   