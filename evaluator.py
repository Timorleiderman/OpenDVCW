import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from math import log10, sqrt
import OpenDVCW
import subprocess
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(evals, fig_name="", title=''):
    font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
    matplotlib.rc('font', **font)
    line_width = 3

    for eva in evals:
        plt_eval_bpp = []
        plt_eval_psnr = []

        for model_name in eva.model_list:
            avg_bpp = 0
            avg_psnr = 0
            for iter in range(eva.num_of_p_frames):
                avg_bpp += eva.res[model_name][iter]["BPP"]
                avg_psnr += eva.res[model_name][iter]["psnr"]
            plt_eval_bpp.append(avg_bpp / eva.num_of_p_frames)
            plt_eval_psnr.append(avg_psnr / eva.num_of_p_frames)
        ours, = plt.plot(plt_eval_bpp, plt_eval_psnr, "k-o", linewidth=line_width, label=eva.proposed_label)

    # H.264
    plt_h264_bpp = []
    plt_h264_psnr = []
    for bit_rate in eva.h264_bitrate_list:
        avg_bpp = 0
        avg_psnr = 0
        for iter in range(eva.num_of_p_frames - 1):
            avg_bpp += eva.h264_res[bit_rate][iter]["BPP"]
            avg_psnr += eva.h264_res[bit_rate][iter]["psnr"]
        plt_h264_bpp.append(avg_bpp / (eva.num_of_p_frames - 1))
        plt_h264_psnr.append(avg_psnr / (eva.num_of_p_frames - 1))

    plt_h265_bpp = []
    plt_h265_psnr = []
    for bit_rate in eva.h265_bitrate_list:
        avg_bpp = 0
        avg_psnr = 0
        for iter in range(eva.num_of_p_frames - 1):
            avg_bpp += eva.h265_res[bit_rate][iter]["BPP"]
            avg_psnr += eva.h265_res[bit_rate][iter]["psnr"]
        plt_h265_bpp.append(avg_bpp / (eva.num_of_p_frames - 1))
        plt_h265_psnr.append(avg_psnr / (eva.num_of_p_frames - 1))

    h264, = plt.plot(plt_h264_bpp, plt_h264_psnr, "m--s", linewidth=line_width, label='H.264')
    h265, = plt.plot(plt_h265_bpp, plt_h265_psnr, "r--v", linewidth=line_width, label='H.265')
    plt.legend(handles=[h264, h265, ours], loc=4)
    plt.grid()
    plt.xlabel('BPP')
    plt.ylabel('PSNR(dB)')
    plt.title(title)
    if fig_name != "":
        plt.savefig(fig_name, format='eps', dpi=600, bbox_inches='tight')
    plt.close()



def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Evaluator(object):
    def __init__(self, bs=1, height=240, width=240, channels=3,
                 input_seq_path="", workdir="", model_list=[], num_of_p_frames=5,
                 prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom",
                 tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave",
                 h264_workdir="", h264_bitrate_list=[50e3, 100e3, 200e3, 400e3, 1e5],
                 h265_workdir="", h265_bitrate_list=[50e3, 100e3, 200e3, 400e3, 1e5],
                 h266_workdir="", h266_bitrate_list=[50e3, 100e3, 200e3, 400e3, 1e5],
                 proposed_label="Proposed-Haar", ffmpeg_path="/usr/local/bin/ffmpeg") -> None:
        # bach size
        self.bs = bs
        self.height = height
        self.width = width
        self.channels = channels
        # input video sequence frames
        self.input_seq_path = input_seq_path
        # input data suffix and prefix 
        self.suffix = suffix
        self.prefix = prefix
        # tf encoder wordir
        self.workdir = workdir
        # tensorflow models paths
        self.model_list = model_list

        # number of frames to compress
        self.num_of_p_frames = num_of_p_frames

        # tf encoder suffix output
        self.bin_suffix = bin_suffix
        # tf model decompess prefix (it will be saved as png)
        self.decom_prefix = decom_prefix

        # the cpp encoder for x264 x265 evaluation
        self.tave_path = tave_path
        self.ffmpeg_path = ffmpeg_path
        
        self.h264_workdir = h264_workdir
        self.h264_bitrate_list = h264_bitrate_list

        self.h265_workdir = h265_workdir
        self.h265_bitrate_list = h265_bitrate_list
        
        self.h266_workdir = h266_workdir
        self.h266_bitrate_list = h266_bitrate_list
        # create paths if not exists
        path_exists(h264_workdir)
        path_exists(h265_workdir)
        path_exists(h266_workdir)
        path_exists(workdir)

        self.encoded_h264 = r"/output_joined.h264"
        self.encoded_h265 = r"/output_joined.h265"
        self.encoded_h266 = r"/output_joined.h266"

        self.proposed_label = proposed_label
        # results
        self.res = dict()
        self.h264_res = dict()
        self.h265_res = dict()
        self.h266_res = dict()
        
        self.i_frame = os.path.join(self.input_seq_path, self.prefix + str(0) + self.suffix)

        self.tave_run = False
        self.plt_our_bpp = []
        self.plt_our_psnr = []
        self.plt_h264_bpp = []
        self.plt_h264_psnr = []
        self.plt_h265_bpp = []
        self.plt_h265_psnr = []

    def PSNR(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def compare(self, original_path, compressed_path, binary_path):
        original = cv2.imread(original_path)
        compressed = cv2.imread(compressed_path)
        bin_size = os.path.getsize(binary_path)
        value = self.PSNR(original, compressed)
        return {"bin_size": bin_size, "psnr": value, "BPP": bin_size/(self.height*self.width*self.channels)}

    def test_tf_model(self, model, iframe, p_on_test, out_decom, p_frame_out_bin):
        OpenDVCW.compress(model, iframe, p_on_test, p_frame_out_bin, self.height, self.width)
        OpenDVCW.decompress(model, iframe, p_frame_out_bin, out_decom,  self.height, self.width)
        return self.compare(p_on_test, out_decom, p_frame_out_bin)
    
    def eval(self, tave_run=False):
        print("Evaluation Started ...")
        for model_name in self.model_list:
            print("working on model", model_name)
            iter_res = []

            model = tf.keras.models.load_model(model_name)
            for idx in np.arange(1, self.num_of_p_frames+1):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_bin = os.path.join(self.workdir, self.prefix + str(idx) + self.bin_suffix)
                p_out_decom = os.path.join(self.workdir, self.decom_prefix + str(idx) + self.suffix)
                iter_res.append(self.test_tf_model(model, self.i_frame, p_frame, p_out_decom, p_frame_out_bin))   

            self.res[model_name] = iter_res   

        if tave_run:
            self.tave_run = True
            print("H264 Evaluation ...")
            for bit_rate in self.h264_bitrate_list:
                self.h264_test(bit_rate)

            print("H265 Evaluation ...")
            for bit_rate in self.h265_bitrate_list:
                self.h265_test(bit_rate)
            
            print("H266 Evaluation ...")
            for bit_rate in self.h266_bitrate_list:
                self.h266_test(bit_rate)
        self.bpp_psnr()


    def h264_test(self, bit_rate):
        command = [self.tave_path, "libx264", str(bit_rate), self.input_seq_path,
                   self.prefix, self.suffix, self.h264_workdir, str(self.num_of_p_frames)]
        print("****************************************************************************************************************")
        print("Command: ", " ".join(command))
        subprocess.run(command)

        command = [self.ffmpeg_path, "-i", self.h264_workdir + self.encoded_h264, self.h264_workdir + r"/decoded_%04d.png"]

        subprocess.run(command)
        print("****************************************************************************************************************")
        print("Command: ", " ".join(command))
        res = []
        for idx in np.arange(1, self.num_of_p_frames):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_decom = os.path.join(self.h264_workdir, "decoded_" + str(idx).zfill(4) + self.suffix)
                p_frame_out_bin = os.path.join(self.h264_workdir, str(idx)+".h264")
                res.append(self.compare(p_frame, p_frame_out_decom, p_frame_out_bin))

        self.h264_res[bit_rate] = res

    def h265_test(self, bit_rate):
        command = [self.tave_path, "libx265", str(bit_rate), self.input_seq_path,
                   self.prefix, self.suffix, self.h265_workdir, str(self.num_of_p_frames)]
        print("****************************************************************************************************************")
        print("Command: ", " ".join(command))
        subprocess.run(command)

        command = [self.ffmpeg_path, "-i", self.h265_workdir + self.encoded_h265, self.h265_workdir + r"/decoded_%04d.png"]

        subprocess.run(command)
        print("****************************************************************************************************************")
        print("Command: ", " ".join(command))
        res = []
        for idx in np.arange(1, self.num_of_p_frames):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_decom = os.path.join(self.h265_workdir, "decoded_" + str(idx).zfill(4) + self.suffix)
                p_frame_out_bin = os.path.join(self.h265_workdir, str(idx)+".h265")
                res.append(self.compare(p_frame, p_frame_out_decom, p_frame_out_bin))

        self.h265_res[bit_rate] = res

    def h266_test(self, bit_rate):
        command = [self.tave_path, "libvvenc", str(bit_rate), self.input_seq_path,
                   self.prefix, self.suffix, self.h266_workdir, str(self.num_of_p_frames)]
        print("****************************************************************************************************************")
        print("Command: ", " ".join(command))
        subprocess.run(command)

        command = [self.ffmpeg_path, "-i", self.h266_workdir + self.encoded_h266, self.h266_workdir + r"/decoded_%04d.png"]
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Command: ", " ".join(command))
        subprocess.run(command)
        res = []
        for idx in np.arange(1, self.num_of_p_frames):
                p_frame = os.path.join(self.input_seq_path, self.prefix + str(idx) + self.suffix)
                p_frame_out_decom = os.path.join(self.h266_workdir, "decoded_" + str(idx).zfill(4) + self.suffix)
                p_frame_out_bin = os.path.join(self.h266_workdir, str(idx)+".h266")
                res.append(self.compare(p_frame, p_frame_out_decom, p_frame_out_bin))

        self.h266_res[bit_rate] = res
        
    def bpp_psnr(self):
        
        self.plt_our_bpp = []
        self.plt_our_psnr = []

        for model_name in self.model_list:
                    avg_bpp = 0
                    avg_psnr = 0
                    for iter in range(self.num_of_p_frames):
                        avg_bpp += self.res[model_name][iter]["BPP"]
                        avg_psnr += self.res[model_name][iter]["psnr"]
                    self.plt_our_bpp.append(avg_bpp/self.num_of_p_frames)
                    self.plt_our_psnr.append(avg_psnr/self.num_of_p_frames)

        if self.tave_run:
            
            # H.264 
            self.plt_h264_bpp = []
            self.plt_h264_psnr = []
            for bit_rate in self.h264_bitrate_list:
                avg_bpp = 0
                avg_psnr = 0
                for iter in range(self.num_of_p_frames-1):
                    avg_bpp += self.h264_res[bit_rate][iter]["BPP"]
                    avg_psnr += self.h264_res[bit_rate][iter]["psnr"]
                self.plt_h264_bpp.append(avg_bpp/(self.num_of_p_frames-1))
                self.plt_h264_psnr.append(avg_psnr/(self.num_of_p_frames-1))
            
            
            self.plt_h265_bpp = []
            self.plt_h265_psnr = []
            for bit_rate in self.h265_bitrate_list:
                avg_bpp = 0
                avg_psnr = 0
                for iter in range(self.num_of_p_frames-1):
                    avg_bpp += self.h265_res[bit_rate][iter]["BPP"]
                    avg_psnr += self.h265_res[bit_rate][iter]["psnr"]
                self.plt_h265_bpp.append(avg_bpp/(self.num_of_p_frames-1))
                self.plt_h265_psnr.append(avg_psnr/(self.num_of_p_frames-1))

            self.plt_h266_bpp = []
            self.plt_h266_psnr = []
            for bit_rate in self.h266_bitrate_list:
                avg_bpp = 0
                avg_psnr = 0
                for iter in range(self.num_of_p_frames-1):
                    avg_bpp += self.h266_res[bit_rate][iter]["BPP"]
                    avg_psnr += self.h266_res[bit_rate][iter]["psnr"]
                self.plt_h266_bpp.append(avg_bpp/(self.num_of_p_frames-1))
                self.plt_h266_psnr.append(avg_psnr/(self.num_of_p_frames-1))
                
                
    def plot_graph(self, fig_name="", title=''):
        font = {'family': 'DeJavu Serif', 'weight': 'normal', 'size': 14}
        matplotlib.rc('font', **font)
        line_width = 3
        
        ours, = plt.plot(self.plt_our_bpp, self.plt_our_psnr, "k-o", linewidth=line_width, label=self.proposed_label)
        
        if self.tave_run:
            h264, = plt.plot(self.plt_h264_bpp, self.plt_h264_psnr, "m--s", linewidth=line_width, label='H.264')
            h265, = plt.plot(self.plt_h265_bpp, self.plt_h265_psnr, "c--v", linewidth=line_width, label='H.265')
            h266, = plt.plot(self.plt_h266_bpp, self.plt_h266_psnr, "g--v", linewidth=line_width, label='H.266')
            
        if self.tave_run:
            plt.legend(handles=[h264, h265, h266, ours], loc=4)
        else:
            plt.legend(handles=[ours], loc=4)

        plt.grid()
        plt.xlabel('BPP')
        plt.ylabel('PSNR(dB)')
        plt.title(title)
        if fig_name != "":
            plt.savefig(fig_name, format='svg', dpi=600, bbox_inches='tight', transparent=True)   
        plt.close()

    def save_csv(self, fig_name="test.csv"):

        with open(fig_name, mode='w') as csv_file:
            if self.tave_run:
                fieldnames = ['H.264 PSNR', 'H.264 BPP',
                            'H.265 PSNR', 'H.265 BPP',
                            'H.266 PSNR', 'H.266 BPP',
                            self.proposed_label + " PSNR", self.proposed_label + " BPP"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for h4_psnr, h4_bpp, h5_psnr, h5_bpp, h6_psnr, h6_bpp, our_psnr, our_bpp in zip(self.plt_h264_psnr, self.plt_h264_bpp,
                                                                            self.plt_h265_psnr, self.plt_h265_bpp, 
                                                                            self.plt_h266_psnr, self.plt_h266_bpp, 
                                                                            self.plt_our_psnr ,self.plt_our_bpp,):
                    
                    writer.writerow({'H.264 PSNR': h4_psnr, 'H.264 BPP': h4_bpp,
                                    'H.265 PSNR': h5_psnr,'H.265 BPP': h5_bpp,
                                    'H.266 PSNR': h5_psnr,'H.266 BPP': h6_bpp,
                                    self.proposed_label + " PSNR": our_psnr, self.proposed_label + " BPP": our_bpp})
                    
            else:
                fieldnames = [self.proposed_label + " PSNR", self.proposed_label + " BPP"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for our_psnr, our_bpp in zip(self.plt_our_psnr ,self.plt_our_bpp,):
                    writer.writerow({self.proposed_label + " PSNR": our_psnr, self.proposed_label + " BPP": our_bpp})
