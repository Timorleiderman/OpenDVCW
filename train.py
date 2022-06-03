import cv2
import os
from math import log10, sqrt
import OpenDVCW
import numpy as np
import load
import tensorflow as tf
import DataGen
import Callbacks
import datetime


class TrainOpenDVCW(object):
    def __init__(self, batch_size=1,
                       epoch=800,
                       steps_per_epoch=100,
                       height=240, width=240, channels=3, 
                       num_filters=128, mv_kernel_size=3, res_kernel_size=5, M=128,
                       lmbda=4096, lr_init=1e-4, lr_alpha=1e-8, early_stop=400,
                       i_qp=27, wavelet_name="haar", checkponts_prev_path="", checkpoints_target_path = "",
                       np_folder="folder_cloud_test.npy") -> None:

        self.batch_size = batch_size
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.height = height
        self.width = width
        self.channels = channels
        self.num_filters = num_filters
        self.mv_kernel_size = mv_kernel_size
        self.res_kernel_size = res_kernel_size
        self.M = M
        self.lmbda = lmbda
        self.lr_init = lr_init
        self.lr_alpha = lr_alpha
        self.early_stop = early_stop
        self.I_QP=i_qp
        self.wavelet_name = wavelet_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "logs/fit/" + self.timestamp
        self.checkponts_last_path = checkponts_prev_path
        self.checkponts_new_path = os.path.join(checkpoints_target_path,"checkpoints_wavelets_{}_Lmbd_{}_nfilt_{}_epcs_{}_stps_{}_I_QP_{}_{}x{}_CosineDecay_{}/".format(self.wavelet_name,
                                                                                                                   self.lmbda,
                                                                                                                   self.num_filters,
                                                                                                                   self.epoch,
                                                                                                                   self.steps_per_epoch,
                                                                                                                   self.I_QP,
                                                                                                                   self.width,
                                                                                                                   self.height,
                                                                                                                   self.timestamp))
        self.save_name = "model_save_" + self.checkponts_new_path
        self.np_folder = np_folder
        # print("* [Loading dataset]...")
        self.data = DataGen.DataVimeo90kGenerator(self.np_folder, 
                                                  self.batch_size,
                                                  (self.height,self.width,self.channels),
                                                  self.channels,
                                                  True, 
                                                  self.I_QP,
                                                  True)
    def compile(self):
        self.model = OpenDVCW.OpenDVCW(width=self.width, height=self.height,channels=self.channels,batch_size=self.batch_size,
                                        num_filters=self.num_filters, mv_kernel_size=self.mv_kernel_size,
                                        res_kernel_size=self.res_kernel_size, M=self.M, lmbda=self.lmbda,
                                        wavelet_name=self.wavelet_name)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr_init, decay_steps=self.epoch*(self.steps_per_epoch), alpha=self.lr_alpha, name="lr_CosineDecay")

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),)
        print("* [Model compiled]...")

        
        if not self.checkponts_last_path == "":
            print("Loading weights")
            self.model.load_weights(self.checkponts_last_path)
    
    def fit(self, save_weights_only=True, save_freq='epoch', monitor="loss", mode='min',  save_best_only=True, verbose=1):
        self.hist = self.model.fit(x=self.data, steps_per_epoch=self.steps_per_epoch, epochs=self.epoch, verbose=verbose, batch_size=self.batch_size,
                callbacks=[
                    # Callbacks.MemoryCallback(),
                    # Callbacks.LearningRateReducer(),
                    tf.keras.callbacks.ModelCheckpoint(filepath=self.checkponts_new_path, save_weights_only=save_weights_only, save_freq=save_freq, monitor=monitor, mode=mode,  save_best_only=save_best_only, verbose=verbose), 
                    tf.keras.callbacks.TerminateOnNaN(),
                    tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=self.early_stop),
                    tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, update_freq=save_freq),            
                    ],
				)
    def test(self, i_frame, p_frame, out_bin, out_decom):

        OpenDVCW.compress(self.model, i_frame, p_frame, out_bin, self.width, self.height)
        OpenDVCW.decompress(self.model, i_frame, out_bin, out_decom, self.width, self.height)
        self.check_psnr(p_frame, out_decom, out_bin)
        
    def save(self):
        self.model.save(self.save_name, save_format="tf")

    def check_psnr(self, p_original, p_decompressed, p_bin_stream):
        def psnr(original, compressed):
            mse = np.mean((original - compressed) ** 2)
            if(mse == 0):  # MSE is zero means no noise is present in the signal .
                        # Therefore PSNR have no importance.
                return 100
            max_pixel = 255.0
            psnr = 20 * log10(max_pixel / sqrt(mse))
            return psnr
            
        original = cv2.imread(p_original)
        compressed = cv2.imread(p_decompressed)
        bin_size = os.path.getsize(p_bin_stream)
        value = psnr(original, compressed)
        print("bin size: ", bin_size , "psnr: ", value, "bpp: ", bin_size/(240*240*3))