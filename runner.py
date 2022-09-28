import evaluator

model_dir = "/mnt/WindowsDev/PreTainedModels/OpenDVCW/"
label = "Proposed-db2"

# model_list = [
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_256_nfilt_256_epcs_15_stps_10000_I_QP_42_240x240_CosineDecay_20220705-061459/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_1024_nfilt_256_epcs_15_stps_10000_I_QP_37_240x240_CosineDecay_20220704-225023/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_4096_nfilt_256_epcs_15_stps_10000_I_QP_32_240x240_CosineDecay_20220704-152812/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_16384_nfilt_256_epcs_15_stps_10000_I_QP_27_240x240_CosineDecay_20220704-044114/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_65536_nfilt_256_epcs_15_stps_10000_I_QP_22_240x240_CosineDecay_20220703-214625/"
# ]

model_list = [
    model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20220729-022643/",
    model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_37_240x240_CosineDecay_20220727-000239/",
    model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20220725-042633/",
    model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20220721-093040/",
    model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20220719-155911/"
]

# model_list_sym3 = [
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20220814-014849/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_37_240x240_CosineDecay_20220812-053904/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20220810-054911/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20220808-140315/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20220806-182807/"
# ]

input_seq_paths = [
    "/mnt/WindowsDev/DataSets/Beauty_1920x1080_120fps_420_8bit_YUV_RAW/",
    "/mnt/WindowsDev/DataSets/Bosphorus_1920x1080_120fps_420_8bit_YUV_raw/",
    "/mnt/WindowsDev/DataSets/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW/",
    "/mnt/WindowsDev/DataSets/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW/"
    ]


h264_bit_rate = [0.5e6, 1.5e6, 3e6, 5e6, 6e6]
h265_bit_rate = [0.5e6, 1.5e6, 3e6, 5e6, 6e6]

num_of_p_frames = 7


for image_seq_path in input_seq_paths:
    eva = evaluator.Evaluator(bs=1, height=240, width=240,
                                channels=3, input_seq_path=image_seq_path,
                                workdir="/workspaces/OpenDVCW/Test_com/eval",
                                model_list=model_list,
                                num_of_p_frames=num_of_p_frames,
                                prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom",
                                tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave",
                                h264_workdir="/workspaces/OpenDVCW/Test_com/h264/test",
                                h264_bitrate_list=h264_bit_rate,
                                h265_workdir="/workspaces/OpenDVCW/Test_com/h265/test",
                                h265_bitrate_list=h265_bit_rate,
                                proposed_label=label)
    eva.eval(tave_run=True)
    eva.plot_graph(fig_name=image_seq_path.split("/")[-2] + label + ".eps")
    eva.save_csv(image_seq_path.split("/")[-2] + label + ".csv")
