import evaluator

model_dir = "/mnt/WindowsDev/PreTainedModels/OpenDVCW/"
label = "Proposed-haar200k"
output_path = "/mnt/WindowsDev/Timor/Study/TEZA/RES/"
num_of_p_frames = 7

# model_list = [
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_256_nfilt_256_epcs_15_stps_10000_I_QP_42_240x240_CosineDecay_20220705-061459/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_1024_nfilt_256_epcs_15_stps_10000_I_QP_37_240x240_CosineDecay_20220704-225023/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_4096_nfilt_256_epcs_15_stps_10000_I_QP_32_240x240_CosineDecay_20220704-152812/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_16384_nfilt_256_epcs_15_stps_10000_I_QP_27_240x240_CosineDecay_20220704-044114/",
#     model_dir + "HAAR/model_save_checkpoints_wavelets_haar_Lmbd_65536_nfilt_256_epcs_15_stps_10000_I_QP_22_240x240_CosineDecay_20220703-214625/"
# ]

# model_list = [
#     model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20220729-022643/",
#     model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_37_240x240_CosineDecay_20220727-000239/",
#     model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20220725-042633/",
#     model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20220721-093040/",
#     model_dir + "DB2/model_save_checkpoints_wavelets_db2_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20220719-155911/"
# ]

# model_list = [
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20221001-130850/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_37_240x240_CosineDecay_20220930-212030/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20220930-053044/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20220929-131432/",
#     model_dir + "SYM3/model_save_checkpoints_wavelets_sym3_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20220928-182917/"
# ]

# model_list = [
#     model_dir + "COIF1/model_save_checkpoints_wavelets_coif1_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20221005-112002/",
#     model_dir + "COIF1/model_save_checkpoints_wavelets_coif1_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_37_240x240_CosineDecay_20221004-192207/",
#     model_dir + "COIF1/model_save_checkpoints_wavelets_coif1_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20221004-033121/",
#     model_dir + "COIF1/model_save_checkpoints_wavelets_coif1_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20221003-115147/",
#     model_dir + "COIF1/model_save_checkpoints_wavelets_coif1_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20221002-194858/"
# ]


# model_list = [
#     model_dir + "HAAR60k/model_save_checkpoints_wavelets_haar_Lmbd_256_nfilt_256_epcs_15_stps_60000_I_QP_42_240x240_CosineDecay_20221203-191019/",
#     model_dir + "HAAR60k/model_save_checkpoints_wavelets_haar_Lmbd_1024_nfilt_256_epcs_15_stps_60000_I_QP_35_240x240_CosineDecay_20221204-145313/",
#     model_dir + "HAAR60k/model_save_checkpoints_wavelets_haar_Lmbd_4096_nfilt_256_epcs_15_stps_60000_I_QP_32_240x240_CosineDecay_20221202-182053/",
#     model_dir + "HAAR60k/model_save_checkpoints_wavelets_haar_Lmbd_16384_nfilt_256_epcs_15_stps_60000_I_QP_27_240x240_CosineDecay_20221202-054609/",
#     model_dir + "HAAR60k/model_save_checkpoints_wavelets_haar_Lmbd_65536_nfilt_256_epcs_15_stps_60000_I_QP_22_240x240_CosineDecay_20221201-171832/"
# ]


model_list = [
    model_dir + "HAAR200k/model_save_checkpoints_wavelets_haar_Lmbd_256_nfilt_256_epcs_5_stps_200000_I_QP_42_240x240_CosineDecay_20221207-161406/",
    model_dir + "HAAR200k/model_save_checkpoints_wavelets_haar_Lmbd_1024_nfilt_256_epcs_5_stps_200000_I_QP_35_240x240_CosineDecay_20221207-022051/",
    model_dir + "HAAR200k/model_save_checkpoints_wavelets_haar_Lmbd_4096_nfilt_256_epcs_5_stps_200000_I_QP_32_240x240_CosineDecay_20221206-123302/",
    model_dir + "HAAR200k/model_save_checkpoints_wavelets_haar_Lmbd_16384_nfilt_256_epcs_5_stps_200000_I_QP_27_240x240_CosineDecay_20221205-224634/",
    model_dir + "HAAR200k/model_save_checkpoints_wavelets_haar_Lmbd_65536_nfilt_256_epcs_5_stps_200000_I_QP_22_240x240_CosineDecay_20221205-085653/"
]


# Beauty_1920x1080_120fps_420_8bit_YUV_RAW for db2
input_seq_path =  "/mnt/WindowsDev/DataSets/Beauty_1920x1080_120fps_420_8bit_YUV_RAW/"
h264_bit_rate = [0.5e6, 1.5e6, 3e6, 5e6, 6e6]
h265_bit_rate = [0.5e6, 1.5e6, 3e6, 5e6, 6e6]
eva = evaluator.Evaluator(bs=1, height=240, width=240, channels=3, input_seq_path=input_seq_path, workdir="/workspaces/OpenDVCW/Test_com/eval", model_list=model_list, num_of_p_frames=num_of_p_frames, prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom", tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave", h264_workdir="/workspaces/OpenDVCW/Test_com/h264/test", h264_bitrate_list=h264_bit_rate, h265_workdir="/workspaces/OpenDVCW/Test_com/h265/test", h265_bitrate_list=h265_bit_rate, proposed_label=label)
eva.eval(tave_run=True)
eva.plot_graph(fig_name=output_path +input_seq_path.split("/")[-2] + label + ".eps")
eva.save_csv(output_path + input_seq_path.split("/")[-2] + label + ".csv")
# Bosphorus_1920x1080_120fps_420_8bit_YUV_raw for db2
input_seq_path = "/mnt/WindowsDev/DataSets/Bosphorus_1920x1080_120fps_420_8bit_YUV_raw/"
h264_bit_rate = [0.8e6, 3e6, 6e6, 8e6, 11e6]
h265_bit_rate = h264_bit_rate
eva = evaluator.Evaluator(bs=1, height=240, width=240, channels=3, input_seq_path=input_seq_path, workdir="/workspaces/OpenDVCW/Test_com/eval", model_list=model_list, num_of_p_frames=num_of_p_frames, prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom", tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave", h264_workdir="/workspaces/OpenDVCW/Test_com/h264/test", h264_bitrate_list=h264_bit_rate, h265_workdir="/workspaces/OpenDVCW/Test_com/h265/test", h265_bitrate_list=h265_bit_rate, proposed_label=label)
eva.eval(tave_run=True)
eva.plot_graph(fig_name=output_path +input_seq_path.split("/")[-2] + label + ".eps")
eva.save_csv(output_path + input_seq_path.split("/")[-2] + label + ".csv")
# ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW for db2
input_seq_path = "/mnt/WindowsDev/DataSets/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW/"
h264_bit_rate = [0.5e6, 2e6, 5e6, 7e6, 9e6]
h265_bit_rate = h264_bit_rate
eva = evaluator.Evaluator(bs=1, height=240, width=240, channels=3, input_seq_path=input_seq_path, workdir="/workspaces/OpenDVCW/Test_com/eval", model_list=model_list, num_of_p_frames=num_of_p_frames, prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom", tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave", h264_workdir="/workspaces/OpenDVCW/Test_com/h264/test", h264_bitrate_list=h264_bit_rate, h265_workdir="/workspaces/OpenDVCW/Test_com/h265/test", h265_bitrate_list=h265_bit_rate, proposed_label=label)
eva.eval(tave_run=True)
eva.plot_graph(fig_name=output_path + input_seq_path.split("/")[-2] + label + ".eps")
eva.save_csv(output_path + input_seq_path.split("/")[-2] + label + ".csv")
# HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW for db2
input_seq_path = "/mnt/WindowsDev/DataSets/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW/"
h264_bit_rate = [1e6, 10e6, 20e6, 25e6, 30e6]
h265_bit_rate = h264_bit_rate
eva = evaluator.Evaluator(bs=1, height=240, width=240, channels=3, input_seq_path=input_seq_path, workdir="/workspaces/OpenDVCW/Test_com/eval", model_list=model_list, num_of_p_frames=num_of_p_frames, prefix="im", suffix=".png", bin_suffix=".bin", decom_prefix="decom", tave_path="/workspaces/OpenDVCW/cpp_encoder/build/tave", h264_workdir="/workspaces/OpenDVCW/Test_com/h264/test", h264_bitrate_list=h264_bit_rate, h265_workdir="/workspaces/OpenDVCW/Test_com/h265/test", h265_bitrate_list=h265_bit_rate, proposed_label=label)
eva.eval(tave_run=True)
eva.plot_graph(fig_name=output_path +input_seq_path.split("/")[-2] + label + ".eps")
eva.save_csv(output_path +input_seq_path.split("/")[-2] + label + ".csv")














