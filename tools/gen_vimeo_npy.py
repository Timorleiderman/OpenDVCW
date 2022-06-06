import sys
import os
import numpy as np
import fnmatch

from subprocess import Popen, PIPE



def encode_decode_iframe_bpg(path, i_qp):
    
    in_img1 = path + "/im1.png"
    enc_img1 = path + "/im1_QP" + str(i_qp) + ".bpg"
    dec_img1 = path + "/im1_bpg444_QP" + str(i_qp) + ".png"
    process_enc = Popen([
                        "bpgenc",
                        "-f", "444",
                        "-m", "9",
                        in_img1,
                        "-o" , enc_img1,
                        "-q", str(i_qp)],
                        stdout=PIPE, stderr=PIPE)
    
    stdout, stderr = process_enc.communicate()
    if (stderr):
        print(stderr)
    
    process_dec = Popen([
                        "bpgdec",
                        enc_img1, 
                        "-o", dec_img1 
                        ],
                        stdout=PIPE, stderr=PIPE)
    
    stdout, stderr = process_dec.communicate()
    if (stderr):
        print(stderr)
        
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                sys.stdout.write('\r'+root + " ... added")  

    return result

def npy_encode_decode(npy, i_qp):
    for root in npy:
        encode_decode_iframe_bpg(root, i_qp)
        sys.stdout.write('\r'+root + " ... decoded and encoded")  

def gen_i_p_seq(npy, iqp=27, num_of_files_in_dir=7):
    res = []
    for path in npy:
        for idx in range(num_of_files_in_dir):
            if idx == 0:
                i_frame = path + '/im' + str(idx + 1) + '.png'
                p_frame = path + '/im1_bpg444_QP' + str(iqp) + '.png'
            else:
                i_frame = path + '/im' + str(1) + '.png'
                p_frame = path + '/im' + str(idx + 1) + '.png'

            res.append([i_frame, p_frame])
    return res

        
if __name__ == "__main__":
    # # generate npy file containing all the paths
    folder = find('im1.png', '/mnt/WindowsDev/DataSets/vimeo_septuplet/sequences/')
    for qp in [32, 37, 42]:
        npy_encode_decode(folder, qp)
    np.save('folder_cloud_test.npy', folder)
    npy = np.load("folder_cloud_test.npy")
    
    # train_set_iqp22 = gen_i_p_seq(npy, 22, 7)
    # np.save("train_set_iqp22.npy", train_set_iqp22)
    # train_set_iqp27 = gen_i_p_seq(npy, 27, 7)
    # np.save("train_set_iqp27.npy",train_set_iqp27)
    # train_set_iqp30 = gen_i_p_seq(npy, 30, 7)
    # np.save("train_set_iqp30.npy", train_set_iqp30)
    train_set_iqp32 = gen_i_p_seq(npy, 32, 7)
    np.save("train_set_iqp32.npy", train_set_iqp32)
    # train_set_iqp35 = gen_i_p_seq(npy, 35, 7)
    # np.save("train_set_iqp35.npy", train_set_iqp35)
    train_set_iqp37 = gen_i_p_seq(npy, 37, 7)
    np.save("train_set_iqp37.npy", train_set_iqp37)
    # train_set_iqp40 = gen_i_p_seq(npy, 40, 7)
    # np.save("train_set_iqp40.npy",train_set_iqp40)
    train_set_iqp42 = gen_i_p_seq(npy, 42, 7)
    np.save("train_set_iqp42.npy", train_set_iqp42)
    

