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

if __name__ == "__main__":
    # generate npy file containing all the paths
    folder = find('im1.png', '/mnt/WindowsDev/DataSets/vimeo_septuplet/sequences/')
    for qp in [22, 30, 35, 40]:
        npy_encode_decode(folder, qp)
    np.save('folder_cloud_test.npy', folder)

