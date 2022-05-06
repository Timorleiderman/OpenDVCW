import os
import cv2
import numpy as np


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_YV12)
        return ret, bgr

    def read_resize(self, target_size):
        ret, img = cap.read()
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return ret, resized

if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "P:\DataSets\Beauty_1920x1080_120fps_420_8bit_YUV_RAW\Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
    output_path = "P:\DataSets\Beauty_1920x1080_120fps_420_8bit_YUV_RAW"
    size = (1080, 1920)
    cap = VideoCaptureYUV(filename, size)

    cnt = 0
    while 1:
        ret, frame = cap.read_resize((240, 240))
        if ret:
            cv2.imwrite(os.path.join(output_path, "im" + str(cnt) + ".png"), frame)
            cnt += 1
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        else:
            break