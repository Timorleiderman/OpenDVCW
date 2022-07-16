FROM tensorflow/tensorflow:latest-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub 88
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub 20
RUN apt-get update

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git -y
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install tensorflow
RUN pip3 install tensorflow-compression
RUN pip3 install tensorflow-probability
RUN pip3 install tensorflow_addons
RUN pip3 install PyWavelets
RUN pip3 install psnr-hvsm
RUN pip3 install imageio
RUN pip3 install numpy 
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install ipykernel
RUN pip3 install jupyter
RUN pip3 install notebook
RUN pip3 install jupyterlab


EXPOSE 8888
RUN useradd -ms /bin/bash ubu-admin

WORKDIR /workspaces

RUN git clone https://github.com/Timorleiderman/tensorflow-wavelets.git


RUN apt-get install wget libsdl-image1.2-dev libsdl1.2-dev libjpeg8-dev yasm cmake -y 
RUN wget http://bellard.org/bpg/libbpg-0.9.7.tar.gz
RUN tar xzf libbpg-0.9.7.tar.gz
RUN cd libbpg-0.9.7/
# RUN make && make install

ENV PYTHONPATH /workspaces/tensorflow-wavelets/src


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    git \
    wget \
    zip \
    unzip \
    python3 \
    python3-pip \
    curl

RUN apt install -y cmake gdb  

RUN mkdir /opencv_build
WORKDIR /opencv_build
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git

WORKDIR /opencv_build/opencv

RUN mkdir -p build && cd build && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_build/opencv_contrib/modules \
    -D BUILD_TESTS=OFF \
    -D DBUILD_EXAMPLES=OFF \
    -D WITH_OPENNI=OFF \
    -D WITH_OPENNI2=OFF \
    -D WITH_PVAPI=OFF \
    -D WITH_ANDROID_MEDIANDK=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D OPENCV_DNN_OPENCL=OFF \
    -D WITH_GSTREAMER=OFF \
    -D WITH_V4L=OFF \
    -D WITH_AVFOUNDATION=OFF \
    -D VIDEOIO_ENABLE_PLUGINS=OFF \
    .. && make -j10 && make install


# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

