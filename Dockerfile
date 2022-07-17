FROM tensorflow/tensorflow:2.6.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub 88
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub 20

RUN apt-get update -y && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    pkg-config \
    ninja-build \
    wget \
    automake \
    autoconf \
    zip \
    unzip \
    python3 \
    python3-pip \
    curl \
    libsdl-image1.2-dev \
    libsdl1.2-dev \
    libjpeg8-dev \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    yasm \
    meson \
    cmake \
    gdb \
    libunistring-dev \
    libaom-dev \
    texinfo \
    zlib1g-dev \
    libssl-dev 
        
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install tensorflow-compression tensorflow-probability tensorflow_addons
RUN pip3 install PyWavelets psnr-hvsm imageio numpy scipy matplotlib ipykernel jupyter notebook jupyterlab

EXPOSE 8888

WORKDIR /workspaces
RUN git clone https://github.com/Timorleiderman/tensorflow-wavelets.git

RUN wget http://bellard.org/bpg/libbpg-0.9.7.tar.gz
RUN tar xzf libbpg-0.9.7.tar.gz
RUN cd libbpg-0.9.7/
# RUN make && make install

ENV PYTHONPATH /workspaces/tensorflow-wavelets/src

RUN mkdir /opencv_build
RUN mkdir /ffmpeg_sources

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


RUN apt install libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libx265-dev libx264-dev libnuma-dev  -y


RUN useradd -ms /bin/bash ubu-admin

# RUN mkdir /workspaces/OpenDVCW/cpp_encoder/build
# RUN cd /workspaces/OpenDVCW/cpp_encoder/build && cmake .. && make



# # Compile NASM
# RUN cd /ffmpeg_sources && \
#   wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2 && \
#   tar xjvf nasm-2.15.05.tar.bz2 && \
#   cd nasm-2.15.05 && \
#   ./autogen.sh && \
#   PATH="/bin:$PATH" ./configure --prefix="/ffmpeg_build" --bindir="/bin" && \
#   make && \
#   make install

# # Compile YASM
# RUN cd /ffmpeg_sources && \
#   wget -O yasm-1.3.0.tar.gz https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz && \
#   tar xzvf yasm-1.3.0.tar.gz && \
#   cd yasm-1.3.0 && \
#   ./configure --prefix="/ffmpeg_build" --bindir="/bin" && \
#   make && \
#   make install

# # Compile x264
# ENV GIT_SSL_NO_VERIFY=1
# RUN cd /ffmpeg_sources && \
#   git clone --depth 1 https://code.videolan.org/videolan/x264.git && \
#   cd x264 && \
#   PATH="/bin:$PATH" PKG_CONFIG_PATH="/ffmpeg_build/lib/pkgconfig" ./configure --prefix="/ffmpeg_build" --bindir="/bin" --enable-static --enable-pic && \
#   PATH="/bin:$PATH" make && \
#   make install

# # Complile HEVC
# RUN apt-get install libx265-dev libnuma-dev -y

# # Compile fdk-aac
# RUN cd /ffmpeg_sources && \
#   wget -O fdk-aac.tar.gz https://github.com/mstorsjo/fdk-aac/tarball/master && \
#   tar xzvf fdk-aac.tar.gz && \
#   cd mstorsjo-fdk-aac* && \
#   autoreconf -fiv && \
#   ./configure --prefix="/ffmpeg_build" --disable-shared && \
#   make && \
#   make install

# # Compile libmp3lame
# RUN cd /ffmpeg_sources && \
#   wget http://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.5.tar.gz && \
#   tar xzvf lame-3.99.5.tar.gz && \
#   cd lame-3.99.5 && \
#   ./configure --prefix="/ffmpeg_build" --enable-nasm --disable-shared && \
#   make && \
#   make install

# # Compile libopus
# RUN cd /ffmpeg_sources && \
#   wget https://archive.mozilla.org/pub/opus/opus-1.1.5.tar.gz && \
#   tar xzvf opus-1.1.5.tar.gz && \
#   cd opus-1.1.5 && \
#   ./configure --prefix="/ffmpeg_build" --disable-shared && \
#   make && \
#   make install

# # Compile libvpx
# RUN apt-get install git -y && \
#   cd /ffmpeg_sources && \
#   git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
#   cd libvpx && \
#   PATH="/bin:$PATH" ./configure --prefix="/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth && \
#   PATH="/bin:$PATH" make && \
#   make install

# # Compile SRT
# RUN cd /ffmpeg_sources && \
#   git clone --depth 1 https://github.com/Haivision/srt.git && \
#   cd srt && \
#   cmake -DCMAKE_INSTALL_PREFIX="/ffmpeg_build" -DENABLE_SHARED=OFF -DENABLE_STATIC=ON && \
#   make && \
#   make install

# RUN apt install libtheora-dev -y
# # Compile ffmpeg
# RUN cd /ffmpeg_sources && \
#   wget http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
#   tar xjvf ffmpeg-snapshot.tar.bz2 && \
#   cd ffmpeg && \
#   PKG_CONFIG_PATH="/ffmpeg_build/lib/pkgconfig" ./configure \
#     --prefix="/ffmpeg_build" \
#     --pkg-config-flags="--static" \
#     --extra-cflags="-I/ffmpeg_build/include" \
#     --extra-ldflags="-L/ffmpeg_build/lib" \
#     --bindir="/bin" \
#     --enable-gpl \
#     --enable-libass \
#     --enable-libfdk-aac \
#     --enable-libfreetype \
#     --enable-libmp3lame \
#     --enable-libopus \
#     --enable-libtheora \
#     --enable-libvorbis \
#     --enable-libvpx \
#     --enable-libx264 \
#     --enable-libx265 \
#     --enable-nonfree \
#     --enable-openssl \
#     --enable-libsrt && \
#   make && \
#   make install && \
#   hash -r



# RUN cd /workspaces/OpenDVCW/cpp_encoder && cmake .
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

