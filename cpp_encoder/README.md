
# Run Examlpe:

```
./tave libx265 1255555 "/mnt/WindowsDev/DataSets/Beauty_1920x1080_120fps_420_8bit_YUV_RAW/" "im" ".png" /workspaces/OpenDVCW/cpp_encoder/build/workdir/
```



```
sudo apt-get update -qq && sudo apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev meson ninja-build pkg-config texinfo wget yasm zlib1g-dev 

mkdir -p ~/ffmpeg_sources ~/bin

sudo apt-get install libopencv-dev
sudo apt-get install libunistring-dev
sudo apt-get install nasm
sudo apt-get install yasm
sudo apt-get install libx264-dev
sudo apt-get install libx265-dev libnuma-dev
sudo apt-get install libvpx-dev
sudo apt-get install libfdk-aac-dev
sudo apt-get install libopus-dev

# AV1
cd ~/ffmpeg_sources && \
git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom && \
mkdir -p aom_build && \
cd aom_build && \
PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
PATH="$HOME/bin:$PATH" make && \
make install


# optional:
sudo apt-get install itstool
git clone --depth=1 https://code.videolan.org/videolan/dav1d.git && \
cd dav1d && \
mkdir build && cd build && \
meson --bindir="/usr/local/bin" .. && \
ninja && \
ninja install


###################################### SVT av1 ###################################
# optional:
git clone --depth=1 https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
sudo make install
cd ~
git clone -b release/4.2 --depth=1 https://github.com/FFmpeg/FFmpeg ffmpeg
cd ffmpeg
export LD_LIBRARY_PATH+=":/usr/local/lib"
export PKG_CONFIG_PATH+=":/usr/local/lib/pkgconfig"
git apply ../SVT-AV1/ffmpeg_plugin/0001-Add-ability-for-ffmpeg-to-run-svt-av1.patch
./configure --enable-libsvtav1
###################################################################################


cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libaom \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libsvtav1 \
  --enable-libdav1d \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

```


