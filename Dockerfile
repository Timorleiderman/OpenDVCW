FROM tensorflow/tensorflow:latest-gpu

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

ENV PYTHONPATH /workspaces/tensorflow-wavelets/src
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

