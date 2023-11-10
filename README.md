

# Training:

prepare the dataset:

install BPG for training we used BPG to compress the first frame for the I frame compression training
- BPG ([Download link](https://bellard.org/bpg/)) 

- Download the training data. We train the models on the [Vimeo90k dataset](https://github.com/anchen1011/toflow) ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)) (82G).

unzip the files to some dir

Use the script to generate the .npy file containing all the paths to the dataset images
tools/gen_vimeo_npy.py

We used 240x240x3 resolution for training


# Docker 
```__________________________
docker build -t tensorflow-wavelets:1.0 .

docker run --privileged=true -v /mnt/:/mnt/ --gpus all --user 1000:1000 -p 6006:6006 -p 8080:8080 tensorflow-wavelets:1.0
```



Based on
> Lu, Guo, *et al.* "DVC: An end-to-end deep video compression framework." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2019.
[Paper](https://arxiv.org/abs/2006.15862)

# conda env
```
conda create -n py38 python=3.8
pip install -r 
```
**Free Software, Hell Yeah!**

