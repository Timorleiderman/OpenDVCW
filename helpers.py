import numpy as np
import flow_vis
import flow_utils
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output
from flow_utils import draw_flow


def plot_ip_ff_vv(i_frame, p_frame, flow, flow_w, figsize=(20, 20), clear=True):
    if clear:
        clear_output(wait=True) 
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
    flow_color_w = flow_vis.flow_to_color(flow_w, convert_to_bgr=True)
    
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    ax[0, 0].imshow(i_frame)
    ax[0, 0].axis('off')
    ax[0, 0].set_title("I-Frame")
    
    ax[0, 1].imshow(p_frame)
    ax[0, 1].axis('off')
    ax[0, 1].set_title("P-Frame")
    
    ax[1, 0].imshow(flow_color)
    ax[1, 0].axis('off')
    ax[1, 0].set_title("OpticalFlow DVC")
    
    ax[1, 1].imshow(flow_color_w)
    ax[1, 1].set_title("OpticalFlow ODVCW")
    ax[1, 1].axis('off')
    
    ax[2, 0].imshow(draw_flow(p_frame, flow_w))
    ax[2, 0].set_title("m-vectors ODVCW")
    ax[2, 0].axis('off')
    
    ax[2, 1].imshow(draw_flow(p_frame, flow))
    ax[2, 1].set_title("m-vectors DVC")
    ax[2, 1].axis('off')
    
    fig.tight_layout()
    plt.show()
    

def plot_ip_f(i_frame, p_frame, flow, figsize=(20, 20), clear=True):
    if clear:
        clear_output(wait=True) 
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(i_frame)
    ax[0].axis('off')
    ax[0].set_title("I-Frame")
    
    ax[1].imshow(p_frame)
    ax[1].axis('off')
    ax[1].set_title("P-Frame")
    
    ax[2].imshow(draw_flow(i_frame, flow))
    ax[2].axis('off')
    ax[2].set_title("P-Frame")
    fig.tight_layout()
    
    plt.show()



def plot_quiver(ax, flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    
    h, w, *_ = flow.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(x, y, u, v, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")
    ax.axis("off")
    

def plot_ffpmv(im1, im2, flow, spacing=8, title="", figsize=(10, 10), clear=True):
    if clear:
        clear_output(wait=True) 
        
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    ax[0].imshow(flow_utils.flow_to_image(flow))
    ax[0].axis("off")
    ax[0].set_title("optical flow vis")
    
    ax[1].imshow(cv2.normalize(im2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    ax[1].axis("off")
    ax[1].set_title("Image 2 with motion vectors")
    
    flow_to_draw = flow_utils.draw_flow(im1, np.swapaxes(flow, 0,1), spacing)
    ax[2].imshow(cv2.normalize(flow_to_draw, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    ax[2].axis("off")
    ax[2].set_title("Image 1")
    
    plot_quiver(ax[3], flow, spacing=spacing, margin=0, scale=1, color="#00FF00") 
    ax[3].set_title("Motion Vectors")
    fig.suptitle(title, fontsize=30)
    fig.tight_layout()
    plt.show()

class FlyingChairsFlags():
    
    data_dir = "/mnt/WindowsDev/DataSets/FlyingChairs/data/"  #, 'Link to dataset directory.')
    train_list = "/mnt/WindowsDev/DataSets/FlyingChairs/train_0.03split.txt"  #'Link to validation list.')
    val_list ="/mnt/WindowsDev/DataSets/FlyingChairs/val_0.03split.txt"  #, "Link to validation list.")
    losses_weight = [0.32, 0.08, 0.02, 0.01, 0.005]  # Loss weights for 6th to 2nd flow predictions, as described in the original paper.')
    gamma = 0.0004  
    batch_size = 1 
    dataset = 'mixed'  # ['mixed', 'chairs', 'things3d_ft'], None)

    #chedule for training from scratch. Train on the mixed dataset of FlyingChairs and FlyingThings3D.
    lr_boundaries = [400000, 600000, 800000, 1000000]
    lr = 0.0001
    num_steps =1500000
    crop_size =[240, 240]

    steps_per_save =10000
    steps_per_eval = 1000
    log_freq = 50

    random_scale = False  
    random_flip = False  
