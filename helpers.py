import flow_vis
import matplotlib.pyplot as plt

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
