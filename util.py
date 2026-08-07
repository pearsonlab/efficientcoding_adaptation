import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

matplotlib.use("agg")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def kernel_images(W, kernel_size, image_channels, rows=None, cols=None, spacing=1):
    """
    Return the kernels as tiled images for visualization
    :return: np.ndarray, shape = [rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing, 1]
    """

    W /= np.linalg.norm(W, axis=0, keepdims=True)
    W = W.reshape(image_channels, -1, W.shape[-1])

    if rows is None:
        rows = int(np.ceil(math.sqrt(W.shape[-1])))
    if cols is None:
        cols = int(np.ceil(W.shape[-1] / rows))

    kernels = np.ones([3, rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x:x+kernel_size, y:y+kernel_size] = kernel

    return kernels.clip(0, 1)


def plot_convolution(weight: torch.Tensor):
    if torch.is_tensor(weight):
        weight = weight.numpy()
    weight = weight / np.linalg.norm(weight, axis=-1, keepdims=True)

    fig = plt.figure(figsize=(4, 4))
    plt.plot(weight[:, 0, :].T)
    plt.tight_layout()
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncol, nrow = fig.canvas.get_width_height()
    buf = buf.reshape(ncol, nrow, 3)
    plt.close()

    return buf.transpose(2, 0, 1)


def find_last_cp(path):
    all_files = np.array(os.listdir(path))
    max_cp = 0
    for file in all_files:
        if file[0:10] == 'checkpoint':
            cp_num = int(file[11:-3])
            if cp_num > max_cp:
                max_cp = cp_num
                max_cp_file = file
    return max_cp_file

def find_last_model(path):
    all_files = np.array(os.listdir(path))
    max_model = 0
    for file in all_files:
        if file[0:5] == 'model':
            model_num = int(file[6:-3])
            if model_num > max_model:
                max_model = model_num
                max_model_file = file
    return max_model_file


def reshape_flat_W(W, n_neurons, kernel_size, n_colors):
    W_pre = W.reshape([n_colors,kernel_size,kernel_size,n_neurons])
    W_reshape = np.swapaxes(W_pre,0,3)
    return W_reshape

def round_kernel_centers(kernel_centers):
    kernel_centers_int = np.zeros(kernel_centers.shape, dtype = np.int16)
    n_neurons = kernel_centers.shape[0]
    for n in range(n_neurons):
        kernel_centers_int[n,0] = int(round(kernel_centers[n,0]))
        kernel_centers_int[n,1] = int(round(kernel_centers[n,1]))
    return kernel_centers_int


def scale(W, W_all = None):
    if W_all is not None:
        W_scale = W_all
    else:
        W_scale = W
    
    
    W_max = np.max(W_scale)
    W_min = np.min(W_scale)
    if abs(W_max) < abs(W_min):
        ext = abs(W_min)
    else:
        ext = abs(W_max)
    
    W = W/(2*ext) + 0.5
    return W


def make_rr(og_size, new_size, kernel_center = [0,0]):
    if new_size%2 !=0:
        Exception("New size must be even!")
    middle = og_size/2
    x,y = np.linspace(-middle, middle - 1, new_size), np.linspace(-middle, middle - 1, new_size)
    rr = np.meshgrid(x,y)
    rr = np.expand_dims(rr, 0)
    return rr 
   

def closest_divisor(number, max_rows = 10, max_cols = 10):
    empty_min = max_rows*max_cols
    best_row = 0
    best_col = 0
    
    for r in range(max_rows):
        for c in range(max_cols):
            tot = r*c
            if tot >= number:
                diff = tot - number
                if diff <= empty_min:
                    if diff < empty_min or r + c < best_row + best_col:                    
                        empty_min = diff
                        best_row = r
                        best_col = c
                        
    if best_col < best_row:
        return best_row, best_col
    else:
        return best_col, best_row
    
    

def get_matrix(matrix_type):
    pca_comps_old = np.array([[ 0.51876956, 0.52288215, 0.67636706],
                 [ 0.48552343, 0.4709887, -0.73650299],
                 [ 0.7036655, -0.71046739, 0.00953693]])
    
    pca_comps = np.array([[ 0.56808728,  0.57183637,  0.59184459],
           [ 0.42497522,  0.41201491, -0.80600234],
           [ 0.70475024, -0.70939896,  0.00895586]])
    pca_inv = np.linalg.inv(pca_comps)  
    
    #Do you see what I see? -Understanding the challenges of colour-blindness in online learning
    rgb_to_lms = np.array([[17.88240413, 43.51609057,  4.11934969],
                           [ 3.45564232, 27.15538246,  3.86713084],
                           [ 0.02995656,  0.18430896,  1.46708614]])
    lms_to_rgb = np.linalg.inv(rgb_to_lms)
    
    if matrix_type == 'pca_comps':
        return pca_comps
    elif matrix_type == 'pca_inv':
        return pca_inv
    elif matrix_type == 'rgb_to_lms':
        return rgb_to_lms
    elif matrix_type == 'lms_to_rgb':
        return lms_to_rgb
    elif matrix_type == 'pca_comps_old':
        return pca_comps_old
    

def hexagonal_grid(n_neurons, kernel_size, n_mosaics):
    if n_neurons%n_mosaics != 0:
        raise ValueError("Number of neurons has to be a multiple of the number of mosaics!")
    neurons_per_mosaic = int(n_neurons/n_mosaics)
    radius = kernel_size/2
    dist = 1
    goal_neurons = False
    while not goal_neurons:
        size = int(kernel_size/2)
        #n_x, n_y = closest_divisor(n_neurons)
        x_all = np.arange(-size,size,dist)
        y_all = np.arange(-size,size,dist)
        
        x, y = np.meshgrid(x_all, y_all)
        for x_pos in range(x.shape[1]):
            if x_pos%2 == 0:
                x[x_pos,:] = x[x_pos,:] - dist/2
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        kernel_centers = np.stack((x_flat,y_flat),1)
    
        center_dist = np.sqrt(x_flat**2 + y_flat**2)
        within_radius = np.sum(center_dist < radius)
        if within_radius < neurons_per_mosaic:
            dist = dist - 0.001
        elif within_radius > neurons_per_mosaic:
            dist = dist + 0.001
        else:
            goal_neurons = True
        
    kernel_centers = kernel_centers[center_dist < radius, :] #Subset to neurons within radius
    kernel_centers = kernel_centers + radius #Change axes to follow [0,kernel_size] convention
    kernel_centers = torch.tensor(kernel_centers, device = 'cuda') #Tensor
    kernel_centers = kernel_centers.tile(n_mosaics,1) #Tile for every mosaic
    return kernel_centers


def check_d(model, nnum, n_colors, n_neurons):
    for param in model.parameters():
        if param.shape == torch.Size([n_colors*4,n_neurons]):
            #print('Printing this epoch')
            dL = param[3,nnum].item()
            dS = param[7,nnum].item()
            
            print("d parameter: ", dL, dS)