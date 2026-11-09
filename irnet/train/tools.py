import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    else:
        name = "Using CPU"
    print(f"Using device: {name}")
    return device

def get_gpu_memory(device):
    if device == "cuda":
        # Total memory:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Convert bytes to GB for easier readability:
        total_memory_GB = total_memory / (1024 ** 3)
        
        print(f"Total memory available (GB): {total_memory_GB:.2f}")
    else:
        print("Using CPU, so no GPU memory details available.")

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized

def prepare(x, normalize):
    x = x.detach().cpu().numpy()[0].transpose(1, 2, 0)
    return normalize(x)

def total_variation_loss(image): # Loss for smoothness across an image 
    tv_x = torch.sum((image[:, :, 1:, :] - image[:, :, :-1, :])**2)
    tv_y = torch.sum((image[:, :, :, 1:] - image[:, :, :, :-1])**2)

    num_elements = image.size(2) * image.size(3) - image.size(2) - image.size(3)
    return (tv_x + tv_y) / (2 * num_elements)

def plot(luminance, albedo, output, loss, i):
    fig, ax = plt.subplots(1, 3)  
    print(f'batch: {i}, loss: {loss}')
    
    output = prepare(output, normalize)
    gt = prepare(albedo, normalize)
    lum_in = prepare(luminance, normalize)

    plt.gray()
    ax[0].imshow(output)
    ax[0].set_title('Model Output')

    ax[1].imshow(gt)
    ax[1].set_title('Ground Truth')

    ax[2].imshow(lum_in)
    ax[2].set_title('Luminance Input')

    for i in range(3):
        ax[i].axis('off')
    fig.set_size_inches(15, 5)

    plt.show()

def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)