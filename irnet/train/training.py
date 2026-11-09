import numpy as np
import sys
import imageio as io
import os
import torch, torch.nn as nn
import dataloader as dl
import irnet as irnet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--imagedir', type=str, default='')
parser.add_argument('--runnum', type=str, default='')
parser.add_argument('--res', type=str, default='')

args = parser.parse_args()
tdir = args.imagedir
res = args.res

run_num = f'run{args.runnum}'

if not os.path.exists(f'eevee-{res}/{run_num}'):
    os.makedirs(f'eevee-{res}/{run_num}', exist_ok = True)
    print(f'creating path eevee-{res}/{run_num}')
else:
    print('Path already exists')

print(tdir)
print(os.listdir(tdir))
sys.stdout.flush()

def get_device():
    # Set up to use multiple GPUs if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def transforms(array):
    array = array[..., np.newaxis]
    img = array.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

device = get_device()

# Dataset loading
imgs = dl.HDRDataset(input_dir=os.path.join(tdir, 'train/luminance/'),
                     gt_dir=os.path.join(tdir, 'train/albedo/'),
                     transforms=transforms
                    )
train_loader = DataLoader(imgs, batch_size=16, shuffle=True, num_workers=4)

# Model setup
model = irnet.IRNet()

# If multiple GPUs are available, wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)
model.train()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

# Define number of epochs
num_epochs = 10

# Training loop with epoch
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for i, (luminance, albedo) in enumerate(train_loader):
        optimizer.zero_grad()

        luminance = luminance.to(device)
        albedo = albedo.to(device)

        output = model(luminance)

        loss = criterion(output, albedo)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item()}')
            sys.stdout.flush()

    # Save the model at the end of each epoch
    model_save_path = f'eevee-{res}/{run_num}/irnet-{res}-{run_num}-v{epoch+1}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved as {model_save_path}')
    sys.stdout.flush()
