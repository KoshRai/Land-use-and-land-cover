import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import torchinfo
import torchvision
import torchvision.transforms as T
import math
import torchmetrics
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


xpath='/home/20bce131/Turkey/img_dir'
ypath='/home/20bce131/Turkey/ann_dir/'
trn_hist_path='/home/20bce131/model/train_hist'
trn_out_path='/home/20bce131/model/trn_imgs_out'
val_out_path='/home/20bce131/model/val_imgs_out'






def visualize_sample(img, label, title, path=''):
    fig, ax = plt.subplots(1, 6, figsize=(20,20))
    for i, subplot_ax in zip(range(5 + 1), ax.flatten()):
        if i == 0: 
            subplot_ax.imshow(img.permute(1,2,0))
            subplot_ax.set_title(title)
        else:
            subplot_ax.imshow(label[i-1], cmap='gray', vmin=0, vmax=1)
            subplot_ax.set_title(f'Label {i}')
    if path != '':
        plt.savefig(f'{path}/{title}.png')

def visualize_predictions(model, img, i, path):
    inp = (img - 127.5) / 255
    inp = torch.unsqueeze(inp, dim=0)
    pred = model(inp).to('cpu')
    visualize_sample(img, pred.squeeze().detach().numpy(), f'Model Predictions {i}', path)
    
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5).to(device)

def acc(preds, label):
    return accuracy(preds, label)





class SARData(Dataset):

    def get(self, idx):        
        xtrain = Image.open(f'{self.xtrain_path}/{self.imgs[idx]}').convert('RGB')
        xtrain = torch.from_numpy(np.array(xtrain)).permute(2,0,1) #HWC -> CHW
        
        ytrain = Image.open(f'{self.ytrain_path}/{self.imgs[idx]}')
        ytrain = torch.from_numpy(np.array(ytrain))
        ytrain = self.process_label(ytrain)
        
        return xtrain, ytrain
    
    def process_label(self, label):
        r = []
        for i in range(self.num_classes):
            mask = label == i+1
            mask = mask.float()
            r.append(mask)
        return torch.stack(r)
        
    def __init__(self):
        self.num_classes = 5
        self.height, self.width = 256,256
        self.xtrain_path = xpath
        self.ytrain_path = ypath
        self.imgs = np.array(os.listdir(self.ytrain_path))
        
        self.xtrain, self.ytrain = [], []
        for i in range(len(self.imgs)):
            t = self.get(i)
            self.xtrain.append(t[0])
            self.ytrain.append(t[1])
        self.xtrain = torch.stack(self.xtrain).to(device)
        self.ytrain = torch.stack(self.ytrain).to(device)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]
        
    def __len__(self):
        return len(self.imgs)





data = SARData()
train, val = torch.utils.data.random_split(data, [0.8, 0.2])






def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return conv

def double_Tconv(in_channels, out_channels):
    Tconv = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Dropout2d(p=0.2),
        nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return Tconv

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.Tconv5 = double_Tconv(1024, 512)
        self.Tconv4 = double_Tconv(1024, 256)
        self.Tconv3 = double_Tconv(512, 128)
        self.Tconv2 = double_Tconv(256, 64)
        self.Tconv1 = double_Tconv(128, 5)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        c1 = self.conv1(x)
        x1, i1 = self.max_pool(c1)

        c2 = self.conv2(x1)
        x2, i2 = self.max_pool(c2)
        
        c3 = self.conv3(x2)
        x3, i3 = self.max_pool(c3)

        c4 = self.conv4(x3)
        x4, i4 = self.max_pool(c4)
        
        i = self.conv5(x4)
        
        i = self.dropout(i)
        
        t4 = self.Tconv5(i)
    
        z4 = torch.cat((self.max_unpool(t4, i4, output_size=c4.size()), c4), dim=1)
        t3 = self.Tconv4(z4)
        
        z3 = torch.cat((self.max_unpool(t3, i3, output_size=c3.size()), c3), dim=1)
        t2 = self.Tconv3(z3)
    
        z2 = torch.cat((self.max_unpool(t2, i2, output_size=c2.size()), c2), dim=1)
        t1 = self.Tconv2(z2)
        
        z1 = torch.cat((self.max_unpool(t1, i1, output_size=c1.size()), c1), dim=1)
        out = self.Tconv1(z1)
        
        return self.softmax(out)




model = nn.DataParallel(Network(), [0,1])
print(torchinfo.summary(model, (1,3, 256, 256)))






epochs = 500
batch_size = 16
lr = 1e-2
momentum = 0.9
#weights = torch.tensor([0.0621, 0.0093, 0.0101, 0.9022, 0.0163]).to(device) * 5






model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

train_size = len(train)
val_size = len(val)
num_train_batches = train_size/batch_size
num_val_batches = val_size/batch_size

print(f'Epochs : {epochs}')
print(f'Batch size : {batch_size}')
print(f'Initial Learning rate : {lr}')
print(f'Momentum : {momentum}')

print(f'Training with {train_size} images')
print(f'Validation with {val_size} images')
print(f'Number of training minibatches = {int(num_train_batches)} per epoch')
print(f'Number of validation minibatches = {int(num_val_batches)} per epoch')

trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val, batch_size=batch_size, shuffle=True)

hist = {'train_loss':[], 
        'val_loss':[],
        'train_acc': [],
        'val_acc': []}

for epoch in range(epochs):
    print(f'{"-"*30}Epoch {epoch+1}/{epochs}{"-"*30}')
    
    total_train_loss = 0
    total_val_loss = 0
    total_train_acc = 0
    total_val_acc = 0
    
    #Training Loop
    model.train()
    for batch, (x,y) in enumerate(trainloader):
        x = (x-127.5) / 255
        #Forward prop
        preds = model(x)
        loss = loss_fn(preds, y)
        total_train_loss += loss.item()
        total_train_acc += acc(preds, y).cpu()
        
        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    #Validation loop
    model.eval()
    with torch.no_grad():
        for batch, (x,y) in enumerate(valloader):
            x = (x-127.5)/255
            #Inference
            preds = model(x)
            loss = loss_fn(preds, y)
            total_val_loss += loss.item()
            total_val_acc += acc(preds, y).cpu()
    if (epoch+1) % 20 == 0:
        scheduler.step()
    if (epoch+1) % 500 == 0:
        torch.save(model, f'model{epoch+1}.pt')
    
    
    avg_train_loss = total_train_loss / num_train_batches
    avg_val_loss = total_val_loss / num_val_batches
    avg_train_acc = total_train_acc / num_train_batches
    avg_val_acc = total_val_acc / num_val_batches
    
    print(f"Training Loss    : {avg_train_loss:<7f}      Training Accuracy    : {avg_train_acc:<7f}")
    print(f"Validation Loss  : {avg_val_loss:<7f}     Validation Accuracy  : {avg_val_acc:<7f}")
    
    hist['train_loss'].append(avg_train_loss)
    hist['train_acc'].append(avg_train_acc)
    hist['val_loss'].append(avg_val_loss)
    hist['val_acc'].append(avg_val_acc)






plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(hist['train_loss'], label='Training Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig(f'{trn_hist_path}/loss_curve.png')
plt.clf()






plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(hist['train_acc'], label='Training Accuracy')
plt.plot(hist['val_acc'], label='Validation Accuracy')
plt.legend()
plt.savefig(f'{trn_hist_path}/acc_curve.png')
plt.clf()






model.to(device)
print('Images from the validation set')
idx = torch.randint(0, val_size, (10,))
model.eval()
for i in (idx):
    img, label = val.__getitem__(i)
    img, label = img.cpu(), label.cpu()
    visualize_sample(img, label, f'Ground Truth {i}', path = val_out_path)
    visualize_predictions(model, img, i, val_out_path)






print('Images from the training set')
idx = torch.randint(0, train_size, (10,))
model.eval()
for i in (idx):
    img, label = train.__getitem__(i)
    img, label = img.cpu(), label.cpu()
    visualize_sample(img, label, f'Ground Truth {i}', trn_out_path)
    visualize_predictions(model, img, i, trn_out_path)