# Auto_Colorizer

### Import Libraries


```python
import os
import shutil
import time
import splitfolders
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline
from skimage import io
from skimage.color import rgb2gray, rgb2lab, lab2rgb
```

### Split the Data


```python
#splitfolders.ratio(r'C:\Users\howle\OneDrive\Documents\indoorCVPR_09\Images', output=r"C:\Users\howle\OneDrive\Documents\indoor_scenes_data",seed=123,ratio=(.8, 0.1,0.1))

#splitfolders.ratio(r'Source folder of dataset', output=r"Destination folder for data splits",seed=123,ratio=(.8, 0.1,0.1))
```

### Check out a Sample Image


```python
from IPython.display import Image, display
display(Image(filename=r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\train\train_images\airport_inside_0003.jpg'))
```


![jpeg](output_6_0.jpg)


### Clear the Cache


```python
# use_gpu = torch.cuda.is_available()
# print(use_gpu)
torch.cuda.empty_cache()
```


```python
# models.resnet18(num_classes=365)
# resnet = models.resnet18(num_classes=365)
# resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))

# resnet.conv1.weight.size()
```

### Model Architecture


```python
class Auto_Colorizer(nn.Module):
    
    '''This model extracts features from grayscale images
        using the resnet-18 model architecture as the encoder.
        
        The decoder is a series of convolutional layers with upsampling.
        
        Model accepts the Lightness channel from LAB iamges.
        
        This model is not utilizing the pretrained 
        weights from resnet-18, so must be retrained on training data.
        
        Generatates A&B color channels in the LAB colorspace'''
    
    def __init__(self):
        
        '''This is the encoder, a modified, untrained resnet-18 architecture.'''
        
        super(Auto_Colorizer, self).__init__()
        resnet = models.resnet18(pretrained=False) 
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        self.upsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):

        midlevel_features = self.midlevel_resnet(input)
        output = self.upsample(midlevel_features)
        return output
```

### Viewing the model Architecture


```python
model = Auto_Colorizer().cuda()
model
```




    Auto_Colorizer(
      (midlevel_resnet): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (upsample): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Upsample(scale_factor=2.0, mode=nearest)
        (4): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Upsample(scale_factor=2.0, mode=nearest)
        (11): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (12): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (13): ReLU()
        (14): Conv2d(32, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): Upsample(scale_factor=2.0, mode=nearest)
      )
    )



### Some Hyperparameters


```python
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
best_losses = 10000000000
epochs = 2
```

### Helper Functions

#### Grayscale Image Conversion


```python
class GrayscaleImageFolder(datasets.ImageFolder):
    '''Image folder, which converts images to grayscale.
        
        Outputs original image, A&B channels and the target'''

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target
```

#### Dataloaders


```python
# Training Dataloader, Shuffled
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = GrayscaleImageFolder(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

# Validation Dataloader, Shuffled
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=True)

# Test Dataloader, Not Shuffled
test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
test_imagefolder = GrayscaleImageFolder(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\test' , test_transforms)
test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=64, shuffle=False)
```

#### Pytorch Average Meter Loss Calculator


```python
class AverageMeter(object):
    
    '''Class for calculating losses'''
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```


```python
def to_rgb(grayscale_input, ab_input, save_path, save_name):
    
    '''Takes in the grayscale and a&b channels,
        converts to RGB, 3-channel image,
        saves images to specified folder
        
        This allows us to view the results of the colorization by the model'''
    
    plt.clf()
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
```


```python
def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):
                save_path = {'grayscale': r"C:\Users\howle\OneDrive\Documents\indoor_scenes_data\outputs\gray", 'colorized': r"C:\Users\howle\OneDrive\Documents\indoor_scenes_data\outputs\color"}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
                
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses))

    print(f'Validation loss: {losses.avg}')
    return losses.avg
```


```python
def train(train_loader, model, criterion, optimizer, epoch):
    
    print('Starting training epoch {}'.format(epoch))
    model.train()
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()
        data_time.update(time.time() - end)
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    print('Epoch {} complete.'.format(epoch))
```


```python
os.makedirs(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\outputs\color', exist_ok=True)
os.makedirs(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\outputs\gray', exist_ok=True)
os.makedirs(r'C:\Users\howle\OneDrive\Documents\indoor_scenes_data\checkpoints', exist_ok=True)
save_images = True
```


```python
for epoch in range(epochs):
    train(train_loader, model, criterion, optimizer, epoch)
    with torch.no_grad():
        losses = validate(val_loader, model, criterion, save_images, epoch)
        
    if losses < best_losses:
        best_losses = losses
        torch.save(model.state_dict(), 'C:/Users/howle/OneDrive/Documents/indoor_scenes_data/checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
```


```python
#model.state_dict()
```


```python
pretrained = torch.load(r"C:\Users\howle\OneDrive\Documents\indoor_scenes_data\checkpoints\model-epoch-1-losses-0.003.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(pretrained)

# Colorizing on test data using the final model
save_images = True
with torch.no_grad():
    validate(test_loader, model, criterion, save_images, 0)
```
