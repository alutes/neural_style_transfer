# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:44:55 2018

@author: alutes
"""

from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from neural_style_transfer import run_style_transfer
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

from matplotlib import pyplot as plt

#######################################
# Image load
#######################################

# For set up I'll run on cpu, make the code flexible for when we go to gpu
device = torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_available = torch.cuda.is_available()

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize([imsize,imsize]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    if gpu_available:
        image = image.gpu()
    return image    

    
#######################################
# Get Content and Style
#######################################
    
# Style images
style_imgs = []
for filename in os.listdir("./monet/"):
    style_imgs.append(image_loader("./monet/"+filename))

# Content Image
content_img = image_loader("./twins.jpg")

# Show image
unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    
# Print each image
imshow(style_imgs[0])
imshow(content_img)


#######################################
# Load Pre-trained model
#######################################

# Load pre-trained model
#cnn_model = models.resnet18(pretrained=True)
cnn_model = models.vgg19(pretrained=True)
cnn = cnn_model.features.eval()

# desired depth layers to compute style/content losses :
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5','conv6']

# Start with either noise or just the original content image
#input_img = content_img.clone()
input_img = torch.randn(content_img.size())
input_img_var = Variable(input_img)
input_img_var.requires_grad=True

# Run the output
output = run_style_transfer(cnn, content_img, style_imgs, input_img_var, num_steps=10)
plt.figure()
imshow(output, title='Output Image')


