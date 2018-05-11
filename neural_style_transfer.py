# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:57:31 2018

@author: alutes
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from torch.autograd import Variable

import copy


#######################################
# Content Loss
#######################################

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
        

#######################################
# Style  Loss
#######################################

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
    
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    
        G = torch.mm(features, features.t())  # compute the gram product
    
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
        
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
        

#######################################
# Load Model
#######################################
        
# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.std = torch.FloatTensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
        


#######################################
# Loss Function
#######################################
        
def get_style_model_and_losses(cnn, style_imgs, # should be a list of image tensors
                               content_img,
                               content_layers,
                               style_layers):
    cnn = copy.deepcopy(cnn)
    
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            print('Unrecognized layer: {}'.format(layer.__class__.__name__))
            continue

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(Variable(content_img)).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Add loss from each style picture as a separate layer
            for img_num,style_img in enumerate(style_imgs):
                # add style loss:
                target_feature = model(Variable(style_img)).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{0}_{1}".format(i,img_num), style_loss)
                style_losses.append(style_loss)

    return model, style_losses, content_losses
    
    
    
#######################################
# Input Images
#######################################
    
    
def run_style_transfer(cnn, 
                       content_img, style_imgs, # should be a list of image tensors 
                       input_img_var, 
                       content_layers=['conv_4'],
                       style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5','conv6'],
                       num_steps=50,
                       style_weight=1000000, content_weight=1, print_iters=10):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_imgs, content_img, content_layers, style_layers)
    
    # Think about changing this
    optimizer = optim.LBFGS([input_img_var])

    print('Optimizing..')
    for run in range(num_steps):
        print(run)

        def closure():
            # correct the values of updated input image
            input_img_var.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img_var)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            return style_score + content_score

        score = optimizer.step(closure)
        
        if run % print_iters == 0:
            print("run {}:".format(run))
            print('Loss: {}'.format(score.data.numpy()[0]))
            print()
            plt.figure()
            imshow(input_img_var.data, title='Content Image')

        
        
    # a last correction...
    input_img_var.data.clamp_(0, 1)

    return input_img_var.data
    
    
#######################################
# Image Display
#######################################

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated