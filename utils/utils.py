import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# HELPER FUNCTIONS 


def calculate_dice_score(y_pred, y):
    '''
    https://github.com/ljwztc/CLIP-Driven-Universal-Model/blob/main/utils/utils.py

    y_pred: predicted labels, torch tensor of shape torch.Size([1, 1, 256, 256])
    y: ground truth labels, torch tensor of shape torch.Size([1, 1, 256, 256])
    
    dice = (2*tp)/(2*tp+fp+fn)
    '''
    # convert labels to 1 and 0
    y_pred = torch.where(y_pred > 0.5, 1, 0)

    # convert the tensors to 1D for easier computing
    predict = y_pred.contiguous().view(1, -1)
    target = y.contiguous().view(1,-1)

    #calculate true positives
    tp = torch.sum(torch.mul(predict,target))

    #calculate false negatives
    fn = torch.sum(torch.mul(predict!=1, target))

    #calculate false positives
    fp = torch.sum(torch.mul(predict, target!=1))

    #calculate true negatives
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    #dice = (2*tp)/(2*tp+fp+fn)
    dice = 2*tp/(torch.sum(predict) + torch.sum(target))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return dice, sensitivity, specificity



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
