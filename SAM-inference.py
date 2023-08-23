'''
Tutorial: https://medium.com/@rekalantar/segment-anything-model-sam-for-medical-image-segmentation-9344ba57f2ca
'''

import os
import glob
import monai
import torch
import argparse
import numpy as np 
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk
from statistics import mean
from torch.optim import Adam
from natsort import natsorted
import matplotlib.pyplot as plt
from transformers import SamModel 
import matplotlib.patches as patches
from transformers import SamProcessor
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize


from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
)

from samDataset import SAMDataset
from samDataset import get_bounding_box

device = torch.device("cuda:0")


def getDataPaths(args):
    '''
    Get the paths to the training, validation and testing datasets
    of the 2d slice images (and their ground truth masks).
    '''
    # Initialize dictionary for storing image and label paths
    data_paths = {}
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images', '2d_masks']

    # Create directories and print the number of images and masks in each
    for dataset in datasets:
        for data_type in data_types:
            # Construct the directory path
            dir_path = os.path.join(args.base_dir, f'{dataset}_{data_type}')
            
            # Find images and labels in the directory
            files = sorted(glob.glob(os.path.join(dir_path, args.organ+"*.nii.gz")))
            
            # Store the image and label paths in the dictionary
            data_paths[f'{dataset}_{data_type.split("_")[1]}'] = files

    print('Number of training images', len(data_paths['train_images']))
    print('Number of validation images', len(data_paths['val_images']))
    print('Number of test images', len(data_paths['test_images']))

    return data_paths


def visualizeExample(example):
    '''
    Visualize an example image (2d slice that contains ROI) from the data,
    with the corresponding ground truth mask and prompt box.
    '''
    for k,v in example.items():
        if k != "name" :
            print(k,v.shape)

    xmin, ymin, xmax, ymax = get_bounding_box(example['ground_truth_mask'])

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(example['pixel_values'][1], cmap='gray')
    axs[0].axis('off')

    axs[1].imshow(example['ground_truth_mask'], cmap='copper')

    # create a Rectangle patch for the bounding box
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

    # add the patch to the second Axes
    axs[1].add_patch(rect)

    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig('example_image.png')





def makePredictions(args, img_type, test_loader):
    '''
    Predictions with the SAM model and inference
    '''

    # load pretrained weights
    model = SamModel.from_pretrained("facebook/sam-vit-base")   #sam-vit-huge
    model.to(device)
    
    # Iteratire through test images
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].cuda(),
                        input_boxes=batch["input_boxes"].cuda(),
                        multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().cuda()
    #         loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                        

            # apply sigmoid
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            
            name = batch["name"][0]

            # save prediction
            #medsam_seg_img = nib.Nifti1Image(medsam_seg, affine=np.eye(4))
            #nib.save(medsam_seg_img, './output/predicted-masks/'+name+'_predicted_mask.nii')
            saveSlicePrediction(args, img_type, name, medsam_seg)


            plt.figure(figsize=(12,4))
            plt.suptitle(name, fontsize=14)
            plt.subplot(1,3,1)
            plt.imshow(batch["pixel_values"][0,1], cmap='gray')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(batch["ground_truth_mask"][0], cmap='copper')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(medsam_seg, cmap='copper')
            plt.axis('off')
            plt.tight_layout()
            

            #plt .savefig('./output/plots/'+name+'-prediction.png')

            plt.savefig('./output/plots/model-prediction.png')

            

def saveSlicePrediction(args, img_type, name, medsam_seg):
    '''
    Save predicted labels in 3D, in nii.gz format (so that they could
    be used as pseudo labels in the future)
    '''
    print(name)
    organ, patient, slice_ind = name.split('_')
    slice_ind = int(slice_ind)

    # get path to (initially) empty mask and load mask data
    pred_masks_dir = args.empty_masks_dir + organ + '_' + img_type + '/'
    path = pred_masks_dir + organ + '_' + patient + '.nii.gz'
    mask = nib.load(path)
    mask_data = mask.get_fdata()

    # next add the predicted slice to the correct spot
    mask_data[slice_ind] = medsam_seg
    mask_img = nib.Nifti1Image(mask_data, affine=mask.affine)
    nib.save(mask_img, path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        default = './content',
                        help='The path to where the 2d slices are stored'
                        )
    parser.add_argument('--organ',
                        default='',
                        help='For which organ the inference should be made [liver,lung,pancreas,hepaticvessel,spleen,colon]. Default is all organs.'
                        )
    parser.add_argument('--data_dir',
                        default='/l/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/',
                        help = 'directory where to find MSD data')
    parser.add_argument('--empty_masks_dir',
                        default = './output/predicted-masks/',
                        help='The path where the empty 3D masks are stored'
                        )
    args = parser.parse_args()

    data_paths = getDataPaths(args)

    # SamProcessor for image preprocessing
    # create an instance of the processor for image preprocessing
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print(processor)

    test_dataset = SAMDataset(image_paths=data_paths['test_images'],
                              mask_paths=data_paths['test_masks'],
                              processor=processor
                              )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # see an example of the test data
    visualizeExample(test_dataset[60])

    # make predictions
    makePredictions(args, 'test', test_loader)



        




if __name__ == "__main__":
    main()