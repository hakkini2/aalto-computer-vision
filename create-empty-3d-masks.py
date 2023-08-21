import os
import SimpleITK as sitk
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import nibabel as nib

def getDataDicts(args, img_type):
    list_img = []
    list_lbl = []
    list_name = []

    for line in open(args.data_txt_path + '_' + img_type + '.txt'):
        name = line.strip().split()[1].split('.')[0]
        list_img.append(args.data_dir + line.strip().split()[0])
        list_lbl.append(args.data_dir + line.strip().split()[1])
        list_name.append(name)
    data_dicts = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(list_img, list_lbl, list_name)]
    
    print(img_type, 'len {}'.format(len(data_dicts)))

    return data_dicts



def createEmptyMaskFiles(args, img_type, data_dicts):
    # create folder
    folder_name = args.data_txt_path.split('/')[2] + '_' + img_type
    dir_path = os.path.join(args.save_dir, folder_name)
    os.makedirs(dir_path, exist_ok=True)
    
    dir_2d_slices = args.slices_path + img_type + '_2d_masks/'

    for idx, item in enumerate(data_dicts):
        name = item['name'].split('/')[3]

        # do only for files that have predictions
        if any(fname.startswith(name) for fname in os.listdir(dir_2d_slices)):
            # get paths to image and label
            img_path = item['image']
            lbl_path = item['label']
            
            # load image and label
            img = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(lbl_path)
            print('processing patient', idx, img.GetSize(), mask.GetSize())

            # Get the mask data as numpy array
            mask_data = sitk.GetArrayFromImage(mask)

            num_slices, _, _ = mask_data.shape
            empty_mask = np.zeros([num_slices, 256, 256]) # hard coded shape for now !!

            # save the empty array as nii.gz
            empty_mask_path = os.path.join(dir_path, os.path.basename(lbl_path))
            #sitk.WriteImage(empty_mask, empty_mask_path)
            empty_mask_img = nib.Nifti1Image(empty_mask, affine=np.eye(4))
            nib.save(empty_mask_img, empty_mask_path)
        
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default = './output/predicted-masks/', help='The path where to save the predicted masks in 3D')
    parser.add_argument('--data_dir', default='/l/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/', help = 'directory where to find MSD data')
    parser.add_argument('--data_txt_path', default = './dataset/MSD_Task09_Spleen', help = 'path to txt file describing the train/val/test spits')
    parser.add_argument('--slices_path', default='./content/', help='where to find the created 2d slice folders such as test_2d_masks, test_2d_images, etc')
    args = parser.parse_args()

    # get data dicts of train, val and test splits
    data_dicts_train = getDataDicts(args, 'train')
    data_dicts_val = getDataDicts(args, 'val')
    data_dicts_test = getDataDicts(args, 'test')

    # create empty 3D label files (same shape as original label) so that
    # the predicted 2d slices can be updated in later
    #createEmptyMaskFiles(args, 'train', data_dicts_train)
    #createEmptyMaskFiles(args, 'val', data_dicts_val)
    createEmptyMaskFiles(args, 'test', data_dicts_test)

    

if __name__ == "__main__":
    main()