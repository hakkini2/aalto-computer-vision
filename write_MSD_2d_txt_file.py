import os
import argparse

def writeFile(args):
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images', '2d_masks']

    with open('./dataset/MSD_2d.txt', 'w') as file:
        #loop through test, val and train folders
        for dataset in datasets:
            print('Writing',dataset,'images')
            # get image and mask directories
            directory_img = os.path.join(args.base_dir, f'{dataset}_2d_images')
            directory_mask = os.path.join(args.base_dir, f'{dataset}_2d_masks')

            #get contents of the directories in lists
            images = sorted(os.listdir(directory_img))
            masks = sorted(os.listdir(directory_mask))
            # loop all the files in directory
            for i in range(len(images)):
                # get paths to images and masks
                img_filename = images[i]
                mask_filename = masks[i]
                path_to_img = os.path.join(directory_img, img_filename)
                path_to_mask = os.path.join(directory_mask, mask_filename)

                # check that image and mask matches before writing
                img_name = path_to_img.split('/')[-1]
                mask_name = path_to_mask.split('/')[-1]

                if img_name == mask_name:
                    file.write(path_to_img + ' ' + path_to_mask + '\n')
                else:
                    print('Image and mask names did not match, files will be skipped.')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        default = './content',
                        help='The path to where the 2d slices are stored'
                        )
    args = parser.parse_args()

    writeFile(args)


if __name__ == "__main__":
    main()