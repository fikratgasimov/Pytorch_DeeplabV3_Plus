import os
import numpy as np
import scipy.misc as m
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class ApolloScapeSegmentation(Dataset):

    NUM_CLASSES = 38

    """ 
    ApolloScape Dataset 
    
    """

    def __init__(self,
                 args,
                 root_dir=Path.db_root_dir('apolloscape'),
                 split='train',
                 ):

            super().__init__()

            # access the dataset via directory
            self._root_dir = root_dir
            
            # Original  Jpg images as input
            self._image_dir = os.path.join(self._root_dir, 'JPEGImages')
            
            # Grayscale images as output

            self._cat_dir = os.path.join(self._root_dir, 'Ground_Truth')


            self._cat_dir = os.path.join(self._root_dir, 'Ground_Truth' )

            
            # if split == train 
            if isinstance(split, str):
                self.split = [split] 

            # otherwise, sort files and find train dataset in directory
            else:
                split.sort()
                self.split = split
                
            # parse arguments 
            self.args = args

            # Defining train images  through__split__dir() - function
            # Input is '.jpg' and output is '.png'

            _splits_dir = os.path.join(self._root_dir, 'ImageSets', 'Segmentation')


            # create image_ids array,images as well as categories in order to append parameters into array
            self.im_ids = []
            self.images = []
            self.categories = []

            # loop through train file
            for splt in self.split:


                # then, concatenating files inside one another, find csv file and read i
                with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:

                    # initialize lines and read all lines of f function insider directory
                    lines = f.read().splitlines()


                # Enumurate lines through loop, once we have also input(.jpg) and output(.png) 
                # existed in the training file as well as txt file all together
                
                for ii, line in enumerate(lines):
                    

                    # read input images which is regarded as jpg
                    _image = os.path.join(self._image_dir, line + ".jpg")
                    _cat = os.path.join(self._cat_dir, line + ".png")
                    # print(ii)
                    # print(_image)



                    # check whether required _image file exist or not
                    assert os.path.isfile(_image)
                    
                    # check whether required _cat file exist or not
                    assert os.path.isfile(_cat)
                    
                    # then as initialized above, append line which is following path numerate
                    # and will be regarded as csv file, append it to im_ids
                    self.im_ids.append(line)
                    
                    # append _image with '.jpg' into images array
                    self.images.append(_image)
                    
                    # append _cat images with '.png' into categories array
                    self.categories.append(_cat)
               
            assert(len(self.images) == len(self.categories))

            # Display status
            print('Number of images in {}: {:d}'.format(split,  len(self.images)))

    # return Length of Ground truth images
    def __len__(self):
        return len(self.images)

    # Use getitem to access each image existed in folder # Use getitem to access each image existed in folder
    def __getitem__(self, index):

        # get an image and corresponding label
        _img, _target = self._make_img_gt_point_pair(index)

        # sample image and corresponding label and class id pair
        sample = {'image': _img, 'label': _target}
            
        # Accordingly, take images one by one from train
        for split in self.split:

            # transfom image-label pair while training
            if split == "train":
                return self.transform_tr(sample)

            # transform image-label pair by validating
            elif split == "val":
                return self.transform_val(sample)
                
                
       #  after accessing image-label pair, 
       # by calling this function we open image and convert to RGB

    def _make_img_gt_point_pair(self, index):
    
        # access JPG images
        _img = Image.open(self.images[index]).convert('RGB')
            
        # access PNG images
        _target = Image.open(self.categories[index])
        
        # return  achieved images with .jpg, and .png extensions
        return _img, _target


    # transform sample which image-label pair if training

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
        
                                     
    # transform sample which image-label pair if validating
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size = self.args.crop_size),
            tr.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    # return length of train images
    def __str__(self):
      return 'ApolloScape(split=' + str(self.split) + ')'

# __name__ == __main__ append desired different modules into dataset to deploy them
# in this dataset

if __name__ == "__main__":
    # import deeplabv3+ and Xception('Extreme Inception') concatenation


    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    apolloscape_train = ApolloScapeSegmentation(args, split='train')

    dataloader = DataLoader(apolloscape_train, batch_size=4, shuffle=True, num_workers=0)


    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='apolloscape')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
