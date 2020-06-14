import os
import numpy as np
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
from PIL import Image as PILImage
import pickle


class FL_data(data.Dataset):
    def __init__(self, **kwargs):
        self.dir = kwargs['data_dir']
        self.stage = kwargs['stage']
        self.create_dataset = kwargs['create_dataset']
        self.img_size = kwargs['img_size']
        if self.create_dataset:
            print("We are building the dataset")
            self.df = self.build_dataset(dir=self.dir, image_size = self.img_size)
        else:
            self.df = pickle.load(open('combined_dataset.pkl', 'rb'))

    def build_dataset(self, dir, image_size):
        combined_dataset = []
        for folder in os.listdir(dir):
            path_fold = os.path.join(dir, folder)
            if folder == 'val':
                for directory in os.listdir(path_fold):
                    if directory == '.DS_Store':
                        os.remove(os.path.join(path_fold, directory))
                        continue
                    path = os.path.join(path_fold, directory)
                    print(directory)
                    for im_dir in os.listdir(path):
                        if im_dir == '.DS_Store':
                            os.remove(os.path.join(path, im_dir))
                            continue
                        img_path = os.path.join(path, im_dir)
                        img = PILImage.open(img_path)  # shape (913,1064,3)
                        img = img.convert('RGB')
                        transform = transforms.Compose([transforms.Resize(image_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                                                             std=[1, 1, 1])])
                        # plt.imshow(img)
                        img = transform(img)
                        if directory == 'COVID19':
                            combined_dataset.append([img, 1])
                        else:
                            combined_dataset.append([img, 0])
            else:
                continue
        with open('combined_dataset.pkl', 'wb') as myfile:
            pickle.dump(combined_dataset, myfile)

        return combined_dataset
