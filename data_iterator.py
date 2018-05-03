from __future__ import print_function, division
import json
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class OutfitsDataset(Dataset):
    def __init__(self, json_file, root_dir, transform = None):
        self.outfits_frame = json.load(open(json_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.outfits_frame)

    def __getitem__(self, idx):
        json_element = self.outfits_frame[idx]
        outfit_id = json_element["outfit_id"]
        
        # Load in the input images
        input_images = []
        for input_id in json_element["input_ids"]
            img_path = self.root_dir + "/data/" + outfit_id + "/" + input_id + ".png"
            input_images.append(io.imread(img_path))

        # Load in the output images
        output_images = []
        for output_id in json_element["output_ids"]
            img_path = self.root_dir + "/data/" + outfit_id + "/" + output_id + ".png"
            output_images.append(io.imread(img_path))

        # Return the sample as a dictionary which contains input images and output images
        return {"input_imgs":input_images, "output_imgs":output_images}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
