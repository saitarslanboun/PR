from __future__ import print_function, division
import json
import os
import torch
import cPickle as pickle
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class OutfitsDataset(Dataset):
    def __init__(self, input_file, output_file, root_dir, transform = None):
        with open(input_file,'r') as f:
            self.input_frame = f.readlines()
        with open(output_file,'r') as f:
            self.output_frame = f.readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_frame)

    def __getitem__(self, idx):
        input_data = self.input_frame[id]
        output_data = self.output_frame[id]
        
        input_embeddings = []
        output_words = []
        
        for word in input_data.split():
            emb_path = self.root_dir + "/data/" + word.split('_')[0] + "/" + word.split('_')[1] + ".p"
            input_embeddings.append(pickle.load(open(emb_path, "rb")))
            
        for word in output_data.split():
            output_words.append(word)
        
        return torch.from_numpy(np.array(input_embeddings)), torch.from_numpy(np.array(output_words)), len(output_data.split())