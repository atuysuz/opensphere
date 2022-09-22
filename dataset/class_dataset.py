import random
import os.path as osp
import numpy as np

from .utils import image_pipeline
from torch.utils.data import Dataset


class ClassDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,
            test_mode=False, noise_ratio=None, seed=None):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode
        self.noise_ratio = noise_ratio
        self.seed = seed

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from a provided annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        self.data_path_list = []
        self.data_label_list = []

        for line in lines:
            path, name = line.rstrip().split()
            self.data_path_list.append(path)
            self.data_label_list.append(name)

        if len(self.data_path_list) == 0:
            raise (RuntimeError('Found 0 files.'))

        self.data_path = np.array(self.data_path_list).astype(np.string_)

    def corrupt_label(self):
        random.seed(self.seed)
        for idx, item in enumerate(self.label_items):
            if random.random() > self.noise_ratio:
                continue
            self.label_items[idx] = np.random.choice(self.label_items)

    def get_label(self):
        """ convert name to label,
            and optionally permutate some labels
        """
        names = self.data_label_list.copy()
        names.sort()
        self.classes = names
        name2label = {name: idx for idx, name in enumerate(names)}

        self.label_items_list = []
        for item in self.data_label_list:
            label = name2label[item]
            self.label_items_list.append(label)

        self.label_items = np.array(self.label_items_list).astype(np.string_)

        if self.noise_ratio:
            self.corrupt_label()

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_path[idx]
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)
        label = self.label_items[idx]

        return image, label

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        return self.prepare(idx)
