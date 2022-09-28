import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from external_library import InceptionResnetV1, MTCNN


class DatasetBasedOnFolders():
    def __init__(self, cfg):
        pretrained = cfg["pretrained"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device is {device}")
        self.face_feature_extractor = InceptionResnetV1(
            pretrained=pretrained,
            device=device
        ).eval()
        
        image_size = cfg["image_size"]
        print(f'image size: {image_size}')
        self.face_detector = MTCNN(
            image_size=image_size,
            margin=0,
            keep_all=False,
            min_face_size=40,
            device=device
        )
        
        source_path = './'+cfg['folder_name_for_source']
        self.path_for_image = os.path.join(
            source_path,
            cfg['folder_name_for_images']
        )
        print(f'Loading faces from {self.path_for_image}')
        
        # image_rotation_angle = cfg["image_rotation_angle"]
        # transformer = transforms.Compose(
        #     [
        #         transforms.Resize(image_size),
        #         # transforms.RandomHorizontalFlip(p=0.5),
        #         # transforms.RandomVerticalFlip(p=0.5),
        #         # transforms.RandomRotation((0, image_rotation_angle)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        #         ),  # normalize to Imagenet mean and std
        #     ]
        # )
        self.path_numpy_emb = os.path.join(
            source_path,
            cfg['file_name_to_save_numpy']+'_emb.npy'
        )
        self.path_numpy_label = os.path.join(
            source_path,
            cfg['file_name_to_save_numpy']+'_label.npy'
        )
        
    def CreateNumpy(self):
        self.image_dataset = datasets.ImageFolder(self.path_for_image)

        self.idx_to_class = {i: c for c, i
                             in self.image_dataset.class_to_idx.items()}

        for i, name in self.idx_to_class.items():
            print(i, name)
            
        data_length = len(self.image_dataset)
        print(f"data length: {data_length}")
        
        def collate_fn(x):
            return x[0]
        
        dataloader = DataLoader(self.image_dataset,
                                collate_fn=collate_fn)
        
        emb_numpy_dataset = []
        idx_numpy_dataset = []
        for i, (img, idx) in enumerate(dataloader):
            print(f'[{i:4}]converting to numpy...')
            # print(type(img), img)
            # print(type(idx))
            face, prob = self.face_detector(img, return_prob=True)
            # print(face.shape, type(face), prob)
            emb = self.face_feature_extractor(face.unsqueeze(0))
            # print(emb.shape, type(emb))
            emb_numpy_dataset.append(emb.detach().numpy())
            idx_numpy_dataset.append(idx)
            
        self.emb_numpy_dataset = np.asarray(emb_numpy_dataset)
        self.idx_numpy_dataset = np.asarray(idx_numpy_dataset)
        
        print(self.emb_numpy_dataset.shape)
        print(self.idx_numpy_dataset.shape)

        with open(self.path_numpy_emb, 'wb') as f:
            np.save(f, self.emb_numpy_dataset)
        with open(self.path_numpy_label, 'wb') as f:
            np.save(f, self.idx_numpy_dataset)
        
    def loadNumpy(self):
        with open(self.path_numpy_emb, 'rb') as f:
            self.emb_numpy_dataset = np.load(f)
        with open(self.path_numpy_label, 'rb') as f:
            self.idx_numpy_dataset = np.load(f)
            
        print(self.emb_numpy_dataset.shape)
        print(self.idx_numpy_dataset.shape)
        
    def ConvertNumpyToDataloader(self):
        return 0