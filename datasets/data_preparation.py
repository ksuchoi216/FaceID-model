import warnings

from numpy import save
warnings.filterwarnings("ignore")

import torch
import torchvision
from torchvision import datasets, utils, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image

import cv2
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from external_library import MTCNN

# from PIL import Image
import matplotlib.pyplot as plt

class DataPreparation():
  def __init__(self, cfg):
    self.path_folder_for_saved_images = cfg['path_folder_for_saved_images']
    self.path_folder_for_saving_cropped_faces =  cfg['path_folder_for_saving_cropped_faces']
    self.image_size = cfg['image_size']

    self.mtcnn = MTCNN(image_size=self.image_size, margin=0, keep_all=False, min_face_size=self.image_size*0.1) # keep_all=False
    self.mtcnn_show = MTCNN(select_largest=False, post_process=False)

    if not os.path.exists(self.path_folder_for_saving_cropped_faces):
      os.makedirs(self.path_folder_for_saving_cropped_faces)
      print('new folder was created in ', self.path_folder_for_saving_cropped_faces)
      
    self.filter_with_face_prob = cfg['filter_with_face_prob']
    self.face_prob_threshold_for_filter = cfg['face_prob_threshold_for_filter']

  def save_cropped_faces(self):
    print("Starting data load...")
    show_image = False
    image_dataset = datasets.ImageFolder(self.path_folder_for_saved_images)

    idx_to_class = {i:c for c,i in image_dataset.class_to_idx.items()} # accessing names of peoples from folder names

    for i, name in idx_to_class.items():
      print(i, name)
      path_ = self.path_folder_for_saving_cropped_faces + name + "/"
      if not os.path.exists(path_):
        os.makedirs(path_)

    def collate_fn(x):
        return x[0]
        
    data_loader = DataLoader(image_dataset, collate_fn=collate_fn)

    img_num = 0
    current_idx = 0
    for i, (img, idx) in enumerate(data_loader):
      if current_idx != idx:
        current_idx += 1
        img_num = 0
      img_num += 1

      if i % 30 == 0 and show_image == True:
        print("="*50)
        print('class: ', idx)
        face = self.mtcnn_show(img)
        face = face.permute(1,2,0).int().numpy()
        plt.imshow(face)
        plt.axis("off")
        plt.show()
        print("="*50)


      name = idx_to_class[idx]
      file_name = str(img_num) + '.png'
      save_path = self.path_folder_for_saving_cropped_faces + name+"/"+file_name

      if self.filter_with_face_prob is True:
        face, prob = self.mtcnn(img, return_prob=True)
        if face is not None and prob >= self.face_prob_threshold_for_filter:
          _ = self.mtcnn_show(img, save_path=save_path)    
          print('saved cropped face image in ',save_path)
      else:
        _ = self.mtcnn_show(img, save_path=save_path)
        print('saved cropped face image in ',save_path)
      # face = face.permute(1,2,0).int().numpy()
      # plt.imshow(face)
      # plt.axis("off")
      # plt.show()
      
  # def save_dataloader(self):
  #   print('starting...for ', self.path_folder_for_saving_cropped_faces)
  #   transformer = transforms.Compose([
  #           transforms.Resize(512), # resize, the smaller edge will be matched.
  #           transforms.RandomHorizontalFlip(p=0.5),
  #           transforms.RandomVerticalFlip(p=0.5),
  #           transforms.RandomRotation((0, 45)),
  #           transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
  #           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalize to Imagenet mean and std
  #   ])

  #   image_dataset = datasets.ImageFolder(self.path_folder_for_saving_cropped_faces, transform = transformer)
  #   # image_dataset = datasets.ImageFolder(self.path_folder_for_saving_cropped_faces)
  #   # dataset_size = len(image_dataset)
  #   # train_size = len(dataset_size * 0.8)
  #   # val_size = len(dataset_size * 0.1)
  #   # print(dataset_size)
  #   # test_size = dataset_size - train_size - val_size
  #   # print('dataset length: {} = tr {} + val {} + tt {}'.format(dataset_size, train_size, val_size, test_size))

  #   idx_to_class = {i:c for c,i in image_dataset.class_to_idx.items()} # accessing names of peoples from folder names

  #   for i, name in idx_to_class.items():
  #     print(i, name)


  #   # print(self.save_path)
  #   # try:
  #   #   torch.save(self.data_loader, self.save_path)
  #   #   print('dataloader was successfully saved to ' + self.save_path)
  #   # except:
  #   #   print('failed saving dataloader')



    
