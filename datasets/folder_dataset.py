import torch
import torchvision
from torchvision import datasets, utils, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image



class Folder_Dataset():
  def __init__(self, cfg):
    self.data = cfg['data_path']
    self.train_ratio = cfg['train_ratio']
    self.val_ratio = cfg['val_ratio']
    self.batch_size = cfg['batch_size']
    self.image_size = cfg['image_size']
    self.image_rotation_angle = cfg['image_rotation_angle']

    print(self.image_size)
 
    if cfg['transformer'] is True:
      self.transformer = transforms.Compose([
              transforms.Resize(self.image_size), 
              transforms.RandomHorizontalFlip(p=0.5),
              # transforms.RandomVerticalFlip(p=0.5),
              transforms.RandomRotation((0, self.image_rotation_angle)),
              transforms.ToTensor(), # convert a PIL image or ndarray to tensor
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalize to Imagenet mean and std
      ])
    else:
      self.transformer = None

    self.image_dataset = datasets.ImageFolder(self.data, transform = self.transformer)

    self.idx_to_class = {i:c for c,i in self.image_dataset.class_to_idx.items()}
    # print(type(self.idx_to_class))
    print(" data ")
    for i, name in self.idx_to_class.items():
        print(i, name)

    print(f'batch_size: {self.batch_size} \n')

  def createDataLoaders(self):
    dataset_size = len(self.image_dataset)
    train_size = int(dataset_size * self.train_ratio)
    val_size = int(dataset_size * self.val_ratio)
    test_size = dataset_size - train_size - val_size
    print('dataset length: ({}) = tr ({}) + val ({}) + tt ({})'.format(dataset_size, train_size, val_size, test_size))

    train_dataset, val_dataset, test_dataset = random_split(self.image_dataset, [train_size, val_size, test_size])
    
    def collate_fn(x):
        return x[0]
    
    if self.transformer is not None:
      collate_fn = None


    train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle=True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, collate_fn = collate_fn)

    #output type has to be dictionary
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test' : test_dataloader}
    dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}

    return dataloaders, dataset_sizes, self.idx_to_class