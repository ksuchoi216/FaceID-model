import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def imshow_dataloader(dataloader):
  imgs, idxs = next(iter(dataloader))
  image_size = imgs[0].numpy().shape

  inp = make_grid(imgs)
  """Imshow for Tensor."""
  print(f'image dimension: {image_size}')
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.axis("off")
  plt.imshow(inp)
  plt.pause(0.001)

  return image_size

def imshow_denormalization(img, show_img = False):
  img = img[0]
  img = img.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = std * img + mean
  img = np.clip(img, 0, 1)
  if show_img == True:
    plt.axis("off")
    plt.imshow(img)
    plt.pause(0.001)
  
  return img


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show_pca(numpy_data_x, numpy_data_y):
  train_x = numpy_data_x
  train_y = numpy_data_y
  
  #standarize features
  scaler=StandardScaler()
  scaler.fit(train_x)
  standarized_x=scaler.transform(train_x)

  pca = PCA(n_components=2)
  pca.fit(standarized_x)

  pca_x=pca.transform(standarized_x)
  fig = plt.figure(1, figsize=(20, 10))
  plt.scatter(pca_x[:,0],pca_x[:,1],c=train_y)
  plt.axis('off')
  plt.show()

