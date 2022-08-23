from external_lib import MTCNN, InceptionResnetV1
from torchsummary import summary
import torch

class Builder():
  def __init__(self, cfg):
    self.pretrained = cfg["pretrained"]
    self.num_classes = cfg["num_classes"]
    self.classify = cfg["classify"] 
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is {self.device}')

    # self.single_face_detector = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
    # self.multi_faces_detector = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    self.face_feature_extractor = InceptionResnetV1(pretrained=self.pretrained, num_classes=self.num_classes, classify=self.classify, device=self.device) 

    
    #Transfer Learning
    for param in self.face_feature_extractor.parameters():
      param.requires_grad = False
      # print(f'{param.requires_grad}')
    
    for param in self.face_feature_extractor.logits.parameters():
      param.requires_grad = True
      # print(f'{param.requires_grad}')
    
  
    print('Loading model was just completed.')

  def setModel(self, pretrained, num_classes, classify):
    self.pretrained = pretrained
    self.num_classes = num_classes
    self.classify = classify

    self.face_feature_extractor = InceptionResnetV1(pretrained=self.pretrained, num_classes=self.num_classes, classify=self.classify, device=self.device) 
    
  def getModel(self):
    return self.face_feature_extractor

  def summary(self, size):
    summary(self.face_feature_extractor, size)
  


    