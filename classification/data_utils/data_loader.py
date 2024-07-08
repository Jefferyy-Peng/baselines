import sys
from torchvision import transforms
import numpy as np

sys.path.append('..')

from PIL import Image
from torch.utils.data import Dataset


class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, mode, label_dict=None, channels=1, transform=None):
    zero = 0
    one = 0
    for id, path in enumerate(path_list):
      label = label_dict[path]
      if label == 0:
        zero += 1
      else:
        one += 1
    self.mode = mode
    self.class_weights = 1. / (np.array([zero, one])/len(path_list))
    self.path_list = path_list
    self.label_dict = label_dict
    self.transform = transform
    self.channels = channels


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
    # Get image and label
    # image: D,H,W
    # label: integer, 0,1,..
    if self.channels == 1:
      image = Image.open(self.path_list[index]).convert('L')
    elif self.channels == 3:
      image = Image.open(self.path_list[index]).convert('RGB')
    if self.mode == 'val':
      seg = Image.open(self.path_list[index].replace('/images_illness_3c', '/labels_illness_3c'))

    if self.transform is not None:
      image = self.transform(image)
    if self.mode == 'val':
      seg = self.transform(seg)

    if self.label_dict is not None:
      label = self.label_dict[self.path_list[index]] 
      if self.mode == 'val':
        sample = {'image':image, 'label':int(label), 'seg': seg}
      else:
        sample = {'image': image, 'label': int(label)}
    else:
      sample = {'image':image}
    
    return sample
