import os 
from PIL import Image 
from torch.utils.data import Dataset 
from torchvision import transforms

class ImageDataset(Dataset):
  def __init__(self, dir, shape):
    self.dir = dir
    self.img_list = os.listdir(dir)
    w_shape, h_shape = shape

    self.in_transform = transforms.Compose([transforms.Resize((w_shape // 4, h_shape // 4), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    self.ou_transform = transforms.Compose([transforms.Resize(shape, Image.BICUBIC), 
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    image = Image.open(self.dir+'/'+self.img_list[idx])
    lr_img = self.in_transform(image)
    hr_img = self.ou_transform(image)
    return lr_img, hr_img

