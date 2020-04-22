import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DeblurDataset(Dataset):
    def __init__(self, img_path, args, is_train=True):
        self.img_path = img_path
        self.args = args
        self.is_train = is_train
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_A = Image.open(self.img_path[index] + '_blur_err.png').convert('RGB')
        img_B = Image.open(self.img_path[index] + '_ref.png').convert('RGB')
        img_name = self.img_path[index][-6:]
        img_name.rstrip()

        #if self.is_train:
        #    '''
        #    Transforms:
        #    1. cut black edges
        #    2. random crop to (256, 256) patch
        #    3. random flip image from left to right
        #    4. PIL.Image(H,W,C) to Tensor(C,H,W)
        #    5. normalize from [0.0, 1.0] to [-1.0, 1.0]
        #    '''
#
        #    w = int(img_A.size[0]) - 40
        #    h = int(img_A.size[1]) - 40
#
        #    img_A = transforms.CenterCrop((h, w))(img_A)
        #    img_B = transforms.CenterCrop((h, w))(img_B)
#
        #    h1 = int(np.ceil(np.random.uniform(1e-2, h - self.args.fine_size)))
        #    w1 = int(np.ceil(np.random.uniform(1e-2, w - self.args.fine_size)))
#
        #    img_A = img_A.crop((w1, h1, w1 + self.args.fine_size, h1 + self.args.fine_size))
        #    img_B = img_B.crop((w1, h1, w1 + self.args.fine_size, h1 + self.args.fine_size))
#
        #    if np.random.random() < 0.5:
        #        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        #        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B, img_name

    def __len__(self):
        return len(self.img_path)

