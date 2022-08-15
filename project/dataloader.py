import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode="train"):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))
        # use val dataset to train model because of small train dataset
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "val") + "/*.jpg")))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) # left image original
        img_B = img.crop((w / 2, 0, w, h)) # right image conditional

        # horizontal flip for data augmentation
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

transforms = transforms.Compose([
    # transforms.Resize((256, 256), Image.BICUBIC),
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    # Test Dataloader
    img_size = 256
    bs = 8
    train_data = ImageDataset("./data/facades", transforms=transforms, mode='train')
    test_data = ImageDataset("./data/facades", transforms=transforms, mode='test')

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)