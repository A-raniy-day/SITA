import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DataFromDir(Dataset):
    def __init__(self, path, transform):
        self.image_list = os.listdir(path)
        self.label_list = os.listdir(path)
        self.dir = path
        self.trans = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.trans(Image.open(os.path.join(self.dir, self.image_list[idx])))
        label = self.label_list[idx]
        return img, label  


def set_loader(opt):
    trans = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.ToTensor(),
    ])

    dataset = DataFromDir(opt.data_folder, transform=trans)
    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=(sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=sampler)

    return dataloader
