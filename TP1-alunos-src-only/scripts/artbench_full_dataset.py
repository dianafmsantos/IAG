import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ArtBenchFullDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item['image'])
        label = item['label']
        return image, label