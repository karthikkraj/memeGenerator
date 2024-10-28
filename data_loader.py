import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MemeDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.captions = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.captions.loc[idx, "Image_name"])
        image = Image.open(img_name).convert("RGB")
        caption = self.captions.loc[idx, "corrected_text"]
        
        if not isinstance(caption, str):
            caption = "No caption available"  

        image = self.transform(image)
        return image, caption

def get_data_loader(image_folder, csv_file, batch_size=32):
    dataset = MemeDataset(image_folder=image_folder, csv_file=csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)