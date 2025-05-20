# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# class RISCDataset(Dataset):
#     def __init__(self, csv_path, images_dir, split="train", image_size=224):
#         self.data = pd.read_csv(csv_path)
#         self.data = self.data[self.data["split"] == split].reset_index(drop=True)
#         self.images_dir = images_dir
#         self.image_size = image_size
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image_path = os.path.join(self.images_dir, row["image"])
#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)
#         # Bazı captionlar NaN olabiliyor, onları filtrele!
#         captions = [row[f"caption_{i}"] for i in range(1, 6) if pd.notnull(row[f"caption_{i}"])]
#         return {
#             "image": image,
#             "captions": captions,
#             "image_id": row["image"],
#             "source": row["source"]
#         }

# # Example usage:
# # dataset = RISCDataset(
# #     csv_path="RISCM/captions.csv",
# #     images_dir="RISCM/resized",
# #     split="train"
# # )

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RISCDataset(Dataset):
    def __init__(self, csv_path, images_dir, split="train", image_size=224):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        captions = [row[f"caption_{i}"] for i in range(1, 6) if pd.notnull(row[f"caption_{i}"])]
        return {
            "image": image,
            "captions": captions,
            "image_id": row["image"],
            "source": row["source"]
        }
