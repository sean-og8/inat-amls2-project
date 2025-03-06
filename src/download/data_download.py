from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.INaturalist('data/train_mini', download=True, version="2021_train_mini")
validation_data = datasets.INaturalist('data/validation', download=True, version="2021_valid")

# define transformations: tensor and normalise
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize for uniformity
#     transforms.ToTensor()           # Convert PIL image to tensor
# ])

# # loaders for batching
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = DataLoader(train_data, batch_size=32, shuffle=True)
                                          