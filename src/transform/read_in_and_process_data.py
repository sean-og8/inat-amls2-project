from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.model.model_classes import ImageDatasetWithContext


def data_preprocessing(train_image_dir, train_json_file, val_image_dir, val_json_file):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_data = ImageDatasetWithContext(image_dir=train_image_dir, json_file=train_json_file, transform=train_transform)
    validation_data = ImageDatasetWithContext(image_dir=val_image_dir, json_file=val_json_file, transform=val_transform)

    bird_indices = []
    with open('indices/bird_indices.txt', 'r') as f:
        for line in f:
            bird_indices.append(int(line[0:-1]))

    bird_test_indices = []
    with open('indices/bird_val_indices.txt', 'r') as f:
        for line in f:
            bird_test_indices.append(int(line[0:-1]))

    # Find all indices where the target (label) corresponds to birds
    # Create a subset dataset containing only birds
    train_data_birds = Subset(train_data, bird_indices)

    label_index_map = {}
    index_list = [] 
    label_list = []

    for i, (_, _, label) in enumerate(train_data_birds):
        index_list.append(i)
        label_list.append(label)
    # now, for each label, randomly select 20% of indices, these will become the val set. 
    index_list_train, index_list_val, labels_train, labels_val = train_test_split(
    index_list, label_list, test_size=0.2, random_state=42, stratify=label_list)

    val_data_birds = Subset(train_data_birds, index_list_val)
    train_data_birds = Subset(train_data_birds, index_list_train)

    # Create a DataLoader for the filtered dataset
    train_loader_birds = DataLoader(train_data_birds, batch_size=32, shuffle=True, num_workers=0)
    val_loader_birds = DataLoader(val_data_birds, batch_size=32, shuffle=True, num_workers=0)

    # use validation set from problem as the unseen test set
    test_data_birds = Subset(validation_data, bird_test_indices)
    test_loader_birds = DataLoader(test_data_birds, batch_size=32, shuffle=True, num_workers=0)

    print(f"Filtered train dataset contains {len(train_data_birds)} bird images.")
    print(f"Filtered val dataset contains {len(val_data_birds)} bird images.")
    print(f"Filtered test dataset contains {len(test_data_birds)} bird images.")
    return train_loader_birds, val_loader_birds, test_loader_birds