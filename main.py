import torch

from src.transform.read_in_and_process_data import data_preprocessing
from src.model.model_classes import EfficientNet_ContextualData
from src.model.training_loop import full_train_pipeline, test_model

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================================================================================================================
# Data preprocessing: retrieve data loaders
train_loader, val_loader, test_loader = data_preprocessing(train_image_dir="data/train_mini/2021_train_mini", train_json_file="data/train_mini/train_mini.json", val_image_dir="data/validation/2021_valid", val_json_file="data/validation/val.json")
# ======================================================================================================================
# Task: iNaturalist bird species identification
model =  EfficientNet_ContextualData(n_classes=1486, contextual_dim=3, tune_all_layers=True).to(device) # Build model object.
trained_model, train_metrics = full_train_pipeline(model, "effnet_final" , train_loader, val_loader, device)
test_loss, test_acc = test_model(trained_model, test_loader, device)   # Test model based on the test set.

# ======================================================================================================================
## Print out your results with following format:
train_acc = train_metrics["train_acc"][-1]
print('Model run complete final results: Train accuracy: {}%, Test accuracy: {}%;'.format(round(100 * train_acc, 2), round(100 * test_acc, 2)))

