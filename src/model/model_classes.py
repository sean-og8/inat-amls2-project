import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet_ContextualData(nn.Module):
    def __init__(self, n_classes, contextual_dim, tune_all_layers=True):
        super(EfficientNet_ContextualData, self).__init__()

        # Load EfficientNet-B0 with pretrained weights
        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.pretrained_model = efficientnet_b0(weights=self.weights)

        # Remove classifier by keeping only the feature extractor cnn layers
        self.feature_extractor = self.pretrained_model.features

        # Freeze or unfreeze the training of cnn layers
        if not tune_all_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # get pretrained model number of output features
        cnn_out_features = self.pretrained_model.classifier[1].in_features
        
        # Define final classifier layer combining CNN + Contextual Data
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_features + contextual_dim, 2048),  # Combine features
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, n_classes)  # Final output layer
        )

    def forward(self, image_data, contextual_data):

        # put image data through feature extactor 
        cnn_output = self.feature_extractor(image_data)
        cnn_output = torch.mean(cnn_output, dim=[2, 3])  # Global average pooling
        # flatten CNN output data
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  

        # combine CNN output with contextual data
        combined_input = torch.cat((cnn_output, contextual_data), dim=1)
        # Pass through the classifier for final prediction
        output = self.classifier(combined_input)
        return output