# import torch
# import torchvision.models as models
# # from torchsummary import summary

# # Load pretrained ResNet50
# model = models.resnet50(pretrained=True)

# print("=== RESNET50 ARCHITECTURE ===")
# print(model)
# print("\n" + "="*60 + "\n")

# print("=== DETAILED LAYER BREAKDOWN ===")
# for name, layer in model.named_children():
#     print(f"\n--- {name.upper()} ---")
#     if name == 'conv1' or name == 'bn1' or name == 'relu' or name == 'maxpool':
#         # Basic layers
#         print(f"  {name}: {layer}")
#     elif name in ['layer1', 'layer2', 'layer3', 'layer4']:
#         # Residual blocks
#         print(f"  {name} (Residual blocks):")
#         for idx, (sub_name, sub_layer) in enumerate(layer.named_children()):
#             print(f"    [{idx:2d}] {sub_name}: {sub_layer}")
#     else:
#         # Final layers (avgpool, fc)
#         print(f"  {name}: {layer}")

# print("\n" + "="*60 + "\n")

# # # Print model summary (requires torchsummary)
# # print("=== MODEL SUMMARY (224x224 input) ===")
# # try:
# #     summary(model, (3, 224, 224))
# # except ImportError:
# #     print("Install torchsummary for detailed summary: pip install torchsummary")

# # Count parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"\nTotal parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")


# import torch
# import torchvision.models as models
# # from torchsummary import summary  # pip install torchsummary

# # Load pretrained VGG16
# model = models.vgg16(pretrained=True)

# print("=== VGG16 ARCHITECTURE ===")
# print(model)
# print("\n" + "="*50 + "\n")

# print("=== DETAILED LAYER BREAKDOWN ===")
# for name, layer in model.named_children():
#     print(f"\n--- {name.upper()} ---")
#     if name == 'features':
#         print("Feature extraction layers (Conv + Pooling):")
#         for idx, (sub_name, sub_layer) in enumerate(model.features.named_children()):
#             print(f"  [{idx:2d}] {sub_name}: {sub_layer}")
#     elif name == 'classifier':
#         print("Classifier layers (Fully Connected):")
#         for idx, (sub_name, sub_layer) in enumerate(model.classifier.named_children()):
#             print(f"  [{idx:2d}] {sub_name}: {sub_layer}")
#     else:
#         print(layer)

# print("\n" + "="*50 + "\n")

# # # Print model summary (requires torchsummary)
# # print("=== MODEL SUMMARY (224x224 input) ===")
# # try:
# #     summary(model, (3, 224, 224))
# # except ImportError:
# #     print("Install torchsummary for detailed summary: pip install torchsummary")

# # Count parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"\nTotal parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")

# import torch
# import torchvision.models as models
# import torch.nn as nn

# # Load pretrained VGG16
# model = models.vgg16(pretrained=True)

# # Replace classifier with Identity (acts as a "pass-through")
# model.avgpool = nn.Identity()
# model.classifier = nn.Identity()

# # Now the model outputs flattened features directly
# x = torch.randn(2, 3, 224, 224)  # Input: [B, C, H, W]
# features = model(x)              # Output: [B, 25088]

# print(f"Input shape: {x.shape}")
# print(f"Feature shape: {features.shape}")  # [2, 25088]

from models import DualModel
import torch

model = DualModel(num_classes=1)

# Test the model
test_input = torch.randn(2, 3, 224, 224)
test_output = model(test_input)
print(f"Model output shape: {test_output.shape}")