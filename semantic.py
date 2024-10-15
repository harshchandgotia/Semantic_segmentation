import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sematic_dataloader import SegmentationDataset
from Model import SegNet
import torch.nn as nn
import torch.optim as optim

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image if needed
    transforms.ToTensor(),          # Convert image to PyTorch tensor
])

label_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize label if needed
    transforms.ToTensor(),          # Convert label to tensor
])

image_dir = 'data/png/train'
label_dir = 'data/png/train_labels'

train_data = SegmentationDataset(image_dir=image_dir, label_dir=label_dir, transform= image_transforms, target_transform= label_transforms)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

val_image_dir = 'data/png/val'
val_label_dir = 'data/png/val_labels'
test_image_dir = 'data/png/test'
test_label_dir = 'data/png/test_labels'

val_data = SegmentationDataset(image_dir=val_image_dir, label_dir=val_label_dir, transform= image_transforms, target_transform= label_transforms)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
test_data = SegmentationDataset(image_dir=test_image_dir, label_dir=test_label_dir, transform= label_transforms, target_transform= label_transforms)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegNet().to(device)

# for param in model.parameters():
#     print(param)
# for inputs, _ in train_dataloader:
# #model.eval()
#     with torch.no_grad():
#         outputs = model(inputs.to(device))
#         print(outputs[0].shape)
#         print(inputs[0].shape)
#         break
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# for inputs, masks in train_dataloader:
#     inputs, masks = inputs.to(device), masks.to(device)
    
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, masks)
    
#     loss.backward()
#     print(inputs.grad)  # Check if gradients are non-zero
#     break

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, masks in train_dataloader:
        inputs, masks = inputs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}")

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, masks in train_dataloader:
#         inputs, masks = inputs.to(device), masks.to(device)
        
#         optimizer.zero_grad()  # Zero the gradients
#         outputs = model(inputs)  # Forward pass
#         loss = criterion(outputs, masks)  # Compute the loss
        
#         # Debugging: Print the loss for the first batch
#         if epoch == 0 and running_loss == 0.0:
#             print(f"Initial loss: {loss.item()}")
        
#         loss.backward()  # Backward pass to compute gradients
        
#         # Gradient check: Print the gradients of the first parameter
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 print(f"Gradient for {name} at epoch {epoch+1}: {param.grad.abs().mean().item()}")
#             else:
#                 print(f"No gradient for {name} at epoch {epoch+1}")
#             break  # Print only for the first parameter for brevity
        
#         optimizer.step()  # Update model parameters
        
#         running_loss += loss.item()
    
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}")

#     # Additional debugging: Print sample input, output, and target
#     if epoch == 0:
#         print("Sample input:", inputs[0])
#         print("Sample output:", outputs[0])
        # print("Sample target:", masks[0])


# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, masks in train_dataloader:
#         inputs, masks = inputs.to(device), masks.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, masks)
        
#         # Debugging: Print the loss for the first batch
#         if epoch == 0 and running_loss == 0.0:
#             print(f"Initial loss: {loss.item()}")
        
#         loss.backward()
        
#         # Check gradients of model parameters
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 print(f"Gradients for {name}: {param.grad.norm()}")
#             else:
#                 print(f"Gradients for {name} are None")
        
#         # Check if parameters are being updated
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 before_update = param.clone().detach()
        
#         optimizer.step()
        
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 after_update = param.clone().detach()
#                 print(f"Update for {name}: {(after_update - before_update).norm()}")
        
#         running_loss += loss.item()
    
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}")
