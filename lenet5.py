#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os

#%%
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 1: Uma imagem por vez
        # 6: numero de filtros (pedaços) que serão usados para escanear a imagem
        # kernel_size: tamanho do filtro 5x5
        # stride: de quantos em quantos pixels nosso filtro irá se mover
        # padding: possibilita nosso filtro acessar as margens e neste caso impede que durante a multiplicação de raizes nossa imagem seja reduzida

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # transformando a matriz em um conjunto vetor linear 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#%%
# Visualizando os filtros iniciados randomicamente, precisamos ainda ensinar a rede a melhorar esses filtros

model = LeNet5(num_classes=10)

filters = model.conv1.weight.detach()

fig, axes = plt.subplots(1, 6, figsize=[15,5])
for i, ax in enumerate(axes):
    ax.imshow(filters[i,0], cmap='gray')
    ax.set_title(f'filter {i+1}')
    ax.axis('off')

plt.suptitle('Initialization State of LeNet5 Conv1 Filters')
plt.show()

#%%
# download dataset

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

dataiter = iter(train_loader)
images, labels = next(dataiter)

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f'Label: {labels[i].item()}')
    ax.axis('off')

plt.suptitle('Sample Images from MNIST Dataset')
plt.show()

#%%
# Get a single sample image from the dataset
sample_image, sample_label = train_dataset[2]
sample_image = sample_image.unsqueeze(0) # Add batch dimension [1, 1, 28, 28]

# Passage through the first layer ONLY (Conv1 + ReLU)
model.eval()
with torch.no_grad():
    # Move sample to the same device as the model (if applicable)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    sample_image = sample_image.to(device)
    
    feature_maps = F.relu(model.conv1(sample_image))

# Move back to CPU for plotting
feature_maps = feature_maps.cpu().squeeze(0) # [6, 28, 28]

print(f"Visualizing feature maps for digit: {sample_label}")

# Plotting
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
for i in range(6):
    axes[i].imshow(feature_maps[i], cmap='viridis')
    axes[i].set_title(f'Map {i+1}')
    axes[i].axis('off')

plt.suptitle(f'Feature Maps from Conv1 (Digit: {sample_label})', fontsize=16)
plt.show()

#%%
#checking if we have cuda or only cpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

learning_rate = 0.001
batch_size = 64 # how many images at the same time
num_epochs =  5 # 5x dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f'Training on {device}...')
model.train() # swiching the model to the train mode

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training finished!")

#%%
# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set model to evaluation mode
model.eval()

correct = 0
total = 0
misclassified_images = []
misclassified_labels = []
predicted_labels = []

# No gradient calculation needed during evaluation
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect misclassified examples
        misclassified_mask = (predicted != labels)
        if misclassified_mask.any():
            misclassified_images.append(images[misclassified_mask].cpu())
            misclassified_labels.append(labels[misclassified_mask].cpu())
            predicted_labels.append(predicted[misclassified_mask].cpu())

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10,000 test images: {accuracy:.2f}%')

#%%
# Concatenate all collected misclassified samples
misclassified_images = torch.cat(misclassified_images)
misclassified_labels = torch.cat(misclassified_labels)
predicted_labels = torch.cat(predicted_labels)

# Display first 12 misclassified images
num_to_show = min(12, len(misclassified_images))
fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for i in range(num_to_show):
    ax = axes[i // 6, i % 6]
    ax.imshow(misclassified_images[i].squeeze(), cmap='gray')
    ax.set_title(f'True: {misclassified_labels[i].item()}\nPred: {predicted_labels[i].item()}', color='red')
    ax.axis('off')

plt.suptitle('Misclassified Images (True Class vs Model Prediction)', fontsize=16)
plt.tight_layout()
plt.show()

#%%
# Create 'weights' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set save path
PATH = './models/lenet5_mnist.pth'

torch.save(model.state_dict(), PATH)
print(f"Model saved to {PATH}")

#%%
# 1. Create a new instance of the model
model = LeNet5(num_classes=10)

# 2. Load the state dictionary
model.load_state_dict(torch.load('./models/lenet5_mnist.pth'))

# 3. Set to evaluation mode if you're using it for inference
model.eval()

print("New model instance created and weights loaded successfully!")