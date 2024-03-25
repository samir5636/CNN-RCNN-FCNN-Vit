# Deep Learning Project README


### CNN Classifier
#### 1. Establish a CNN architecture based on the PyTorch library to classify the MINST dataset. Define layers (Convolution, pooling, fully connect layer),

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### 2. Do the same thing with Faster R-CNN.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the architecture for the modified Faster R-CNN
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten before FC
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### 3. Compare the two models (By using several metrics (Accuracy, F1 score, Loss, Training time))

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Function to calculate F1 score
def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

# Training function
def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()  # Start time for epoch
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()  # End time for epoch
    return running_loss / len(train_loader), end_time - start_time  # Return loss and training time

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return accuracy_score(y_true, y_pred), calculate_f1_score(y_true, y_pred)

# Inside your evaluation function, move input tensors to the same device
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Initialize models
cnn_model = CNN().to(device)
faster_rcnn_model = FasterRCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
faster_rcnn_optimizer = optim.Adam(faster_rcnn_model.parameters(), lr=learning_rate)

# Training and evaluation
for epoch in range(num_epochs):
    cnn_loss, cnn_time = train_model(cnn_model, criterion, cnn_optimizer, train_loader, device)
    faster_rcnn_loss, faster_rcnn_time = train_model(faster_rcnn_model, criterion, faster_rcnn_optimizer, train_loader, device)
    cnn_accuracy, cnn_f1_score = test_model(cnn_model, test_loader, device)
    faster_rcnn_accuracy, faster_rcnn_f1_score = test_model(faster_rcnn_model, test_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}], CNN Loss: {cnn_loss:.4f}, Faster R-CNN Loss: {faster_rcnn_loss:.4f}")
    print(f"Training Time - CNN: {cnn_time:.2f} seconds, Faster R-CNN: {faster_rcnn_time:.2f} seconds")
    print(f"CNN Accuracy: {cnn_accuracy*100:.2f}%, CNN F1 Score: {cnn_f1_score:.4f}")
    print(f"Faster R-CNN Accuracy: {faster_rcnn_accuracy*100:.2f}%, Faster R-CNN F1 Score: {faster_rcnn_f1_score:.4f}")

# Ensure model and input tensors are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-----------")
print("CNN Accuracy:", evaluate(cnn_model.to(device),test_loader,device))
print("Faster RCNN Accuracy:", evaluate(faster_rcnn_model.to(device),test_loader,device))
```

##### Model Comparison

| Metric       | CNN           | Faster R-CNN  |
|--------------|---------------|---------------|
| Accuracy     | 0.9911        | 0.9909        |
| F1 Score     | 0.9887        | 0.9902        |
| Loss         | 0.0145        | 0.0103        |
| Training Time| 15.08s        | 15.12s        |


#### 4. By using retrained models (VGG16 and AlexNet) fine tune your model to the new dataSet,then compare the obtained results to CNN and Faster R-CNN, what is your conclusion.

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models
vgg16 = torchvision.models.vgg16(pretrained=True)
alexnet = torchvision.models.alexnet(pretrained=True)

# Modify classifier layers
num_classes = 10  # MNIST has 10 classes
vgg16.classifier[6] = nn.Linear(4096, num_classes)
alexnet.classifier[6] = nn.Linear(4096, num_classes)

# Convert grayscale images to RGB format
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit VGG16 and AlexNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
vgg16_optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
alexnet_optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

# Move models to device
alexnet = alexnet.to(device)
vgg16 = vgg16.to(device)

# Training and evaluation
num_epochs = 5
for epoch in range(num_epochs):
    alexnet_loss, alexnet_time = train_model(alexnet, criterion, alexnet_optimizer, train_loader, device)
    vgg16_loss, vgg16_time = train_model(vgg16, criterion, vgg16_optimizer, train_loader, device)
    alexnet_accuracy, alexnet_f1_score = test_model(alexnet, test_loader, device)
    vgg16_accuracy, vgg16_f1_score = test_model(vgg16, test_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}], AlexNet Loss: {alexnet_loss:.4f}, VGG16 Loss: {vgg16_loss:.4f}")
    print(f"Training Time - AlexNet: {alexnet_time:.2f} seconds, VGG16: {vgg16_time:.2f} seconds")
    print(f"AlexNet Accuracy: {alexnet_accuracy*100:.2f}%, AlexNet F1 Score: {alexnet_f1_score:.4f}")
    print(f"VGG16 Accuracy: {vgg16_accuracy*100:.2f}%, VGG16 F1 Score: {vgg16_f1_score:.4f}")

print("-----------")
print("VGG16 Accuracy:", evaluate(vgg16, test_loader))
print("AlexNet Accuracy:", evaluate(alexnet, test_loader))
```

### Vision Transformer (VIT):
Vision Transformers (ViT), since their introduction by Dosovitskiy et. al. [reference](https://arxiv.org/abs/2010.11929) in 2020, have dominated the field of Computer Vision, obtaining state-of-the-art performance in image classification first, and later on in other tasks as well.

#### 1. By following this tutorial : [article]( https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c), establish a Vit model architecture from scratch, then do classification task on MINST Dataset.

##### Import library :
```python
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)
```

#### Patchifying and the linear mapping :
```python
def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
```

```python
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
```

```python
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
```

```python
class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out)
```

#### Positional encoding : 

```python
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
```
#### main function :
```python
def main():    
    transform = ToTensor()
    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    N_EPOCHS = 7
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
if __name__ == "__main__":
    main() 
```





| Metric       | CNN           | Faster R-CNN  |
|--------------|---------------|---------------|
| Accuracy     | 0.9911        | 0.9909        |
| F1 Score     | 0.9887        | 0.9902        |
| Loss         | 0.0145        | 0.0103        |
| Training Time| 15.08s        | 15.12s        |

Next, we fine-tuned pre-trained models (VGG16 and AlexNet) on the MNIST dataset and compared their performance with CNN and Faster R-CNN. The results showed that the CNN model achieved the highest accuracy among all models, closely followed by Faster R-CNN.

Lastly, we explored Vision Transformers (ViT) by implementing a ViT model architecture from scratch and performing image classification on the MNIST dataset. Although the ViT model achieved decent performance, it did not outperform the CNN model in this particular task.

Overall, the CNN model demonstrated strong performance for image classification tasks on the MNIST dataset, making it a suitable choice for such applications. However, further experimentation and evaluation on different datasets and tasks are necessary to determine the best model for specific use cases.


