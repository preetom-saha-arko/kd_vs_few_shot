# Modify the final fully connected layer according to the number of classes
import medmnist
from medmnist import INFO, Evaluator
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import utils, optim, device, inference_mode
import tqdm
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import medmnist
from medmnist import INFO, Evaluator
from medmnist import PathMNIST
from sklearn.metrics import f1_score

data_flag = 'pathmnist'
# data_flag = 'dermamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

image_size = 128

train_data = PathMNIST(root='pathmnist_data', split='train', transform=data_transform, size=image_size, mmap_mode='r' ,download=True)
val_data = PathMNIST(root='pathmnist_data', split='val', transform=data_transform, size=image_size, mmap_mode='r' ,download=True)
test_data = PathMNIST(root='pathmnist_data', split='test', transform=data_transform, size=image_size, mmap_mode='r' ,download=True)

# change data into dataloader form
BATCH_SIZE = 256
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        # Modify the final fully connected layer according to the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
    
model = Resnet50(num_classes=n_classes)   
    
def test_multiple_outputs(model, test_loader, device, validation = False):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            labels = labels.squeeze(1)
            all_true.extend(labels.cpu())
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs) # Disregard the first tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)
            all_pred.extend(predicted.cpu())
            # print("predicted.shape =", predicted.shape)
            # print("label.shape =", labels.shape)
            
            # print("total number of samples in this batch:", labels.size(0))
            # print("correctly classified samples in this batch:", (predicted == labels).sum().item())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print("correct = ", correct)
    # print("total = ", total)
    f1 = f1_score(all_true, all_pred, average='macro')
    
    accuracy = 100 * correct / total
    if validation:
        with open("validation_accuracy_f1_finetune.txt", "w") as f:
            print("f1:", f1, file=f)
            print(f"Validation Accuracy: {accuracy:.2f}%", file=f)
    else:
        with open("test_accuracy_f1_finetune.txt", "w") as f:
            print("f1:", f1, file=f)
            print(f"Test Accuracy: {accuracy:.2f}%", file=f)
    return accuracy, f1

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # print("outputs.shape", outputs.shape)
            # print("labels.shape", labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Perform validation
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader):
                val_labels = val_labels.type(torch.LongTensor).squeeze(1)
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            # torch.save(model, 'finetuned_model.pth')

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        
    torch.save(best_model, 'finetuned_model.pth')
    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list

# Train and test the lightweight network with cross entropy loss
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs, device)

import matplotlib.pyplot as plt

epochs = range(1, num_epochs+1)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_list)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('train_loss.png')
# plt.show()

# Plot validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_loss_list)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('val_loss.png')
# plt.show()

# Plot training accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy_list)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('train_accuracy.png')
# plt.show()

# Plot validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_accuracy_list)
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('val_accuracy.png')
# plt.show()

model = torch.load('finetuned_model.pth')

test_accuracy, test_f1 = test_multiple_outputs(model, test_dataloader, device)
