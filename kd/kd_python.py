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
import numpy as np
import random
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

random_seed = 0

np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import medmnist
from medmnist import INFO, Evaluator
from medmnist import PathMNIST

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

class TeacherResnet50(torch.nn.Module):
    def __init__(self):
        super(TeacherResnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = torch.nn.Identity()  # Remove the last layer

    def forward(self, x):
        return self.resnet50(x)
    
teacher_model = TeacherResnet50()
    
class StudentMobileNetV3(torch.nn.Module):
    def __init__(self, num_classes):
        super(StudentMobileNetV3, self).__init__()
        self.mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=False)
        self.mobilenet_v3.classifier[3] = torch.nn.Identity()  # Remove the last layer
        self.linear = torch.nn.Linear(1024, 2048)
        self.fc = torch.nn.Linear(1024, num_classes)  # Add a new fully connected layer for classification

    def forward(self, x):
        x = self.mobilenet_v3(x)
        return self.linear(x), self.fc(x)
    
student_model = StudentMobileNetV3(num_classes=n_classes)
    
from tqdm import tqdm    

def test_multiple_outputs(student_model, test_loader, device, validation = False):
    student_model.to(device)
    student_model.eval()
    
    correct = 0
    total = 0

    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            labels = labels.squeeze(1)
            all_true.extend(labels.cpu())
            inputs, labels = inputs.to(device), labels.to(device)
            
            _, outputs = student_model(inputs) # Disregard the first tensor of the tuple
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
        with open("validation_accuracy_f1.txt", "w") as f:
            print("Validation f1:", f1, file=f)
            print(f"Validation Accuracy: {accuracy:.2f}%", file=f)
    else:
        with open("test_accuracy_f1.txt", "w") as f:
            print("Test f1:", f1, file=f)
            print(f"Test Accuracy: {accuracy:.2f}%", file=f)
    return accuracy, f1

def train_cosine_loss(teacher, student, train_loader, val_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    
    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                teacher_hidden_representation = teacher(inputs)
                
            # Forward pass with the student model
            student_hidden_representation, student_logits = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            # print("student logits shape:", student_logits.shape)
            # print("labels shape:", labels.shape)
            labels = labels.squeeze(1)
            # print("labels shape:", labels.shape)
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        
        # # perform validation
        # accuracy = test_multiple_outputs(student, val_loader, device, validation=True)
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # perform validation
        student.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader):
                val_labels = val_labels.type(torch.LongTensor).squeeze(1)
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                
                
                teacher_hidden_representation = teacher(val_inputs)
                val_student_hidden_representation, val_student_logits = student(val_inputs)
                
                val_hidden_rep_loss = cosine_loss(val_student_hidden_representation, teacher_hidden_representation, target=torch.ones(val_inputs.size(0)).to(device))
                
                val_student_logits = val_student_logits.to(device)
                # print("logits device:", val_student_logits.get_device())
                # print("labels device:", val_labels.get_device())
                val_label_loss = ce_loss(val_student_logits, val_labels)
                val_loss = hidden_rep_loss_weight * val_hidden_rep_loss + ce_loss_weight * val_label_loss
                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_student_logits.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = student

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)

        student.train()
        
    torch.save(best_model, 'student_model.pth')
    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list

# Train and test the lightweight network with cross entropy loss
num_epochs = 50

train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = train_cosine_loss(teacher=teacher_model, student=student_model, train_loader=train_dataloader, val_loader=val_dataloader, epochs=num_epochs, learning_rate=0.0005, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)

with open("train_loss.txt", "w") as fout:
    print(*train_loss_list, sep="\n", file=fout)
    
with open("val_loss.txt", "w") as fout:
    print(*val_loss_list, sep="\n", file=fout)
    
with open("train_accuracy.txt", "w") as fout:
    print(*train_accuracy_list, sep="\n", file=fout)
    
with open("val_accuracy.txt", "w") as fout:
    print(*val_accuracy_list, sep="\n", file=fout)

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

student_model = torch.load('student_model.pth')

test_accuracy, f1 = test_multiple_outputs(student_model, test_dataloader, device)
