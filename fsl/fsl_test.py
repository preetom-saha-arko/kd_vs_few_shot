# -*- coding: utf-8 -*-
"""FSL_MODIFIED.ipynb



Original file is located at
    https://colab.research.google.com/drive/1FauVmnosFoQDJamjUSpW5IUQTA6dGJ2H


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

I have been working on few-shot classification for a while now. The more I talk about it, the more the people around me seem to feel that it's some kind of dark magic. Even sadder: I noticed that very few actually used it on their projects. I think that's too bad, so I decided to make a tutorial so you'll have no excuse to deprive yourself of the power of few-shot learning methods.

In 15 minutes and just a few lines of code, we are going to implement
the [Prototypical Networks](https://arxiv.org/abs/1703.05175). It's the favorite method of
many few-shot learning researchers (~2000 citations in 3 years), because 1) it works well,
and 2) it's incredibly easy to grasp and to implement.

## Discovering Prototypical Networks
First, let's install the [tutorial GitHub repository](https://github.com/sicara/easy-few-shot-learning) and import some packages. If you're on Colab right now, you should also check that you're using a GPU (Edit > Notebook settings).
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install easyfsl

from easyfsl.datasets import WrapFewShotDataset
from medmnist import PathMNIST
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
import random
import pickle
from collections import Counter
import os
from PIL import Image

NUM_WORKERS = 1  # may need to change it

# !pip install medmnist

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--n_query', type=int, default=10)
parser.add_argument('--t_episodes', type=int, default=100000)
parser.add_argument('--model', type=str, default='resnet50')

args = parser.parse_args()

N_WAY = args.n_way  # Number of classes in a task
N_SHOT = args.n_shot  # Number of images per class in the support set
N_QUERY = args.n_query  # Number of images per class in the query set
N_TEST_EVALUATION_TASKS = args.t_episodes
log_update_frequency = 500
val_frequency = 500
IMAGE_TO_SAMPLE = 30  # 1: 25
LR = args.lr  # default 0.001


print('*'*50)
print(f'\n\nRunning with LR: {args.lr}\n\n')
print('*'*50)
print()
# make a directory to save the best model

image_size = 128   # change it to 224 for actual dataset

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_set = PathMNIST(split='test', transform=data_transform,
                     size=image_size, mmap_mode='r', download=True)

test_set.get_labels = lambda: [int(i) for i in test_set.labels]

def sort_dataset(partition):
    images = partition.imgs
    labels = [int(i) for i in partition.labels]
    obj = [(labels[i], images[i]) for i in range(len(images))]
    obj.sort(key=lambda a: a[0])
    images = [obj[i][1] for i in range(len(obj))]
    labels = [obj[i][0] for i in range(len(obj))]
    partition.imgs = images
    partition.labels = labels

sort_dataset(test_set)


def get_item(self, index):
    """
          return: (without transform/target_transofrm)
              img: PIL.Image
              target: int
    """
    img, target = self.imgs[index], self.labels[index]
    img = Image.fromarray(img)

    if self.as_rgb:
        img = img.convert("RGB")

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target


PathMNIST.__getitem__ = get_item

# BATCH_SIZE = N_WAY * (N_SHOT + N_QUERY)
# test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)


# len(train_set.imgs), len(train_set.labels)
# sanity check

assert len(test_set.imgs) == len(test_set.labels) 

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


if args.model == 'resnet50':
    convolutional_network = torchvision.models.resnet50(pretrained=True)
elif args.model == 'resnet18':
    convolutional_network = torchvision.models.resnet18(pretrained=True)
elif args.model == 'mobilenet_v3_small':
    convolutional_network = torchvision.models.mobilenet_v3_small(
        pretrained=True)

# convolutional_network = torchvision.models.mobilenet_v3_small(pretrained=True)
convolutional_network.fc = nn.Flatten()
# print(convolutional_network)

model = PrototypicalNetworks(convolutional_network).cuda()

# Load the model parameters
import os
if N_SHOT == 5:
    model_path = os.path.join(os.path.join(os.getcwd(), "models"), "best_proto_model_0.0008_30_10000_5_5.pth")
elif N_SHOT == 1:
    model_path = os.path.join(os.path.join(os.getcwd(), "models"), "best_proto_model_resnet18_0.0004_30_10000_5_1.pth")

model.load_state_dict(torch.load(model_path))

test_sampler = TaskSampler(
    test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TEST_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

criterion = nn.CrossEntropyLoss()


def evaluate_on_one_task(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """

    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    val_loss_per_task = criterion(classification_scores, query_labels.cuda())

    predicted_labels = torch.max(classification_scores.detach().data, 1)[1]

    # Convert tensors to numpy arrays for use with sklearn
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = query_labels.cpu().numpy()

    return predicted_labels, true_labels, val_loss_per_task.item()


def evaluate(data_loader: DataLoader):

    # We'll accumulate the predicted and true labels for all tasks
    all_predicted_labels = []
    all_true_labels = []
    total_loss = 0.0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            predicted_labels, true_labels, loss = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(true_labels)
            total_loss += loss

    # Calculate average loss
    average_loss = total_loss / len(data_loader)

    # Calculate F1 score and accuracy
    f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)

    with open('results.txt', 'w') as f:
        # print('Filename:', filename, file=f)  # Python 3.x
        print(f'\nF1 score: {f1}, Accuracy: {accuracy}, Average loss: {average_loss}\n', file=f)

    return f1, accuracy, average_loss


evaluate(test_loader)



# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)


# # Initialize best validation loss to a high value
# best_val_loss = float('inf')


# # Train the model yourself with this cell


# torch.cuda.empty_cache()
# model.train()



# # Or just load mine

# # !wget https://public-sicara.s3.eu-central-1.amazonaws.com/easy-fsl/resnet18_with_pretraining.tar
# # model.load_state_dict(torch.load("resnet18_with_pretraining.tar", map_location="cuda"))

# # Load the best model after training
# model.load_state_dict(torch.load(BEST_MODEL_PATH))

# print(val_loss)




"""Now let's see if our model got better!"""

# evaluate(test_loader) # 0.001_25_10k_5_5

# test_f1, test_acc, test_avg_loss = evaluate(test_loader)
# training_logs.write(f'Test F1: {test_f1}, Test Acc: {test_acc}, Test Avg Loss: {test_avg_loss}\n\n')
# training_logs.write('-'*50 + '\n')
