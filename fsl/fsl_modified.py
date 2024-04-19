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
parser.add_argument('--t_episodes', type=int, default=10000)
parser.add_argument('--model', type=str, default='resnet50')

args = parser.parse_args()

N_WAY = args.n_way  # Number of classes in a task
N_SHOT = args.n_shot  # Number of images per class in the support set
N_QUERY = args.n_query  # Number of images per class in the query set
N_TEST_EVALUATION_TASKS = 500
N_TRAINING_EPISODES = args.t_episodes
N_VALIDATION_TASKS = 500
log_update_frequency = 500
val_frequency = 500
IMAGE_TO_SAMPLE = 30  # 1: 25
LR = args.lr  # default 0.001


print('*'*50)
print(f'\n\nRunning with LR: {args.lr}\n\n')
print('*'*50)
print()
# make a directory to save the best model
os.makedirs(
    f'fsl_protorun/metric_folder/{args.model}_lr_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}', exist_ok=True)
os.makedirs(
    f'fsl_protorun/plots/{args.model}_lr_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}', exist_ok=True)

BEST_MODEL_PATH = f'fsl_protorun/best_models/best_proto_model_{args.model}_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}.pth'
PATH_TO_SAVE_PLOT = f'fsl_protorun/plots/{args.model}_lr_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}/'
PATH_TO_METRIC_DATA = f'fsl_protorun/metric_folder/{args.model}_lr_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}'
training_logs = open(
    f'LOGS_{args.model}_{LR}_{IMAGE_TO_SAMPLE}_{N_TRAINING_EPISODES}_{N_WAY}_{N_SHOT}.txt', 'a')

# set the logs for the current training parameters
training_logs.write('-'*50 + '\n')
training_logs.write(
    f'LR: {LR}, IMAGE_TO_SAMPLE: {IMAGE_TO_SAMPLE}, N_TRAINING_EPISODES: {N_TRAINING_EPISODES}, N_WAY: {N_WAY}, N_SHOT: {N_SHOT}\n')

image_size = 128   # change it to 224 for actual dataset

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_set = PathMNIST(split='train', transform=data_transform,
                      size=image_size, mmap_mode='r', download=True)
val_set = PathMNIST(split='val', transform=data_transform,
                    size=image_size, mmap_mode='r', download=True)
test_set = PathMNIST(split='test', transform=data_transform,
                     size=image_size, mmap_mode='r', download=True)

# img, label = train_set.__getitem__(0)
# print(type(img))
# print(label)
# print(type(label))

train_set.get_labels = lambda: [int(i) for i in train_set.labels]
val_set.get_labels = lambda: [int(i) for i in val_set.labels]
test_set.get_labels = lambda: [int(i) for i in test_set.labels]

# train_set.get_labels()

# train_set = WrapFewShotDataset(train_set)
# val_set = WrapFewShotDataset(val_set)
# test_set = WrapFewShotDataset(test_set)

# dir(train_set)


def sort_dataset(partition):
    images = partition.imgs
    labels = [int(i) for i in partition.labels]
    obj = [(labels[i], images[i]) for i in range(len(images))]
    obj.sort(key=lambda a: a[0])
    images = [obj[i][1] for i in range(len(obj))]
    labels = [obj[i][0] for i in range(len(obj))]
    partition.imgs = images
    partition.labels = labels


sort_dataset(train_set)
sort_dataset(val_set)
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

train_dicts = Counter(train_set.labels)
val_dicts = Counter(val_set.labels)
test_dicts = Counter(test_set.labels)


def random_sample_dataset(dataset, counter_dict, num_data=IMAGE_TO_SAMPLE):
    start = 0
    total_images, total_labels = [], []
    for key, value in counter_dict.items():
        end = start + value
        # print(start, end)
        imgs = dataset.imgs[start:end]
        labels = dataset.labels[start:end]

        # make them tuple then randomly sample from the list of tuples
        img_labels = [(img, label) for img, label in zip(imgs, labels)]
        random_img_labels = random.sample(img_labels, num_data)

        for sampled_image, sampled_label in random_img_labels:
            total_images.append(sampled_image)
            total_labels.append(sampled_label)

        start = end

    return total_images, total_labels


sampled_train_images, sampled_train_labels = random_sample_dataset(
    train_set, train_dicts)
sampled_val_images, sampled_val_labels = random_sample_dataset(
    val_set,  val_dicts)
sampled_test_images, sampled_test_labels = random_sample_dataset(
    test_set, test_dicts)

train_set.imgs = sampled_train_images
train_set.labels = sampled_train_labels

val_set.imgs = sampled_val_images
val_set.labels = sampled_val_labels

test_set.imgs = sampled_test_images
test_set.labels = sampled_test_labels

# len(train_set.imgs), len(train_set.labels)
# sanity check
assert len(train_set.imgs) == len(train_set.labels) == IMAGE_TO_SAMPLE * 9
assert len(val_set.imgs) == len(val_set.labels) == IMAGE_TO_SAMPLE * 9
assert len(test_set.imgs) == len(test_set.labels) == IMAGE_TO_SAMPLE * 9


def plot_metric(metric_list, title, ylabel, val_log_freq, save_path=None):
    plt.figure()
    xvalues = [(i+1) * val_log_freq for i in range(len(metric_list))]
    plt.plot(xvalues, metric_list)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.grid()
    # save the plot with the title name to the path if given but first append the file name with the title
    save_path = save_path + title + '.png' if save_path is not None else None
    if save_path is not None:
        plt.savefig(save_path)

# print(train_set.labels)
# print(val_set.labels)
# print(test_set.labels)

# train_set.get_labels()

# img, label = train_set.__getitem__(0)
# print(type(img))
# print(type(label))

# images = train_set.imgs
# labels = train_set.labels.squeeze(1).tolist()

# len(labels)

# training_obj = [(labels[i], images[i]) for i in range(len(images))]

# training_obj.sort(key=lambda a: a[0])

# len(training_obj)

# images = [training_obj[i][1] for i in range(len(training_obj))]
# labels = [training_obj[i][0] for i in range(len(training_obj))]

# train_set.imgs = images
# train_set.labels = labels

# dir(train_set)

# for i in range(len(train_set)):
#     image, label = train_set[i]
#     print(label)

# image_size = 28

# # NB: background=True selects the train set, background=False selects the test set
# # It's the nomenclature from the original paper, we just have to deal with it

# train_set = Omniglot(
#     root="./data",
#     background=True,
#     transform=transforms.Compose(
#         [
#             transforms.Grayscale(num_output_channels=3),
#             transforms.RandomResizedCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     ),
#     download=True,
# )
# test_set = Omniglot(
#     root="./data",
#     background=False,
#     transform=transforms.Compose(
#         [
#             # Omniglot images have 1 channel, but our model will expect 3-channel images
#             transforms.Grayscale(num_output_channels=3),
#             transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#         ]
#     ),
#     download=True,
# )


"""Let's take some time to grasp what few-shot classification is. Simply put, in a few-shot classification task, you have a labeled support set (which kind of acts
like a catalog) and query set. For each image of the query set, we want to predict a label from the
labels present in the support set. A few-shot classification model has to use the information from the
support set in order to classify query images. We say *few-shot* when the support set contains very
few images for each label (typically less than 10). The figure below shows a 3-way 2-shots classification task. "3-way" means "3 different classes" and "2-shots" means "2 examples per class".
We expect a model that has never seen any Saint-Bernard, Pug or Labrador during its training to successfully
predict the query labels. The support set is the only information that the model has regarding what a Saint-Bernard,
a Pug or a Labrador can be.

![few-shot classification task](https://images.ctfassets.net/be04ylp8y0qc/bZhboqYXfYeW4I88xmMNv/7c5efdc368206feaad045c674b1ced95/1_AteD0yXLkQ1BbjQTB3Ytwg.png?fm=webp)

Most few-shot classification methods are *metric-based*. It works in two phases : 1) they use a CNN to project both
support and query images into a feature space, and 2) they classify query images by comparing them to support images.
If, in the feature space, an image is closer to pugs than it is to labradors and Saint-Bernards, we will guess that
it's a pug.

From there, we have two challenges :

1. Find the good feature space. This is what convolutional networks are for. A CNN is basically a function that takes an image as input and outputs a representation (or *embedding*) of this image in a given feature space. The challenge here is to have a CNN that will
project images of the same class into representations that are close to each other, even if it has not been trained
on objects of this class.
2. Find a good way to compare the representations in the feature space. This is the job of Prototypical Networks.


![Prototypical classification](https://images.ctfassets.net/be04ylp8y0qc/45M9UcUp6KnzwDaBHeGZb7/bb2dcda5942ee7320600125ac2310af6/0_M0GSRZri859fGo48.png?fm=webp)

From the support set, Prototypical Networks compute a prototype for each class, which is the mean of all embeddings
of support images from this class. Then, each query is simply classified as the nearest prototype in the feature space,
with respect to euclidean distance.

If you want to learn more about how this works, I explain it
[there](https://www.sicara.fr/blog-technique/few-shot-image-classification-meta-learning).
But now, let's get to coding.
In the code below (modified from [this](https://github.com/sicara/easy-few-shot-learning/blob/master/easyfsl/methods/prototypical_networks.py)), we simply define Prototypical Networks as a torch module, with a `forward()` method.
You may notice 2 things.

1. We initiate `PrototypicalNetworks` with a *backbone*. This is the feature extractor we were talking about.
Here, we use as backbone a ResNet18 pretrained on ImageNet, with its head chopped off and replaced by a `Flatten`
layer. The output of the backbone, for an input image, will be a 512-dimensional feature vector.
2. The forward method doesn't only take one input tensor, but 3: in order to predict the labels of query images,
we also need support images and labels as inputs of the model.
"""


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
elif args.model == 'resnet34':
    convolutional_network = torchvision.models.resnet34(pretrained=True)
elif args.model == 'mobilenet_v3_small':
    convolutional_network = torchvision.models.mobilenet_v3_small(
        pretrained=True)

# convolutional_network = torchvision.models.mobilenet_v3_small(pretrained=True)
convolutional_network.fc = nn.Flatten()
# print(convolutional_network)

model = PrototypicalNetworks(convolutional_network).cuda()

"""Now we have a model! Note that we used a pretrained feature extractor,
so our model should already be up and running. Let's see that.

Here we create a dataloader that will feed few-shot classification tasks to our model.
But a regular PyTorch dataloader will feed batches of images, with no consideration for
their label or whether they are support or query. We need 2 specific features in our case.

1. We need images evenly distributed between a given number of classes.
2. We need them split between support and query sets.

For the first point, I wrote a custom sampler: it first samples `n_way` classes from the dataset,
then it samples `n_shot + n_query` images for each class (for a total of `n_way * (n_shot + n_query)`
images in each batch).
For the second point, I have a custom collate function to replace the built-in PyTorch `collate_fn`.
This baby feed each batch as the combination of 5 items:

1. support images
2. support labels between 0 and `n_way`
3. query images
4. query labels between 0 and `n_way`
5. a mapping of each label in `range(n_way)` to its true class id in the dataset
(it's not used by the model but it's very useful for us to know what the true class is)

You can see that in PyTorch, a DataLoader is basically the combination of a sampler, a dataset and a collate function
(and some multiprocessing voodoo): sampler says which items to fetch, the dataset says how to fetch them, and
the collate function says how to present these items together. If you want to dive into these custom objects,
they're [here](https://github.com/sicara/easy-few-shot-learning/tree/master/easyfsl/data_tools).
"""

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
# test_set.get_labels = lambda: test_set.labels.squeeze().tolist()
# test_set.get_labels = lambda: [int(i) for i in test_set.labels]
# print(test_set.labels.squeeze(1).tolist())
# labels = [int(i) for i in test_set.labels]
# print(labels)

# test_set.get_labels = lambda: labels
# test_set.get_labels = lambda: [
#     instance[1] for instance in test_set._flat_character_images
# ]
# print(test_set.get_labels)
# print(test_set.get_labels())

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

# test_set.get_labels()

"""We created a dataloader that will feed us with 5-way 5-shot tasks (the most common setting in the litterature).
Now, as every data scientist should do before launching opaque training scripts,
let's take a look at our dataset.
"""

# type(test_set.get_labels())

# (
#     example_support_images,
#     example_support_labels,
#     example_query_images,
#     example_query_labels,
#     example_class_ids,
# ) = next(iter(test_loader))

# plot_images(example_support_images, "support images", images_per_row=N_SHOT)
# plot_images(example_query_images, "query images", images_per_row=N_QUERY)

"""For both support and query set, you should have one line for each class.

How does our model perform on this task?
"""

# model.eval()
# example_scores = model(
#     example_support_images.cuda(),
#     example_support_labels.cuda(),
#     example_query_images.cuda(),
# ).detach()

# _, example_predicted_labels = torch.max(example_scores.data, 1)

# print("Ground Truth / Predicted")
# for i in range(len(example_query_labels)):
#     print(
#         f"{test_set._characters[example_class_ids[example_query_labels[i]]]} / {test_set._characters[example_class_ids[example_predicted_labels[i]]]}"
#     )

"""This doesn't look bad: keep in mind that the model was trained on very different images, and has only seen 5 examples for each class!

Now that we have a first idea, let's see more precisely how good our model is.
"""


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

    # print(f'\nF1 score: {f1}, Accuracy: {accuracy}, Average loss: {average_loss}\n')

    return f1, accuracy, average_loss


# evaluate(test_loader)

"""With absolutely zero training on Omniglot images, and only 5 examples per class, we achieve around 86% accuracy! Isn't this a great start?

Now that you know how to make Prototypical Networks work, you can see what happens if you tweak it
a little bit (change the backbone, use other distances than euclidean...) or if you change the problem
(more classes in each task, less or more examples in the support set, maybe even one example only,
but keep in mind that in that case Prototypical Networks are just standard nearest neighbour).

When you're done, you can scroll further down and learn how to **meta-train this model**, to get even better results.

## Training a meta-learning algorithm

Let's use the "background" images of Omniglot as training set. Here we prepare a data loader of 40 000 few-shot classification
tasks on which we will train our model. The alphabets used in the training set are entirely separated from those used in the testing set.
This guarantees that at test time, the model will have to classify characters that were not seen during training.

Note that we don't set a validation set here to keep this notebook concise,
but keep in mind that **this is not good practice** and you should always use validation when training a model for production.
"""

# train_set.get_labels = lambda: [instance[1] for instance in train_set._flat_character_images]
# train_set.get_labels = lambda: train_set.labels
# train_set.get_labels = lambda: [int(i) for i in train_set.labels]
train_sampler = TaskSampler(
    train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_sampler = TaskSampler(
    val_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_VALIDATION_TASKS
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

"""We will keep the same model. So our weights will be pre-trained on ImageNet. If you want to start a training from scratch,
feel free to set `pretrained=False` in the definition of the ResNet.

Here we define our loss and our optimizer (cross entropy and Adam, pretty standard), and a `fit` method.
This method takes a classification task as input (support set and query set). It predicts the labels of the query set
based on the information from the support set; then it compares the predicted labels to ground truth query labels,
and this gives us a loss value. Then it uses this loss to update the parameters of the model. This is a *meta-training loop*.
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()

    predicted_labels = torch.max(classification_scores.detach().data, 1)[1]
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = query_labels.cpu().numpy()

    f1 = f1_score(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return loss.item(), f1, accuracy


"""To train the model, we are just going to iterate over a large number of randomly generated few-shot classification tasks,
and let the `fit` method update our model after each task. This is called **episodic training**.

This took me 20mn on an RTX 2080 and I promised you that this whole tutorial would take 15mn.
So if you don't want to run the training yourself, you can just skip the training and load the model that I trained
using the exact same code.
"""

# Initialize best validation loss to a high value
best_val_loss = float('inf')


# Train the model yourself with this cell

train_loss = []  # train loss averaged over val_frequency episodes
train_acc = []
train_f1 = []

val_loss = []
val_acc = []
val_f1 = []

temp_train_loss = []
temp_train_acc = []
temp_train_f1 = []

point_train_loss = []  # train loss at the val_frequency episodes
point_train_acc = []
point_train_f1 = []

torch.cuda.empty_cache()
model.train()

with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        # model.train()
        train_loss_value, train_f1_value, train_acc_value = fit(
            support_images, support_labels, query_images, query_labels)
        temp_train_loss.append(train_loss_value)
        temp_train_f1.append(train_f1_value)
        temp_train_acc.append(train_acc_value)

        # evaluate(val_loader)
        if (episode_index + 1) % val_frequency == 0:
            val_f1_value, val_acc_value, val_avg_loss_value = evaluate(
                val_loader)

            val_f1.append(val_f1_value)
            val_acc.append(val_acc_value)
            val_loss.append(val_avg_loss_value)

            train_loss.append(np.mean(temp_train_loss))
            train_f1.append(np.mean(temp_train_f1))
            train_acc.append(np.mean(temp_train_acc))

            point_train_loss.append(train_loss_value)
            point_train_f1.append(train_f1_value)
            point_train_acc.append(train_acc_value)

            training_logs.write(
                f'Episode: {episode_index+1}, Training Loss: {train_loss_value}, Train F1: {train_f1_value}, Train Acc: {train_acc_value}\n')
            training_logs.write(
                f'Episode: {episode_index+1}, Avg Train Loss: {train_loss[-1]}, Avg Train F1: {train_f1[-1]}, Avg Train Acc: {train_acc[-1]}\n')
            training_logs.write(
                f'Episode: {episode_index+1}, Avg Val Loss: {val_avg_loss_value}, Val F1: {val_f1_value}, Val Acc: {val_acc_value}\n')

            print(
                f'Episode: {episode_index+1}, Training Loss: {train_loss_value}, Train F1: {train_f1_value}, Train Acc: {train_acc_value}')
            print(
                f'Episode: {episode_index+1}, Avg Train Loss: {train_loss[-1]}, Avg Train F1: {train_f1[-1]}, Avg Train Acc: {train_acc[-1]}')
            print(
                f'Episode: {episode_index+1}, Avg Val Loss: {val_avg_loss_value}, Val F1: {val_f1_value}, Val Acc: {val_acc_value}')

            temp_train_loss = []
            temp_train_f1 = []
            temp_train_acc = []

            # Save the model if it has the best validation loss so far
            if val_avg_loss_value < best_val_loss:
                print(
                    f'Saving Best Model with Avg Validation Loss: {val_avg_loss_value}!!!')
                training_logs.write(
                    f'Saving Best Model with Avg Validation Loss: {val_avg_loss_value}!!!\n')
                best_val_loss = val_avg_loss_value
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            model.train()

        # if (episode_index+1) % log_update_frequency == 0:
        #     tqdm_train.set_postfix(loss=sliding_average(train_loss, log_update_frequency))

# Or just load mine

# !wget https://public-sicara.s3.eu-central-1.amazonaws.com/easy-fsl/resnet18_with_pretraining.tar
# model.load_state_dict(torch.load("resnet18_with_pretraining.tar", map_location="cuda"))

# Load the best model after training
model.load_state_dict(torch.load(BEST_MODEL_PATH))

# print(val_loss)

# After training
plot_metric(train_loss, 'Avg Training Loss', 'Loss',
            val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(train_f1, 'Avg Training F1 Score',
            'F1 Score', val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(train_acc, 'Avg Training Accuracy',
            'Accuracy', val_frequency, PATH_TO_SAVE_PLOT)

plot_metric(point_train_loss, 'Training Loss',
            'Loss', val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(point_train_f1, 'Training F1 Score',
            'F1 Score', val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(point_train_acc, 'Training Accuracy',
            'Accuracy', val_frequency, PATH_TO_SAVE_PLOT)

plot_metric(val_loss, 'Validation Loss', 'Loss',
            val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(val_f1, 'Validation F1 Score', 'F1 Score',
            val_frequency, PATH_TO_SAVE_PLOT)
plot_metric(val_acc, 'Validation Accuracy', 'Accuracy',
            val_frequency, PATH_TO_SAVE_PLOT)


with open(os.path.join(PATH_TO_METRIC_DATA, 'val_loss.pkl'), 'wb') as f:
    pickle.dump(val_loss, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'val_f1.pkl'), 'wb') as f:
    pickle.dump(val_f1, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'val_acc.pkl'), 'wb') as f:
    pickle.dump(val_acc, f)

with open(os.path.join(PATH_TO_METRIC_DATA, 'train_loss.pkl'), 'wb') as f:
    pickle.dump(train_loss, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'train_f1.pkl'), 'wb') as f:
    pickle.dump(train_f1, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'train_acc.pkl'), 'wb') as f:
    pickle.dump(train_acc, f)

with open(os.path.join(PATH_TO_METRIC_DATA, 'point_train_loss.pkl'), 'wb') as f:
    pickle.dump(point_train_loss, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'point_train_f1.pkl'), 'wb') as f:
    pickle.dump(point_train_f1, f)
with open(os.path.join(PATH_TO_METRIC_DATA, 'point_train_acc.pkl'), 'wb') as f:
    pickle.dump(point_train_acc, f)


"""Now let's see if our model got better!"""

# evaluate(test_loader) # 0.001_25_10k_5_5

# test_f1, test_acc, test_avg_loss = evaluate(test_loader)
# training_logs.write(f'Test F1: {test_f1}, Test Acc: {test_acc}, Test Avg Loss: {test_avg_loss}\n\n')
# training_logs.write('-'*50 + '\n')
