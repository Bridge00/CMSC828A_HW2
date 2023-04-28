!pip install transformers evaluate datasets

from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetConfig
import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score

"""# Training the Task Ensemble"""

# Download & process dataset
dataset = load_dataset("Maysee/tiny-imagenet")
processor = AutoImageProcessor.from_pretrained("/content/drive/MyDrive/ensembling/preprocessor_config.json")

def process_example(example):
    if example['image'].mode != 'RGB':
        example['image'] = example['image'].convert('RGB')
    example = processor(example['image'], return_tensors='pt')
    example['pixel_values'] = example['pixel_values'].squeeze()
    return example

# Subset the train split : Computational constraints
dataset['train'] = dataset['train'][0:60000]

#train and test data format
dataset['train'] = dataset['train'].map(process_example)
dataset['train'].set_format("pt", columns=['pixel_values'], output_all_columns=True)
dataset['valid'] = dataset['valid'].map(process_example)
dataset['valid'].set_format("pt", columns=['pixel_values'], output_all_columns=True)

# tasks = {
#     1: [182, 61, 120, 193, 12, 23, 146, 165, 142, 171, 9, 45, 50, 192, 123, 156, 31, 89, 100, 65, 5, 75,
#         157, 158, 139, 154, 35, 67, 58, 105, 29, 17, 150, 122, 15, 62, 167, 174, 60, 110, 133, 145, 199],
#     2: [114, 169, 40, 18, 19, 49, 187, 83, 16, 34, 47, 59, 166, 68, 32, 197, 52, 3, 51, 190, 66, 94, 170,
#         196, 116, 138, 184, 181, 137, 128, 14, 55, 140, 76, 135, 121, 88, 124, 85, 130, 43, 162, 24, 180,
#         20, 63, 155, 107, 96, 134, 175, 69, 82, 109, 56, 115, 136, 70, 80, 41],
#     3: [10, 92, 103, 86, 189, 64, 179, 147, 13, 53, 198, 1, 72, 48, 77, 36, 42, 73],
#     4: [39, 195, 126, 191, 99, 144, 160, 104, 159, 21, 161, 6, 176, 113, 168, 102, 194, 148, 30, 119,
#         87, 27, 106, 2, 143, 74, 79, 132, 178, 101, 28, 186, 97, 111, 91, 117, 127, 22, 71, 118, 44, 177,
#         153, 172],
#     5: [81, 151, 0, 37, 33, 11, 141, 112, 183, 149, 7, 173, 125, 108, 185, 25, 129, 163, 84, 54, 26, 152,
#         78, 38, 188, 4, 95, 98, 57, 131, 90, 46, 8, 164, 93]
# }

# train baseline models
# config = ResNetConfig(num_labels=200, num_channels=3)
# metric = evaluate.load("accuracy")
# models = {}

# for t in tasks.keys():
#     models[t] = ResNetForImageClassification(config)
#     training_args = TrainingArguments(output_dir=f"./task_{t}", evaluation_strategy="epoch", num_train_epochs=50,)
#     train_data = train_datasets[t].map(process_example)
#     train_data.set_format("pt", columns=['pixel_values'], output_all_columns=True)
#     trainer = Trainer(
#         model=models[t],
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=dataset['valid'],
#     )
#     trainer.train()

# evaluate baseline models on combined validation dataset
# for t in tasks.keys():
#     predictions = []
#     with torch.no_grad():
#         for example in dataset['valid']:
#             predicted_label = models[t](example['pixel_values'].unsqueeze(0).cuda()).logits.argmax(-1).item()
#             predictions.append(predicted_label)
#     acc = accuracy_score(dataset['valid']['label'], predictions)
#     print('Task {} model\tAccuracy: {:.2f}%'.format(t,acc))

# evaluate baseline models on tasks only validation dataset
# for t in tasks.keys():
#     predictions = []
#     with torch.no_grad():
#         for example in val_datasets[t]:
#             predicted_label = models[t](example['pixel_values'].unsqueeze(0).cuda()).logits.argmax(-1).item()
#             predictions.append(predicted_label)
#     acc = accuracy_score(val_datasets[t]['label'], predictions)
#     print('Task {}\tAccuracy: {:.2f}%'.format(t,acc))

# tasks.keys()

# load models : Load the models and save as a dictionary
models = {}
models[1] = ResNetForImageClassification.from_pretrained('/content/drive/MyDrive/ensembling/new_output/task_1')
models[2] = ResNetForImageClassification.from_pretrained('/content/drive/MyDrive/ensembling/new_output/task_2')
models[3] = ResNetForImageClassification.from_pretrained('/content/drive/MyDrive/ensembling/new_output/task_3')
models[4] = ResNetForImageClassification.from_pretrained('/content/drive/MyDrive/ensembling/task_4')
models[5] = ResNetForImageClassification.from_pretrained('/content/drive/MyDrive/ensembling/task_5')

"""# Selection and Aggregation Schemes"""

# random selection
# random.seed(2)
# predictions = []
# with torch.no_grad():
#     for example in dataset['valid']:
#         m = random.choice(list(models.keys()))
#         predicted_label = models[m](example['pixel_values'].unsqueeze(0).cuda()).logits.argmax(-1).item()
#         predictions.append(predicted_label)
# acc = accuracy_score(dataset['valid']['label'], predictions)
# print('Accuracy: {:.2f}%'.format(acc))

# Oracle Selection
# predictions = []
# with torch.no_grad():
#     for example in dataset['valid']:
#         m = None
#         for t in tasks.keys():
#             if example['label'] in tasks[t]:
#                 m = t
#                 break
#         predicted_label = models[m](example['pixel_values'].unsqueeze(0).cuda()).logits.argmax(-1).item()
#         predictions.append(predicted_label)
# acc = accuracy_score(dataset['valid']['label'], predictions)
# print('Accuracy: {:.2f}%'.format(acc))

# Confidence-based Selection
# predictions = []
# with torch.no_grad():
#     for example in dataset['valid']:
#         candidates = {}
#         for t in tasks.keys():
#             logits =  models[t](example['pixel_values'].unsqueeze(0).cuda()).logits
#             predicted_label = logits.argmax(-1).item()
#             pred_confidence = logits.max().item()
#             candidates[predicted_label] = pred_confidence
#         predicted_label = max(candidates, key=candidates.get)
#         predictions.append(predicted_label)
# acc = accuracy_score(dataset['valid']['label'], predictions)
# print('Accuracy: {:.2f}%'.format(acc))

# Entropy-based Selection
# def entropy(logits):
#     probs = torch.softmax(logits, dim=1)
#     log_probs = torch.log(probs + 1e-7) # Add a small epsilon to avoid log(0)
#     entropy = -torch.sum(probs * log_probs, dim=1)
#     return entropy.item()
# predictions = []
# with torch.no_grad():
#     for example in dataset['valid']:
#         candidates = {}
#         for t in tasks.keys():
#             logits =  models[t](example['pixel_values'].unsqueeze(0).cuda()).logits
#             predicted_label = logits.argmax(-1).item()
#             entr = entropy(logits)
#             candidates[predicted_label] = entr
#         predicted_label = min(candidates, key=candidates.get)
#         predictions.append(predicted_label)
# acc = accuracy_score(dataset['valid']['label'], predictions)
# print('Accuracy: {:.2f}%'.format(acc))

# Stacking
# Define Stacked model
class StackingEnsemble(nn.Module):
    def __init__(self, models, num_classes=200, hidden_size=512):
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(models) * hidden_size, num_classes)
        )

    def forward(self, pixel_values: torch.FloatTensor = None,):
        # Pass input through the ensemble
        outputs = []
        for model in self.models:
            last_hidden_state = model(pixel_values, return_dict=True, output_hidden_states=True)[
                'hidden_states'][-1]
            pooled_output = self.pooler(last_hidden_state)
            outputs.append(pooled_output)

        # Stack the penultimate layer feature representations
        stacked_features = torch.cat(outputs, dim=1)
        stacked_features = stacked_features.view(stacked_features.size(0), -1)

        # Pass stacked features through the classification head
        logits = self.classifier(stacked_features)

        return logits, stacked_features

# Dataset 
#Train baseline models on 80% of train data
# config = ResNetConfig.from_json_file('task_1\config.json')
# metric = evaluate.load("accuracy")
# models = {}
# Keep 20% of each task to finetune Stacked model
# finetune_data_tr = []
# finetune_data_val = []

# for t in tasks.keys():
#     # models[t] = ResNetForImageClassification(config)
#     # training_args = TrainingArguments(output_dir=f"./resnet_task_80%_{t}", num_train_epochs=10,
#     #                                   save_total_limit=1, overwrite_output_dir=True, auto_find_batch_size=True, save_strategy='epoch')
#     split = train_datasets[t].map(
#         process_example).train_test_split(test_size=0.2)
    
#     #get trained data
#     finetune_data_tr.append(split['train'])

#     #get test data
#     finetune_data_val.append(split['test'])
    
#     split['train'].set_format("pt", columns=['pixel_values'], output_all_columns=True)
#     split['test'].set_format("pt", columns=['pixel_values'], output_all_columns=True)
    
    # trainer = Trainer(
    #     model=models[t],
    #     args=training_args,
    #     train_dataset= split['train'],
    # )
    # trainer.train()

# Train Stacked model
# stacking_ensemble = StackingEnsemble(list(models.values()))
# stacked_dataset = concatenate_datasets(finetune_data).with_format("torch")
# train_dataloader = torch.utils.data.DataLoader(stacked_dataset, batch_size=32)

# num_epochs = 3

# # Freeze all other params to train only the last classification layer
# for name, param in stacking_ensemble.named_parameters():
#     if name not in ['classifier.1.weight', 'classifier.1.bias']:
#         param.requires_grad = False
# optimizer = optim.Adam(stacking_ensemble.parameters(), lr=0.001) 
# criterion = nn.CrossEntropyLoss() 
# for epoch in range(num_epochs):
#     print(f'EPOCH: {epoch}')
#     for ex in train_dataloader:
#         logits = stacking_ensemble(ex['pixel_values'])
#         loss = criterion(logits, ex['label'])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#initialize model
stacking_ensemble = StackingEnsemble(list(models.values()))

#Load the model
checkpoint = torch.load('/content/drive/MyDrive/ensembling/stacking_ensemble.pt')

# Load the weights into the model
stacking_ensemble.load_state_dict(checkpoint['model_state_dict'])

train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=32)

# from torch.utils.data import DataLoader

# for batch in train_dataloader:
#     logits, stacked_features = stacking_ensemble(batch['pixel_values'])
#     print (stacked_features.shape)
#     break

#Save tge data of the set
label_tr = dataset['train']['label']
df_lab_tr = pd.DataFrame(label_tr)
df_lab_tr.to_csv('/content/drive/MyDrive/ensembling/lab_tr.csv', index = False)

#save the labelled data val
label_val = dataset['valid']['label']
df_lab_te = pd.DataFrame(label_val)
df_lab_te.to_csv('/content/drive/MyDrive/ensembling/lab_te.csv', index = False)

# Define validation dataloader
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# batch_size = 32  # adjust this as needed

# #get the train loader
# train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=32)

# # Set the model to evaluation mode
# stacking_ensemble.eval()

# # Create empty tensors to store the embeddings and labels for all training points
# embeddings_tr = []
# temp = []
# # labels = torch.empty(len(train_dataset))

# # Extract embeddings for all training points

# with torch.no_grad():
#     for batch in tqdm(train_dataloader):
#         logits, stacked_features = stacking_ensemble(batch['pixel_values'])
#         embeddings_tr.append(stacked_features)

# embeddings_tr = torch.cat(embeddings_tr, dim=0)
# torch.save(embeddings_tr, "/content/drive/MyDrive/ensembling/embeddings_tr.pt")

# embeddings_tr_np = embeddings_tr.numpy()

# embeddings_tr_np.shape



# Define validation dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm

batch_size = 32  # adjust this as needed

#get the train loader
val_dataloader = torch.utils.data.DataLoader(dataset['valid'], batch_size=32)

# Set the model to evaluation mode
stacking_ensemble.eval()

# Create empty tensors to store the embeddings and labels for all training points
embeddings_val = []
temp = []
# labels = torch.empty(len(train_dataset))

# Extract embeddings for all val points

with torch.no_grad():
    for batch in tqdm(val_dataloader):
        logits, stacked_features = stacking_ensemble(batch['pixel_values'])
        embeddings_val.append(stacked_features)

embeddings_val = torch.cat(embeddings_val, dim=0)
torch.save(embeddings_val, "/content/drive/MyDrive/ensembling/embeddings_val.pt")

embeddings_val_np = embeddings_val.numpy()

embeddings_tr =torch.load("/content/drive/MyDrive/ensembling/embeddings_tr.pt")

embeddings_tr_np = embeddings_tr.numpy()

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= 500)
pca.fit(embeddings_tr_np)

train_embed = pca.transform(embeddings_tr_np)
val_embed =  pca.transform(embeddings_val_np)

train_embed.shape, val_embed.shape

train_labels = pd.read_csv('/content/drive/MyDrive/ensembling/lab_tr.csv')
train_labels = train_labels['0']

val_labels = pd.read_csv('/content/drive/MyDrive/ensembling/lab_te.csv')
val_labels = val_labels['0']

"""Nearest Neighbour Algorithm"""

#train_embed :  Numpy array for Training 
#train_labels : Labels for Training

#val_embed : Numpy array for Validation Data 
#val_labels : Labels for Training

# # Initialize a NearestNeighbors object and fit it on the stacked training data
from sklearn.neighbors import NearestNeighbors
k = 10
nn = NearestNeighbors(n_neighbors=k)
nn.fit(train_embed)

#Inferencing on the Test data
# Retrieve the indices of the top-k nearest neighbors for each validation sample
distances, indices = nn.kneighbors(val_embed)

#Strategy 1
# For each validation sample, count the number of votes for each class among the k neighbors
votes = np.zeros((len(val_embed), len(np.unique(train_labels))))

for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        #Update the matrix by taking votes on the train label for validation set
        votes[i, train_labels[neighbor]] += 1

# Select the ensemble member with the most votes for each validation sample
ensemble_members = np.argmax(votes, axis=1)

#compute validation accuracy
from sklearn.metrics import accuracy_score
accuracy_score(val_labels, ensemble_members)

from collections import Counter
import pickle

#get the prediction dict :  Prediction of the train samples from all the models
#key : task or model index, predictions_dict[key] = labels predicted by index of the train data. 
predictions_dict = pickle.load('/content/drive/MyDrive/ensembling/train_pred.pkl')

#Strategy 2
# For each validation sample, count the number of votes for each class among the k neighbors
votes = np.zeros((len(val_embed), len(np.unique(train_labels))))


# For each neighbour - find the best model out of the 5 ensembles
# Append the models in a dict with the count
# Sparsify the model list and select the top 3 
# Get predictions from the top 3 models from the train
# Do majority vote for pred on val set

#Load the json task list dict
import json
task_list = json.loads(pth_json)  #task list has the dictionary with key as the task and and value is the corr. labels

#test_pred
predictions_val = []

for i, neighbors in enumerate(indices):
    
    # Initialize a dictionary to store the number of votes for each ensemble member
    ensemble_model_ids = []
    
    # Loop over the k nearest neighbors and find the best ensemble member for each
    for neighbor in neighbors:

        # Get the class label of the neighbor in the training set
        class_label = train_labels[neighbor]

        #Get the corresponding task/model : task_list dict
        model_ids = [key for key in task_list if class_label in task_list[key]]
        
        # Task 1 : Get all the models for the nearest neighbours
        ensemble_model_ids.append(model_ids)
    
    # Sparsify the ensemble list and select the top 3 members by vote count
    counts = Counter(ensemble_model_ids)

    # Get the top 3 most common elements
    top_model_ids = [mid for mid, count in counts.most_common(3)]

    #init
    temp_pred = []
    pred_neb = []

    #Get the pre-saved predictions of the samples
    for neighbor in neighbors:
        for id in  ensemble_model_ids :
            #get prediction for each model for model i
            pred = predictions_dict[id][neighbor]
            #append the predicitons for all models
            temp_pred.append(pred)
        #join it for all neirhbours
        pred_neb = temp_pred + pred_neb

    # Get the predictions for this validation sample based on majority vote
    val_pred = np.argmax(np.bincount(pred_neb))
    
    #final pred
    predictions_val.append(val_pred)

#compute validation accuracy
accuracy_score(val_labels, predictions_val)

# Evaluate the StackingEnsemble model on the combined validation set
# Define validation dataloader
# val_dataset = dataset['valid'].with_format("torch")
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# device = 'cpu'
# correct = 0
# total = 0

# stacking_ensemble.eval()
# stacking_ensemble.to(device)

# with torch.no_grad():
    
#     # Loop through the data loader
#     for batch in  val_dataloader:
#         # Move images and labels to device
#         images = batch['pixel_values'].to(device)
#         # labels = ex['label'].to(device)

#         # Forward pass to get model predictions
#         outputs = stacking_ensemble(images)

#         # Get the predicted labels as the index of the maximum output value
#         _, predicted = torch.max(outputs.data, 1)

#         # Update total samples
#         total += labels.size(0)

#         # Update correct predictions
#         correct += (predicted == labels).sum().item()

# # Calculate accuracy
# accuracy = (correct / total) * 100

# print('Accuracy: {:.2f}%'.format(accuracy))





