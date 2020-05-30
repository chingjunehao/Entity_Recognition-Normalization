import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from models import CharacterLevelCNN
from dataModule import MyDataset
from utils import *

# import lera 
# monitor loss from anywhere https://lera.ai/, the special link to your loss will be showed once it is logged

full_dataset = pd.read_csv('data/full_dataset.csv')

X = full_dataset["text"].tolist()
Y = full_dataset["label"].tolist()

# Find the maximum length of the words
max_len = 0
for t in full_dataset["text"]:
  max_len = max(max_len, len(str(t)))


train_bs = 1024
val_bs = 128
training_params = {"batch_size": train_bs,"shuffle": True}
val_params = {"batch_size": val_bs,"shuffle": False}

train_set = MyDataset(X, Y, max_length=max_len, train=True)
val_set = MyDataset(X, Y, max_length=max_len, train=False)

train_generator = DataLoader(train_set, **training_params)
val_generator = DataLoader(val_set, **val_params)

n_classes = 5
feature = "small"
if feature == "small":
    model = CharacterLevelCNN(input_length=max_len, n_classes=n_classes, input_dim=68, n_conv_filters=256, n_fc_neurons=1024)
elif feature == "large":
    model = CharacterLevelCNN(input_length=max_len, n_classes=n_classes, input_dim=68, n_conv_filters=1024, n_fc_neurons=2048)

model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = "adam"
if optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
elif optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# lera.log_hyperparams({
#         'title': 'Entities classification cross entropy loss',
#         'optimizer': 'Adam'
#         })

best_loss = 1e5
best_epoch = 0

model.train()
num_iter_per_epoch = len(train_generator)

num_epoch = 100
for epoch in range(num_epoch):
    for iter, batch in enumerate(train_generator):
        feature, label = batch

        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        predictions = model(feature)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()

        training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                          list_metrics=["accuracy", "f1_score"])
        print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, F1: {}".format(
            epoch + 1,
            num_epoch,
            iter + 1,
            num_iter_per_epoch,
            optimizer.param_groups[0]['lr'],
            loss, training_metrics["accuracy"], training_metrics["f1_score"]))
        # lera.log('training loss', loss.detach())

    model.eval()
    loss_ls = []
    val_label_ls = []
    val_pred_ls = []

    for batch in val_generator:
        val_feature, val_label = batch
        num_sample = len(val_label)
        if torch.cuda.is_available():
            val_feature = val_feature.cuda()
            val_label = val_label.cuda()
        with torch.no_grad():
            val_predictions = model(val_feature)

        val_loss = criterion(val_predictions, val_label)
        loss_ls.append(val_loss * num_sample)
        val_label_ls.extend(val_label.clone().cpu())
        val_pred_ls.append(val_predictions.clone().cpu())

    val_loss = sum(loss_ls) / val_set.__len__()
    val_pred = torch.cat(val_pred_ls, 0)
    val_label = np.array(val_label_ls)
    val_metrics = get_evaluation(val_label, val_pred.numpy(), list_metrics=["accuracy", "confusion_matrix", "f1_score"])
    print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, F1: {}".format(
        epoch + 1,
        num_epoch,
        optimizer.param_groups[0]['lr'],
        val_loss, val_metrics["accuracy"], val_metrics["f1_score"]))
    # lera.log('validation loss', val_loss.detach())
    
    model.train()
    es_patience = 5
    if val_loss  < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        # torch.save(model, "{}/char-cnn_{}_{}".format(opt.output, opt.dataset, opt.feature))
    # Early stopping
    if epoch - best_epoch > es_patience > 0:
        print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, val_loss, best_epoch))
        break
    if optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        current_lr /= 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

torch.save(model.state_dict(), 'entity-classifier.ckpt')