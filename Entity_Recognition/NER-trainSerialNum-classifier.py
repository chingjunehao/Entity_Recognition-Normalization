import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

from models import CharacterLevelCNN
from dataModule import char2vecDataLoader
from utils import get_evaluation

ner_dataset = pd.read_csv('data/sn_dataset.csv')

X = ner_dataset["text"].tolist()
Y = ner_dataset["label"].tolist()

# Find the maximum length of the words
text_max_len = 0
for t in ner_dataset["text"]:
    text_max_len = max(text_max_len, len(str(t)))
print(text_max_len)


train_bs = 1024
val_bs = 128
n_classes = 2
num_epoch = 200

def train(X, Y, train_bs, val_bs, n_classes, num_epoch):
    training_params = {"batch_size": train_bs,"shuffle": True}
    val_params = {"batch_size": val_bs,"shuffle": False}

    train_set = char2vecDataLoader(X, Y, max_length=text_max_len, train=True)
    val_set = char2vecDataLoader(X, Y, max_length=text_max_len, train=False)

    train_generator = DataLoader(train_set, **training_params)
    val_generator = DataLoader(val_set, **val_params) 

    model = CharacterLevelCNN(input_length=text_max_len, input_dim=train_set.vocab_length, n_classes=n_classes, n_conv_filters=256, n_fc_neurons=1024)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_loss = 1e5
    best_epoch = 0
    
    model.train()
    num_iter_per_epoch = len(train_generator)

    train_loss_plt = []
    val_loss_plt = []

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
            train_loss_plt.append(loss)
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
        val_loss_plt.append(val_loss)
        val_pred = torch.cat(val_pred_ls, 0)
        val_label = np.array(val_label_ls)
        val_metrics = get_evaluation(val_label, val_pred.numpy(), list_metrics=["accuracy", "confusion_matrix", "f1_score"])
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}, F1: {}".format(
            epoch + 1,
            num_epoch,
            optimizer.param_groups[0]['lr'],
            val_loss, val_metrics["accuracy"], val_metrics["f1_score"]))
        
        model.train()
        es_patience = 5
        if val_loss  < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'trained_weights/serial-num-classifier.ckpt')
        # Early stopping
        if epoch - best_epoch > es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, val_loss, best_epoch))
            break

train(X, Y, train_bs, val_bs, n_classes, num_epoch)
# For plotting of loss to see if there's any issue converging
# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.plot(train_loss_plt)
# plt.show()
