import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from models import SiameseNetwork, ContrastiveLoss
from dataModule import SiameseDataLoader

similarity_dataset = pd.read_csv('/content/drive/My Drive/vector.ai/en_similarity_dataset.csv')

tmp_X = similarity_dataset['Name'].tolist()
X = []
for x in tmp_X:
    X.append(ast.literal_eval(x))
    
Y = similarity_dataset['label'].tolist()
print(len(X), len(Y))

text_max_len = 0
for t in X:
    text_max_len = max(text_max_len, len(str(t[0])))
    text_max_len = max(text_max_len, len(str(t[1])))



train_bs = 1024
val_bs = 128
training_params = {"batch_size": train_bs,"shuffle": True}
val_params = {"batch_size": val_bs,"shuffle": False}

train_set = SiameseDataLoader(X, Y, max_length=text_max_len, train=True)
val_set = SiameseDataLoader(X, Y, max_length=text_max_len, train=False)

train_generator = DataLoader(train_set, **training_params)
val_generator = DataLoader(val_set, **val_params)

model = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

counter = []
loss_history = [] 
iteration_number= 0

best_loss = 1e5
best_epoch = 0

model.train()
num_iter_per_epoch = len(train_generator)

train_loss_plt = []
val_loss_plt = []
num_epoch = 1200

for epoch in range(num_epoch):
    for iter, batch in enumerate(train_generator):
        X0, X1, label = batch

        if torch.cuda.is_available():
            X0, X1 = X0.cuda(), X1.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        predict1, predict2 = model(X0, X1)
        loss = criterion(predict1, predict2, label)
        loss.backward()
        optimizer.step()
        train_loss_plt.append(loss)
        print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}".format(
            epoch + 1,
            num_epoch,
            iter + 1,
            num_iter_per_epoch,
            optimizer.param_groups[0]['lr'],
            loss))

    model.eval()
    loss_ls = []

    for batch in val_generator:
        val_X0, val_X1, val_label = batch
        num_sample = len(val_label)
        if torch.cuda.is_available():
            val_X0, val_X1 = val_X0.cuda(), val_X1.cuda()
            val_label = val_label.cuda()
        with torch.no_grad():
            val_pred0, val_pred1 = model(val_X0, val_X1)

        val_loss = criterion(val_pred0, val_pred1, val_label)
        loss_ls.append(val_loss * num_sample)

    val_loss = sum(loss_ls) / val_set.__len__()
    val_loss_plt.append(val_loss)
    print("Epoch: {}/{}, Lr: {}, Loss: {}".format(
        epoch + 1,
        num_epoch,
        optimizer.param_groups[0]['lr'],
        val_loss))
    
    model.train()
    es_patience = 10
    if val_loss  < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), '/content/drive/My Drive/vector.ai/test-s.ckpt')
    # Early stopping
    if epoch - best_epoch > es_patience > 0:
        print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, val_loss, best_epoch))
        break

# For plotting of loss to see if there's any issue converging
# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.plot(train_loss_plt)
# plt.show()
