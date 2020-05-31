from numpy import zeros, newaxis
import numpy as np
from models import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch 
import pandas as pd

n_model = CharacterLevelCNN(input_length=98, input_dim=68, n_classes=5, n_conv_filters=256, n_fc_neurons=1024)
n_model.load_state_dict(torch.load('trained_weights/entity-classifier.ckpt'))
# - Company names - “Marks and Spencers Ltd”, “M&S Limited”, “NVIDIA Ireland”, etc.
# - Company addresses: “SLOUGH SE12 2XY”, “33 TIMBER YARD, LONDON, L1 8XY”, “44 CHINA ROAD, KOWLOON, HONG KONG”
# - Serial numbers: “XYZ 13423 / ILD”, “ABC/ICL/20891NC”
# - Physical Goods: “HARDWOOD TABLE”, “PLASTIC BOTTLE”
# - Locations: “LONDON”, “HONG KONG”, “ASIA” 
test_case = ["LONDON", "ASIA", "HONG KONG", "PLASTIC BOTTLE", "HARDWOOD TABLE", "XYZ 13423 / ILD", "ABC/ICL/20891NC", "44 CHINA ROAD, KOWLOON, HONG KONG",
             "33 TIMBER YARD, LONDON, L1 8XY", "SLOUGH SE12 2XY", "NVIDIA Ireland", "Marks and Spencers Ltd", "M&S Limited"]

max_len = 98
def preprocessing(raw_text):
    vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    identity_mat = np.identity(len(vocabulary))
    data = np.array([identity_mat[vocabulary.index(i)] for i in list(raw_text) if i in vocabulary], dtype=np.float32)
    if len(data) > max_len:
        data = data[:max_len]
    elif 0 < len(data) < max_len:
        data = np.concatenate(
          (data, np.zeros((max_len - len(data), len(vocabulary)), dtype=np.float32)))
    elif len(data) == 0:
        data = np.zeros((max_len, len(vocabulary)), dtype=np.float32)
    tmp_data = data[newaxis, :, :]
    return tmp_data

def predict(raw_text):
    for i in DataLoader(preprocessing(raw_text.upper())):
      with torch.no_grad():
        print(raw_text, entities[int(np.argmax(n_model(i)))])
# 2 was supposed to be PERSON entity, because some of the company names 
# are just the same as human names, so I made it as company name too
entities = {0:"Company", 2: "Company",
         1: "Location",
         4: "SerialNumber",
         3: "PhysicalGoods"}


for raw_text in test_case:
  predict(raw_text)


### classifier 1 result (fixed value mean and std initialization, val-loss:0.49)
# LONDON Location
# ASIA Location
# HONG KONG Location
# PLASTIC BOTTLE Location (false)
# HARDWOOD TABLE Location (false)
# XYZ 13423 / ILD SerialNumber
# ABC/ICL/20891NC SerialNumber
# 44 CHINA ROAD, KOWLOON, HONG KONG Company (false)
# 33 TIMBER YARD, LONDON, L1 8XY Location
# SLOUGH SE12 2XY Location
# NVIDIA Ireland Location (false)
# Marks and Spencers Ltd Company
# M&S Limited Company

#### classifier 2 result (xavier initialization, train with bigger batch size, val-loss: 0.43~)
# LONDON Location
# ASIA Location
# HONG KONG Location
# PLASTIC BOTTLE PhysicalGoods
# HARDWOOD TABLE Location (false)
# XYZ 13423 / ILD Location (false)
# ABC/ICL/20891NC SerialNumber
# 44 CHINA ROAD, KOWLOON, HONG KONG Location
# 33 TIMBER YARD, LONDON, L1 8XY Company (false)
# SLOUGH SE12 2XY SerialNumber
# NVIDIA Ireland Location (false)
# Marks and Spencers Ltd Company
# M&S Limited Company
