from numpy import zeros, newaxis
import numpy as np
from models import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch 
import pandas as pd

model = CharacterLevelCNN(input_length=98, input_dim=68, n_classes=4, n_conv_filters=256, n_fc_neurons=1024)
model.load_state_dict(torch.load('trained_weights/char-cnn-classifier-4c.ckpt'))

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
      print(raw_text, entities[int(np.argmax(model(i)))])

entities = {0:"Company",
         1: "Location",
         2: "PhysicalGoods",
         3: "SerialNumber"}

for raw_text in test_case:
  predict(raw_text)


### classification result 
# LONDON Location
# ASIA Location
# HONG KONG Location
# PLASTIC BOTTLE Company (false)
# HARDWOOD TABLE Location (false)
# XYZ 13423 / ILD SerialNumber
# ABC/ICL/20891NC SerialNumber
# 44 CHINA ROAD, KOWLOON, HONG KONG Location
# 33 TIMBER YARD, LONDON, L1 8XY Company (false)
# SLOUGH SE12 2XY Company (false)
# NVIDIA Ireland Location (false)
# Marks and Spencers Ltd Company
# M&S Limited Company
