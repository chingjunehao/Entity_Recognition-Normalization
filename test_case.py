from numpy import zeros, newaxis
import numpy as np
from models import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch 
import pandas as pd

full_dataset = pd.read_csv('data/full_dataset.csv')
# Find the maximum length of the words
max_len = 0
for t in full_dataset["text"]:
  max_len = max(max_len, len(str(t)))

n_model = CharacterLevelCNN(input_length=max_len, n_classes=5, input_dim=68, n_conv_filters=256, n_fc_neurons=1024)
n_model.load_state_dict(torch.load('trained_weights/entity-classifier.ckpt'))

# - Company names - “Marks and Spencers Ltd”, “M&S Limited”, “NVIDIA Ireland”, etc.
# - Company addresses: “SLOUGH SE12 2XY”, “33 TIMBER YARD, LONDON, L1 8XY”, “44 CHINA ROAD, KOWLOON, HONG KONG”
# - Serial numbers: “XYZ 13423 / ILD”, “ABC/ICL/20891NC”
# - Physical Goods: “HARDWOOD TABLE”, “PLASTIC BOTTLE”
# - Locations: “LONDON”, “HONG KONG”, “ASIA” 
test_case = ["LONDON", "ASIA", "HONG KONG", "PLASTIC BOTTLE", "HARDWOOD TABLE", "XYZ 13423 / ILD", "ABC/ICL/20891NC", "44 CHINA ROAD, KOWLOON, HONG KONG",
             "33 TIMBER YARD, LONDON, L1 8XY", "SLOUGH SE12 2XY", "NVIDIA Ireland", "Marks and Spencers Ltd", "M&S Limited"]



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

# 2 was supposed to be PERSON entity, because some of the company names 
# are just the same as human names, so I made it as company name too
entities = {0:"Company", 2: "Company",
         1: "Location",
         4: "SerialNumber",
         3: "PhysicalGoods"}


for raw_text in test_case:
  for i in DataLoader(preprocessing(raw_text.upper())):
    with torch.no_grad():
      print(raw_text, entities[int(np.argmax(n_model(i)))])
