import numpy as np
from numpy import zeros, newaxis

import torch 
from torch.utils.data import DataLoader
import re

from models import CharacterLevelCNN

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

max_len = 98
model_sn = CharacterLevelCNN(input_length=max_len, input_dim=68, n_classes=2, n_conv_filters=256, n_fc_neurons=1024)
model_sn.load_state_dict(torch.load('trained_weights/serial-num-classifier.ckpt'))
model_3c = CharacterLevelCNN(input_length=max_len, input_dim=69, n_classes=3, n_conv_filters=256, n_fc_neurons=1024)
model_3c.load_state_dict(torch.load('trained_weights/ccnn-classifier-3c.ckpt'))

test_case = ["LONDON", "ASIA", "HONG KONG", "PLASTIC BOTTLE", "HARDWOOD TABLE", "XYZ 13423 / ILD", "ABC/ICL/20891NC", "44 CHINA ROAD, KOWLOON, HONG KONG",
             "33 TIMBER YARD, LONDON, L1 8XY", "SLOUGH SE12 2XY", "NVIDIA Ireland", "Marks and Spencers Ltd", "M&S Limited"]

def preprocessing(raw_text, vocab_space=False):
    vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    if vocab_space:
        vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
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

entities_sn = {0:"Non-SerialNumber",
         1: "SerialNumber"}

entities_3c = {0:"Company Name",
         1: "Location",
         2: "Physical Goods"}

def predict(rt):
    rt_no_space = rt.replace(" ", "")
    vec = preprocessing(rt_no_space.upper(), vocab_space=False)
    vec = torch.from_numpy(vec)

    with torch.no_grad():
        pred = str(entities_sn[int(np.argmax(model_sn(vec)))])
    if pred == "SerialNumber":
        return rt, pred
    else:
        vec = preprocessing(rt.upper(), vocab_space=True)
        vec = torch.from_numpy(vec)
        with torch.no_grad():
            pred = str(entities_3c[int(np.argmax(model_3c(vec)))])
        if pred == "Location" and hasNumbers(rt):
            pred = "Company address"
        return rt, pred

for raw_text in test_case:
    print(predict(raw_text))

# Final result 
# ('LONDON', 'Location')
# ('ASIA', 'Location')
# ('HONG KONG', 'Location')
# ('PLASTIC BOTTLE', 'Physical Goods')
# ('HARDWOOD TABLE', 'Physical Goods')
# ('XYZ 13423 / ILD', 'SerialNumber')
# ('ABC/ICL/20891NC', 'SerialNumber')
# ('44 CHINA ROAD, KOWLOON, HONG KONG', 'Company address')
# ('33 TIMBER YARD, LONDON, L1 8XY', 'Company address')
# ('SLOUGH SE12 2XY', 'Physical Goods')               (false)
# ('NVIDIA Ireland', 'Location')                      (false)
# ('Marks and Spencers Ltd', 'Company Name')
# ('M&S Limited', 'Company Name')