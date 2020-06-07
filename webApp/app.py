
from flask import Flask, jsonify, render_template, request
from numpy import zeros, newaxis
import numpy as np
from models import CharacterLevelCNN
from torch.utils.data import DataLoader
import torch 

model = CharacterLevelCNN(input_length=98, n_classes=5, input_dim=68, n_conv_filters=256, n_fc_neurons=1024)
model.load_state_dict(torch.load('trained_weights/entity-classifier.ckpt'))

max_len = 98
entity = ""
entities = {0:"Company", 2: "Company",
         1: "Location",
         4: "SerialNumber",
         3: "PhysicalGoods"}

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

# webapp
app = Flask(__name__)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    raw_text = request.text # To be completed
    for i in DataLoader(preprocessing(raw_text.upper())):
      with torch.no_grad():
        entity = entities[int(np.argmax(model(i)))]

    return jsonify(entity)

if __name__ == '__main__':
    app.run(debug=True)
