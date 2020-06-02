from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class char2vecDataLoader(Dataset):
    def __init__(self, X, Y, max_length, train=True, vocab_space=False):
        self.vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        if vocab_space:
            self.vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")

        self.vocab_length = len(self.vocabulary)

        self.identity_mat = np.identity(self.vocab_length)
        self.max_length = max_length
        self.texts = []
        self.labels = []

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=2020, stratify=Y)

        if train:
            self.texts = X_train
            self.labels = y_train
        else:
            self.texts = X_val
            self.labels = y_val

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def char2vec(self, raw_text): 
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(str(raw_text)) if i in self.vocabulary], dtype=np.float32)

        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate((data, np.zeros((self.max_length - len(data), self.vocab_length), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, self.vocab_length), dtype=np.float32)

        return data

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = self.char2vec(raw_text)
        label = self.labels[index]
        return data, label

class SiameseDataLoader(Dataset):
    def __init__(self, X, Y, max_length=98, train=True):
        self.vocabulary = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.vocab_length = len(self.vocabulary)

        self.identity_mat = np.identity(len(self.vocabulary))
        self.max_length = max_length
        self.texts = []
        self.labels = []

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=2020, stratify=Y)

        if train:
          self.texts = X_train
          self.labels = y_train
        else:
          self.texts = X_val
          self.labels = y_val

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def char2vec(self, raw_text):
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(str(raw_text)) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate((data, np.zeros((self.max_length - len(data), self.vocab_length), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)

        return data

    def __getitem__(self, index):
        raw_text1 = self.texts[index][0]
        raw_text2 = self.texts[index][1]
        label = self.labels[index]

        X0, X1 = "", ""
        X0, X1 = self.char2vec(raw_text1), self.char2vec(raw_text2)

        return X0, X1, label