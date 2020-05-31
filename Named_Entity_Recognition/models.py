import torch.nn as nn

class CharacterLevelCNN(nn.Module):
    def __init__(self, input_length, input_dim, n_classes, n_conv_filters=256, n_fc_neurons=1024):

        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=1, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=1, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=1, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=1, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        self.fc1 = nn.Sequential(nn.Linear(768, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
              nn.init.zeros_(m.bias)
              

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output