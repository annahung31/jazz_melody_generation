import torch
from torch import nn

from utils.util import *
# from configs.data_config import DATA_CONFIG
# from configs.config_vae import MODEL_CONFIG, TRAIN_CONFIG



class classifier_binary(nn.Module):
    """Class that defines the model."""
    def __init__(self,DATA_CONFIG,MODEL_CONFIG,device):
        super(classifier_binary, self).__init__()
        self.device = device
        self.hidden_m = MODEL_CONFIG['classifier']['hidden_m']
        self.bar = DATA_CONFIG['bar']
        self.Bi = MODEL_CONFIG['classifier']['bidirectional'] 
        self.feature_size = DATA_CONFIG['feature_size']
        self.num_layers_en = MODEL_CONFIG['classifier']['num_layers_en']
        self.hidden_factor = self.Bi * self.num_layers_en
        self.BGRUm      = nn.GRU(input_size=DATA_CONFIG['feature_size'], hidden_size=self.hidden_m, num_layers=self.num_layers_en, batch_first=True, bidirectional=True)
        self.hid2label    = nn.Linear(self.hidden_m*self.Bi*self.bar, 1)


    def forward(self, m):
        batch_size = m.shape[0]
        m, hn = self.BGRUm(m)  #hn:(num_layers * num_directions, batch, hidden_size):
        h1 =  m.contiguous().view(batch_size,self.hidden_m*self.Bi*self.bar)
        predict_y = self.hid2label(h1)

        # predict_y_softmax = torch.softmax(predict_y, 1)
        predict_y_sigmoid = torch.sigmoid(predict_y)
        return predict_y_sigmoid, predict_y, h1





class classifier(nn.Module):
    """Class that defines the model."""
    def __init__(self,DATA_CONFIG,MODEL_CONFIG,device):
        super(classifier, self).__init__()
        self.device = device
        self.hidden_m = MODEL_CONFIG['classifier']['hidden_m']
        self.bar = DATA_CONFIG['bar']
        self.Bi = MODEL_CONFIG['classifier']['bidirectional'] 
        self.feature_size = DATA_CONFIG['feature_size']
        self.num_layers_en = MODEL_CONFIG['classifier']['num_layers_en']
        self.hidden_factor = self.Bi * self.num_layers_en

        self.BGRUm      = nn.GRU(input_size=DATA_CONFIG['feature_size'], hidden_size=self.hidden_m, num_layers=self.num_layers_en, batch_first=True, bidirectional=True)
        self.hid2label    = nn.Linear(self.hidden_m*self.Bi*self.bar, 2)


    def forward(self, m):
        batch_size = m.shape[0]
        m, hn = self.BGRUm(m)  #hn:(num_layers * num_directions, batch, hidden_size):
        h1 =  m.contiguous().view(batch_size,self.hidden_m*self.Bi*self.bar)
        predict_y = self.hid2label(h1)

        # predict_y_softmax = torch.softmax(predict_y, 1)
        predict_y_sigmoid = torch.sigmoid(predict_y)
        return predict_y_sigmoid, predict_y





