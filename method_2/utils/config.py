import torch
import os
from sequence import EventSeq, ControlSeq

#pylint: disable=E1101
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device:", torch.cuda.current_device())

model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),
    'control_dim': 24,
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3,
}

train = {
    # 'learning_rate': 0.001,
    'learning_rate': 0.00000000001,
    'batch_size': 8,
    'window_size': 500,
    'stride_size': 10,
    'use_transposition': False,
    'control_ratio': 0.7,
    'teacher_forcing_ratio': 1.0
}
