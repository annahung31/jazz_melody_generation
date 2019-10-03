import numpy as np
import os
import random
import time
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import ipdb
import time

from utils.util import *
from configs.data_config import DATA_CONFIG
from utils.pianoroll2midi import *



class ClassifierTrainer_binary():
    def __init__(self, TRAIN_CONFIG, device):
        super(ClassifierTrainer_binary, self).__init__()
        self.device = device
        self.lr = TRAIN_CONFIG['lr']
        self.lr_step1 = TRAIN_CONFIG['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['lr_gamma']
    def loss_classifier(self,predict_y,predict_y_sigmoid, test_y):
        test_y = test_y.reshape(test_y.shape[0],1)
        BCE = F.binary_cross_entropy(predict_y_sigmoid, test_y, reduction='sum')

        predict_y = [1 if item>=0.5 else 0 for item in predict_y_sigmoid] 
        correct_predict = 0
        for i in range(len(predict_y)):
            correct_predict += (predict_y[i] == int(test_y[i].item()))
        return BCE, correct_predict

    def train_classifier(self,model, epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0; train_acc=0
        for batch_idx, (m,y) in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_y_sigmoid, predict_y = model(batch_m)
            loss, acc= self.loss_classifier(predict_y,predict_y_sigmoid, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('====> Epoch: {:3d} Average loss: {:.8f}, average acc: {:.8f}'.format(epoch, train_loss, train_acc))
        return train_loss, train_acc
        

    def test_classifier(self,model, epoch,test_loader):
        model.eval()
        test_loss = 0; test_acc = 0
        for batch_idx,  (m,y) in enumerate(test_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_y_sigmoid, predict_y = model(batch_m)
            loss, acc = self.loss_classifier(predict_y, predict_y_sigmoid, batch_y)
            test_loss += loss.item()
            test_acc += acc
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        print('===============>    test loss: {:.8f}, average acc: {:.8f}'.format(test_loss,test_acc ))
        return test_loss, test_acc
        

    # def eval_c():
    #     model_C = classifier(hidden_m=MODEL_CONFIG['c_hidden_m']).to(device)
    #     model_C.load_state_dict(torch.load('/home/annahung/project/anna_jam/presets/classifer_balance.pt'))

    #     _,_,_,_, eval_m = parse_data_c()
    #     eval_m = torch.from_numpy(eval_m).type(torch.FloatTensor)
    #     eval_m = Variable(eval_m).to(device)
    #     predict_y = model_C(eval_m)
    #     predict_y = [1 if item>=0.5 else 0 for item in predict_y] 
    #     answer_y = np.concatenate((np.zeros(10),np.ones(20)),axis=0)

    #     correct_predict = 0
    #     for i in range(len(predict_y)):
    #         correct_predict += (predict_y[i] == int(answer_y[i]))
    #     print(correct_predict, 'total:',answer_y.shape[0], 'ac:{:.4f}'.format(correct_predict/answer_y.shape[0]))
    #     print(predict_y)



class ClassifierTrainer():
    def __init__(self, TRAIN_CONFIG, device):
        super(ClassifierTrainer, self).__init__()
        self.device = device
        self.lr = TRAIN_CONFIG['lr']
        self.lr_step1 = TRAIN_CONFIG['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['lr_gamma']
    def loss_classifier(self,predict_y,predict_y_sigmoid, test_y):
        
        _, answer = torch.max(test_y, 1)
        BCE = F.binary_cross_entropy(predict_y_sigmoid, test_y, reduction='sum')
        _, predict = torch.max(predict_y_sigmoid, 1)
        
        correct_predict = 0
        for i in range(len(predict)):
            correct_predict += (predict[i].item() == answer[i].item())
        return BCE, correct_predict

    def train_classifier(self,model, epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0; train_acc=0
        for batch_idx, (m,y) in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_y_sigmoid, predict_y = model(batch_m)
            loss, acc= self.loss_classifier(predict_y,predict_y_sigmoid, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('====> Epoch: {:3d} Average loss: {:.8f}, average acc: {:.8f}'.format(epoch, train_loss, train_acc))
        return train_loss, train_acc
        

    def test_classifier(self,model, epoch,test_loader):
        model.eval()
        test_loss = 0; test_acc = 0
        for batch_idx,  (m,y) in enumerate(test_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_y_sigmoid, predict_y = model(batch_m)
            loss, acc = self.loss_classifier(predict_y, predict_y_sigmoid, batch_y)
            test_loss += loss.item()
            test_acc += acc
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        print('===============>    test loss: {:.8f}, average acc: {:.8f}'.format(test_loss,test_acc ))
        return test_loss, test_acc
        

    # def eval_c():
    #     model_C = classifier(hidden_m=MODEL_CONFIG['c_hidden_m']).to(device)
    #     model_C.load_state_dict(torch.load('/home/annahung/project/anna_jam/presets/classifer_balance.pt'))

    #     _,_,_,_, eval_m = parse_data_c()
    #     eval_m = torch.from_numpy(eval_m).type(torch.FloatTensor)
    #     eval_m = Variable(eval_m).to(device)
    #     predict_y = model_C(eval_m)
    #     predict_y = [1 if item>=0.5 else 0 for item in predict_y] 
    #     answer_y = np.concatenate((np.zeros(10),np.ones(20)),axis=0)

    #     correct_predict = 0
    #     for i in range(len(predict_y)):
    #         correct_predict += (predict_y[i] == int(answer_y[i]))
    #     print(correct_predict, 'total:',answer_y.shape[0], 'ac:{:.4f}'.format(correct_predict/answer_y.shape[0]))
    #     print(predict_y)
