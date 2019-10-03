import numpy as np
import os
import random
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import ipdb
import time

from utils.util import *
from utils.pianoroll2midi import *



class VAE_jazz_only_Trainer_no_pcd():
    def __init__(self,MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device):
        super(VAE_jazz_only_Trainer_no_pcd, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.lr = TRAIN_CONFIG['vae']['lr_vae']
        self.lr_step1 = TRAIN_CONFIG['vae']['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['vae']['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['vae']['lr_gamma']
        self.loss_beta = TRAIN_CONFIG['vae']['loss_beta']
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAIN_CONFIG = TRAIN_CONFIG
        self.DATA_CONFIG = DATA_CONFIG
    def loss_vae(self,epoch, predict_m, test_m,mu, logvar):
        # loss 
        # beta = (0.0055*epoch)*(0.0055*epoch)
        beta = self.loss_beta

        BCEm = F.binary_cross_entropy(predict_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), test_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), reduction='sum')
        # MSE = criterion(predict_m, test_m)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        

        predict_pitch_type = pitch_type(predict_m)
        data_pitch_type = pitch_type(test_m)
        pitch_type_diff = abs(data_pitch_type - predict_pitch_type)
        loss = BCEm*(1 - beta) + beta* KLD
        
        # accuracy
        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
        accm = torch.sum(m==pm)/m.shape[1]

        mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
        pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
        accmr = torch.sum(mr==pmr)/mr.shape[1]
        return BCEm, KLD, loss, accm, accmr

    def train_vae(self,model,epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0 ; train_accm=0; train_accmr=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx, m in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            predict_m,  mu, logvar, z = model(batch_m)

            BCEm, KLD,loss, accm, accmr = self.loss_vae(epoch, predict_m, batch_m, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss  += loss.item()
            BCE_loss    += BCEm.item()
            KLD_loss    += KLD.item()
            train_accm  += accm.item()
            train_accmr += accmr.item()

            if batch_idx % 20 == 0:
                file_name = '_epoch_'+ str(epoch) +  'batch_'+ str(batch_idx)

        history_train = [[] for i in range(3)]
        train_loss /= len(train_loader.dataset)
        BCE_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)
        train_accm /= len(train_loader.dataset)
        train_accmr /= len(train_loader.dataset)
        history_train[0] += [train_loss]
        history_train[1] += [train_accm]
        history_train[2] += [train_accmr]
        print('====> Epoch: {:3d} Average loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}'.format(epoch, 
           train_loss, BCE_loss,KLD_loss, train_accm, train_accmr ))
        return train_loss, BCE_loss, KLD_loss, len(train_loader.dataset)

    def test_vae(self,model, epoch,test_loader):
        model.eval()
        test_loss = 0; test_accm=0; test_accmr=0; 
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx,  m in enumerate(test_loader):
          batch_m = Variable(m).to(self.device)
          predict_m, mu, logvar, z = model(batch_m)       

          BCEm, KLD,loss, accm, accmr= self.loss_vae(epoch, predict_m, batch_m, mu, logvar)
          test_loss += loss.item()
          BCE_loss    += BCEm.item()
          KLD_loss    += KLD.item()
          test_accm += accm.item()
          test_accmr += accmr.item()


        test_loss /= len(test_loader.dataset)
        BCE_loss /= len(test_loader.dataset)
        KLD_loss /= len(test_loader.dataset)
        test_accm /= len(test_loader.dataset)
        test_accmr /= len(test_loader.dataset)

        # print('===============>    test loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}'.format(test_loss,           
        #    BCE_loss,KLD_loss, test_accm, test_accmr))  # accuracy_melody, accuracy_melody_rhythm

        return test_loss, BCE_loss, KLD_loss, len(test_loader.dataset)






class VAE_jazz_only_Trainer():
    def __init__(self,MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device):
        super(VAE_jazz_only_Trainer, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.lr = TRAIN_CONFIG['vae']['lr_vae']
        self.lr_step1 = TRAIN_CONFIG['vae']['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['vae']['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['vae']['lr_gamma']
        self.loss_beta = TRAIN_CONFIG['vae']['loss_beta']
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAIN_CONFIG = TRAIN_CONFIG
        self.DATA_CONFIG = DATA_CONFIG
    def loss_vae(self,epoch, predict_m, test_m,mu, logvar):
        # loss 
        # beta = (0.0055*epoch)*(0.0055*epoch)
        beta = self.loss_beta

        BCEm = F.binary_cross_entropy(predict_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), test_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), reduction='sum')
        # MSE = criterion(predict_m, test_m)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        

        predict_pitch_type = pitch_type(predict_m)
        data_pitch_type = pitch_type(test_m)
        pitch_type_diff = abs(data_pitch_type - predict_pitch_type)
        loss = BCEm*(1 - beta) + beta* KLD  +  pitch_type_diff
        
        # accuracy
        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
        accm = torch.sum(m==pm)/m.shape[1]

        mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
        pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
        accmr = torch.sum(mr==pmr)/mr.shape[1]
        return BCEm, KLD, loss, accm, accmr, pitch_type_diff

    def train_vae(self,model,epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0 ; train_accm=0; train_accmr=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx, m in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            predict_m,  mu, logvar, z = model(batch_m)

            BCEm, KLD,loss, accm, accmr, pitch_type_diff = self.loss_vae(epoch, predict_m, batch_m, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss  += loss.item()
            BCE_loss    += BCEm.item()
            KLD_loss    += KLD.item()
            train_accm  += accm.item()
            train_accmr += accmr.item()

            if batch_idx % 20 == 0:
                file_name = '_epoch_'+ str(epoch) +  'batch_'+ str(batch_idx)

        history_train = [[] for i in range(3)]
        train_loss /= len(train_loader.dataset)
        BCE_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)
        train_accm /= len(train_loader.dataset)
        train_accmr /= len(train_loader.dataset)
        history_train[0] += [train_loss]
        history_train[1] += [train_accm]
        history_train[2] += [train_accmr]
        print('====> Epoch: {:3d} Average loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, pitch_type_diff:{:.4f}'.format(epoch, 
           train_loss, BCE_loss,KLD_loss, train_accm, train_accmr , pitch_type_diff))
        return train_loss, BCE_loss, KLD_loss, len(train_loader.dataset)

    def test_vae(self,model, epoch,test_loader):
        model.eval()
        test_loss = 0; test_accm=0; test_accmr=0; 
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx,  m in enumerate(test_loader):
          batch_m = Variable(m).to(self.device)
          predict_m, mu, logvar, z = model(batch_m)       

          BCEm, KLD,loss, accm, accmr, pitch_type_diff= self.loss_vae(epoch, predict_m, batch_m, mu, logvar)
          test_loss += loss.item()
          BCE_loss    += BCEm.item()
          KLD_loss    += KLD.item()
          test_accm += accm.item()
          test_accmr += accmr.item()


        test_loss /= len(test_loader.dataset)
        BCE_loss /= len(test_loader.dataset)
        KLD_loss /= len(test_loader.dataset)
        test_accm /= len(test_loader.dataset)
        test_accmr /= len(test_loader.dataset)

        # print('===============>    test loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, pitch_type_diff:{:.4f} '.format(test_loss,           
        #    BCE_loss,KLD_loss, test_accm, test_accmr,pitch_type_diff))  # accuracy_melody, accuracy_melody_rhythm

        return test_loss, BCE_loss, KLD_loss, len(test_loader.dataset)









class VAETrainer_no_pcd_one_hot():
    def __init__(self,MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device):
        super(VAETrainer_no_pcd_one_hot, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.lr = TRAIN_CONFIG['vae']['lr_vae']
        self.lr_step1 = TRAIN_CONFIG['vae']['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['vae']['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['vae']['lr_gamma']
        self.loss_beta = TRAIN_CONFIG['vae']['loss_beta']
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAIN_CONFIG = TRAIN_CONFIG
        self.DATA_CONFIG = DATA_CONFIG
    def loss_vae(self,epoch, predict_m, test_m, test_y,model_C,mu, logvar):
        beta = self.loss_beta

        BCEm = F.binary_cross_entropy(predict_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), test_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


        predict_y_sigmoid, _, _ = model_C(predict_m)
        _, answer = torch.max(test_y,1)
        answer = answer.type(torch.FloatTensor).to(self.device)
        BCEc = F.binary_cross_entropy(predict_y_sigmoid,answer, reduction='sum')
        predict_y2 = [1 if item>=0.5 else 0 for item in predict_y_sigmoid] 
        

        correct_predict = 0
        for i in range(len(predict_y2)):
            correct_predict += (predict_y2[i] == int(answer[i].item()))
        ####
        loss = BCEm*(1 - beta) + beta* KLD + BCEc 
        
        # accuracy
        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
        accm = torch.sum(m==pm)/m.shape[1]

        mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
        pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
        accmr = torch.sum(mr==pmr)/mr.shape[1]
        return BCEm, KLD, loss, accm, accmr, correct_predict

    def train_vae(self,model, model_C,epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0 ; train_accm=0; train_accmr=0; train_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx, (m,y) in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_m,  mu, logvar, z = model(batch_m, batch_y)

            BCEm, KLD,loss, accm, accmr, correct_predict = self.loss_vae(epoch, predict_m, batch_m,batch_y,model_C, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss  += loss.item()
            BCE_loss    += BCEm.item()
            KLD_loss    += KLD.item()
            train_accm  += accm.item()
            train_accmr += accmr.item()
            train_correct_predict += correct_predict


        history_train = [[] for i in range(3)]
        train_loss /= len(train_loader.dataset)
        BCE_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)
        train_accm /= len(train_loader.dataset)
        train_accmr /= len(train_loader.dataset)
        train_correct_predict /= len(train_loader.dataset)
        history_train[0] += [train_loss]
        history_train[1] += [train_accm]
        history_train[2] += [train_accmr]
        print('====> Epoch: {:3d} Average loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, correct_genre:{:.4f},train_correct_predict:{:.4f}'.format(epoch, 
           train_loss, BCE_loss,KLD_loss, train_accm, train_accmr,train_correct_predict))
        return train_loss, BCE_loss, KLD_loss, len(train_loader.dataset)

    def test_vae(self,model,model_C, epoch,test_loader):
        model.eval()
        test_loss = 0; test_accm=0; test_accmr=0; test_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx,  (m,y) in enumerate(test_loader):
          batch_m = Variable(m).to(self.device)
          batch_y = Variable(y).to(self.device)
          predict_m, mu, logvar, z = model(batch_m, batch_y)       

          BCEm, KLD,loss, accm, accmr, correct_predict= self.loss_vae(epoch, predict_m, batch_m,batch_y, model_C, mu, logvar)
          test_loss += loss.item()
          BCE_loss    += BCEm.item()
          KLD_loss    += KLD.item()
          test_accm += accm.item()
          test_accmr += accmr.item()
          test_correct_predict += correct_predict


        test_loss /= len(test_loader.dataset)
        BCE_loss /= len(test_loader.dataset)
        KLD_loss /= len(test_loader.dataset)
        test_accm /= len(test_loader.dataset)
        test_accmr /= len(test_loader.dataset)
        test_correct_predict /= len(test_loader.dataset)

        # print('===============>    test loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, correct genre:{:.4f},:{:.4f} '.format(test_loss,           
           # BCE_loss,KLD_loss, test_accm, test_accmr,test_correct_predict))  # accuracy_melody, accuracy_melody_rhythm

        return test_loss, BCE_loss, KLD_loss, len(test_loader.dataset)







class VAETrainer_no_pcd():
    def __init__(self,MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device):
        super(VAETrainer_no_pcd, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.lr = TRAIN_CONFIG['vae']['lr_vae']
        self.lr_step1 = TRAIN_CONFIG['vae']['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['vae']['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['vae']['lr_gamma']
        self.loss_beta = TRAIN_CONFIG['vae']['loss_beta']
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAIN_CONFIG = TRAIN_CONFIG
        self.DATA_CONFIG = DATA_CONFIG
    def loss_vae(self,epoch, predict_m, test_m, test_y,model_C,mu, logvar):
        # loss 
        # beta = (0.0055*epoch)*(0.0055*epoch)
        beta = self.loss_beta

        BCEm = F.binary_cross_entropy(predict_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), test_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), reduction='sum')
        # MSE = criterion(predict_m, test_m)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        predict_y_sigmoid, _, _ = model_C(predict_m)
        #### find classification problem loss 
        BCEc = F.binary_cross_entropy(predict_y_sigmoid,test_y, reduction='sum')
        predict_y2 = [1 if item>=0.5 else 0 for item in predict_y_sigmoid] 

        correct_predict = 0
        for i in range(len(predict_y2)):
            correct_predict += (predict_y2[i] == int(test_y[i].item()))
        ####
        loss = BCEm*(1 - beta) + beta* KLD + BCEc 
        
        # accuracy
        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
        accm = torch.sum(m==pm)/m.shape[1]

        mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
        pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
        accmr = torch.sum(mr==pmr)/mr.shape[1]
        return BCEm, KLD, loss, accm, accmr, correct_predict

    def train_vae(self,model, model_C,epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0 ; train_accm=0; train_accmr=0; train_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx, (m,y) in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_m,  mu, logvar, z = model(batch_m, batch_y)

            BCEm, KLD,loss, accm, accmr, correct_predict = self.loss_vae(epoch, predict_m, batch_m,batch_y,model_C, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss  += loss.item()
            BCE_loss    += BCEm.item()
            KLD_loss    += KLD.item()
            train_accm  += accm.item()
            train_accmr += accmr.item()
            train_correct_predict += correct_predict

            # if batch_idx % 20 == 0:
            #     file_name = '_epoch_'+ str(epoch) +  'batch_'+ str(batch_idx)

                # m1, mr1 = shape_to_pianoroll(batch_m[0], DATA_CONFIG['bar'],   DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
                # save_image(torch.from_numpy(m1).type(torch.FloatTensor),'/home/annahung/project/anna_jam/output/train_sample_' + file_name + '.png')
                # predict_m = predict_m.cpu().detach()
                # gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
                # save_image(torch.from_numpy(gen_m1).type(torch.FloatTensor),'./output/train_output_' + file_name + '.png')


        history_train = [[] for i in range(3)]
        train_loss /= len(train_loader.dataset)
        BCE_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)
        train_accm /= len(train_loader.dataset)
        train_accmr /= len(train_loader.dataset)
        train_correct_predict /= len(train_loader.dataset)
        history_train[0] += [train_loss]
        history_train[1] += [train_accm]
        history_train[2] += [train_accmr]
        print('====> Epoch: {:3d} Average loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, correct_genre:{:.4f},train_correct_predict:{:.4f}'.format(epoch, 
           train_loss, BCE_loss,KLD_loss, train_accm, train_accmr,train_correct_predict))
        return train_loss, BCE_loss, KLD_loss, len(train_loader.dataset)

    def test_vae(self,model,model_C, epoch,test_loader):
        model.eval()
        test_loss = 0; test_accm=0; test_accmr=0; test_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx,  (m,y) in enumerate(test_loader):
          batch_m = Variable(m).to(self.device)
          batch_y = Variable(y).to(self.device)
          predict_m, mu, logvar, z = model(batch_m, batch_y)       

          BCEm, KLD,loss, accm, accmr, correct_predict= self.loss_vae(epoch, predict_m, batch_m,batch_y, model_C, mu, logvar)
          test_loss += loss.item()
          BCE_loss    += BCEm.item()
          KLD_loss    += KLD.item()
          test_accm += accm.item()
          test_accmr += accmr.item()
          test_correct_predict += correct_predict
          # if batch_idx % 50 == 0:
          #   file_name = '_epoch_'+ str(epoch)
            # m1, mr1 = shape_to_pianoroll(batch_m[0],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
            # save_image(torch.from_numpy(m1).type(torch.FloatTensor),'/home/annahung/project/anna_jam/output/train_sample_' + file_name + '.png')
            # predict_m = predict_m.cpu().detach()
            # gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0], DATA_CONFIG['bar'], DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
            # save_image(torch.from_numpy(gen_m1).type(torch.FloatTensor),'./output/train_output_' + file_name + '.png')


        test_loss /= len(test_loader.dataset)
        BCE_loss /= len(test_loader.dataset)
        KLD_loss /= len(test_loader.dataset)
        test_accm /= len(test_loader.dataset)
        test_accmr /= len(test_loader.dataset)
        test_correct_predict /= len(test_loader.dataset)

        # print('===============>    test loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, correct genre:{:.4f},:{:.4f} '.format(test_loss,           
           # BCE_loss,KLD_loss, test_accm, test_accmr,test_correct_predict))  # accuracy_melody, accuracy_melody_rhythm

        return test_loss, BCE_loss, KLD_loss, len(test_loader.dataset)










class VAETrainer():
    def __init__(self,MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device):
        super(VAETrainer, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.lr = TRAIN_CONFIG['vae']['lr_vae']
        self.lr_step1 = TRAIN_CONFIG['vae']['lr_step1']
        self.lr_step2 = TRAIN_CONFIG['vae']['lr_step2']
        self.lr_gamma = TRAIN_CONFIG['vae']['lr_gamma']
        self.loss_beta = TRAIN_CONFIG['vae']['loss_beta']
        self.MODEL_CONFIG = MODEL_CONFIG
        self.TRAIN_CONFIG = TRAIN_CONFIG
        self.DATA_CONFIG = DATA_CONFIG
    def loss_vae(self,epoch, predict_m, test_m, test_y,model_C,mu, logvar):
        # loss 
        # beta = (0.0055*epoch)*(0.0055*epoch)
        beta = self.loss_beta

        BCEm = F.binary_cross_entropy(predict_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), test_m.view(-1, self.bar*self.DATA_CONFIG['feature_size']), reduction='sum')
        # MSE = criterion(predict_m, test_m)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        predict_y_sigmoid, _, _ = model_C(predict_m)
        #### find classification problem loss 
        BCEc = F.binary_cross_entropy(predict_y_sigmoid,test_y, reduction='sum')
        predict_y2 = [1 if item>=0.5 else 0 for item in predict_y_sigmoid] 

        correct_predict = 0
        for i in range(len(predict_y2)):
            correct_predict += (predict_y2[i] == int(test_y[i].item()))
        ####

        predict_pitch_type = pitch_type(predict_m)
        data_pitch_type = pitch_type(test_m)
        pitch_type_diff = abs(data_pitch_type - predict_pitch_type)
        loss = BCEm*(1 - beta) + beta* KLD + BCEc +  pitch_type_diff
        
        # accuracy
        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
        accm = torch.sum(m==pm)/m.shape[1]

        mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
        pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
        accmr = torch.sum(mr==pmr)/mr.shape[1]
        return BCEm, KLD, loss, accm, accmr, correct_predict, pitch_type_diff

    def train_vae(self,model, model_C,epoch,train_loader):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones=[self.lr_step1,self.lr_step2], gamma= self.lr_gamma) 
        train_loss=0 ; train_accm=0; train_accmr=0; train_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx, (m,y) in enumerate(train_loader):
            batch_m = Variable(m).to(self.device)
            batch_y = Variable(y).to(self.device)
            predict_m,  mu, logvar, z = model(batch_m, batch_y)

            BCEm, KLD,loss, accm, accmr, correct_predict, pitch_type_diff = self.loss_vae(epoch, predict_m, batch_m,batch_y,model_C, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss  += loss.item()
            BCE_loss    += BCEm.item()
            KLD_loss    += KLD.item()
            train_accm  += accm.item()
            train_accmr += accmr.item()
            train_correct_predict += correct_predict

            # if batch_idx % 20 == 0:
            #     file_name = '_epoch_'+ str(epoch) +  'batch_'+ str(batch_idx)

                # m1, mr1 = shape_to_pianoroll(batch_m[0], DATA_CONFIG['bar'],   DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
                # save_image(torch.from_numpy(m1).type(torch.FloatTensor),'/home/annahung/project/anna_jam/output/train_sample_' + file_name + '.png')
                # predict_m = predict_m.cpu().detach()
                # gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
                # save_image(torch.from_numpy(gen_m1).type(torch.FloatTensor),'./output/train_output_' + file_name + '.png')


        history_train = [[] for i in range(3)]
        train_loss /= len(train_loader.dataset)
        BCE_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)
        train_accm /= len(train_loader.dataset)
        train_accmr /= len(train_loader.dataset)
        train_correct_predict /= len(train_loader.dataset)
        history_train[0] += [train_loss]
        history_train[1] += [train_accm]
        history_train[2] += [train_accmr]
        print('====> Epoch: {:3d} Average loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, correct_genre:{:.4f}, pitch_type_diff:{:.4f}'.format(epoch, 
           train_loss, BCE_loss,KLD_loss, train_accm, train_accmr,train_correct_predict , pitch_type_diff))
        return train_loss, BCE_loss, KLD_loss, len(train_loader.dataset)

    def test_vae(self,model,model_C, epoch,test_loader):
        model.eval()
        test_loss = 0; test_accm=0; test_accmr=0; test_correct_predict=0
        BCE_loss = 0 ; KLD_loss = 0
        for batch_idx,  (m,y) in enumerate(test_loader):
          batch_m = Variable(m).to(self.device)
          batch_y = Variable(y).to(self.device)
          predict_m, mu, logvar, z = model(batch_m, batch_y)       

          BCEm, KLD,loss, accm, accmr, correct_predict, pitch_type_diff= self.loss_vae(epoch, predict_m, batch_m,batch_y, model_C, mu, logvar)
          test_loss += loss.item()
          BCE_loss    += BCEm.item()
          KLD_loss    += KLD.item()
          test_accm += accm.item()
          test_accmr += accmr.item()
          test_correct_predict += correct_predict
          # if batch_idx % 50 == 0:
          #   file_name = '_epoch_'+ str(epoch)
            # m1, mr1 = shape_to_pianoroll(batch_m[0],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
            # save_image(torch.from_numpy(m1).type(torch.FloatTensor),'/home/annahung/project/anna_jam/output/train_sample_' + file_name + '.png')
            # predict_m = predict_m.cpu().detach()
            # gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0], DATA_CONFIG['bar'], DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
            # save_image(torch.from_numpy(gen_m1).type(torch.FloatTensor),'./output/train_output_' + file_name + '.png')


        test_loss /= len(test_loader.dataset)
        BCE_loss /= len(test_loader.dataset)
        KLD_loss /= len(test_loader.dataset)
        test_accm /= len(test_loader.dataset)
        test_accmr /= len(test_loader.dataset)
        test_correct_predict /= len(test_loader.dataset)

        # print('===============>    test loss: {:.4f} , BCE: {:.4f},KLD: {:.4f}, Acc: {:.4f}, {:.4f}, correct genre:{:.4f}, pitch_type_diff:{:.4f} '.format(test_loss,           
           # BCE_loss,KLD_loss, test_accm, test_accmr,test_correct_predict,pitch_type_diff))  # accuracy_melody, accuracy_melody_rhythm

        return test_loss, BCE_loss, KLD_loss, len(test_loader.dataset)
