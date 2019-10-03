import numpy as np
import os
import random
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import ipdb
# import cv2
import time
import subprocess

from utils.util import *
from utils.util import slerp, one_hot
from utils.pianoroll2midi import *



class VAE_jazz_only(nn.Module):
    """Class that defines the model."""
    def __init__(self,DATA_CONFIG,MODEL_CONFIG,device):
        super(VAE_jazz_only, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.ts_per_bar = DATA_CONFIG['ts_per_bar']
        self.feature_size = DATA_CONFIG['feature_size']
        self.freq_range = DATA_CONFIG['freq_range']
        self.primary_event = self.feature_size - 1
        self.hidden_m = MODEL_CONFIG['vae']['encoder']['hidden_m']
        self.Bi = 2 if MODEL_CONFIG['vae']['encoder']['direction'] else 1
        self.Bi_de = 2 if MODEL_CONFIG['vae']['decoder']['direction'] else 1
        self.num_layers_en = MODEL_CONFIG['vae']['encoder']['num_of_layer']
        self.gru_dropout_en = MODEL_CONFIG['vae']['encoder']['gru_dropout_en']
        self.num_layers_de = MODEL_CONFIG['vae']['decoder']['num_of_layer']
        self.gru_dropout_de = MODEL_CONFIG['vae']['decoder']['gru_dropout_de']
        self.teacher_forcing_ratio = MODEL_CONFIG['vae']['decoder']['teacher_forcing_ratio']

        self.BGRUm      = nn.GRU(input_size=self.feature_size, 
                                hidden_size=self.hidden_m, num_layers=self.num_layers_en, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['encoder']['direction'],
                                # dropout=self.gru_dropout_en
                                )
        self.BGRUm2     = nn.GRU(input_size=self.hidden_m*self.Bi_de, 
                                hidden_size=self.hidden_m, 
                                num_layers=self.num_layers_de, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
                                # dropout=self.gru_dropout_de
                                )

        # self.BGRUm2     = nn.GRU(input_size=self.feature_size, 
        #                         hidden_size=self.hidden_m, 
        #                         num_layers=self.num_layers_de, 
        #                         batch_first=True, 
        #                         bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
        #                         dropout=self.gru_dropout_de
        #                         )
        


        
        self.hid2mean   = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.hid2var    = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.lat2hidm   = nn.Linear(self.hidden_m , self.hidden_m)
        
        # self.outm     = nn.Linear(self.hidden_m*self.Bi_de + self.hidden_m  , self.feature_size)
        self.outm     = nn.Linear(self.hidden_m*self.Bi_de  , self.feature_size)

    def encode(self, m):
        batch_size = m.shape[0]
        m, hn = self.BGRUm(m)  #hn:(num_layers * num_directions, batch, hidden_size):
        # h1 = hn.contiguous().view(batch_size, self.hidden_m*self.hidden_factor)
        h1 =  m.contiguous().view(batch_size,self.hidden_m*self.Bi*self.bar)
        mu = self.hid2mean(h1)
        var = self.hid2var(h1)
        return mu, var


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mu.shape).cuda()
        z = eps * std + mu
        return z

    def decode(self, z):
        melody = torch.zeros((z.shape[0], self.bar, self.feature_size))
        melody = melody.to(self.device) 

        m = self.lat2hidm(z)
        m = m.view(m.shape[0], 1, m.shape[1])

        for i in range(self.bar):
            m, _ = self.BGRUm2(m)
            out_m = self.outm(m[:,0,:])
            melody[:,i,:] = torch.sigmoid(out_m)
        return melody


    def binarize(self,test_m):
        m_binarized = torch.zeros(test_m.shape)
        m = test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49)

        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        m_idx = m.reshape(m.shape[0], self.bar, -1) 
        sample_num = test_m.shape[0]
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    pitch = m_idx[i,j,k]
                    m_binarized[i,j,49*k + pitch] = 1

        
        pmr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*self.ts_per_bar)
        pmr = pmr.cpu().detach().numpy()
       
        pm2_binarized = np.zeros_like(pmr, dtype=bool)
        pm2_binarized[pmr>0.55] = 1

        # _,pm2_binarized = cv2.threshold(pmr,0.55,1,cv2.THRESH_BINARY)

        pmr_idx = pm2_binarized.reshape(pmr.shape[0], self.bar, -1) 
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    mr = pmr_idx[i,j,k]
                    m_binarized[i,j,-16+k] = int(mr)
        return m_binarized.to(self.device)

    def forward(self, m):
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
          
          ### reparameter
        z = self.reparameterize(mu, logvar)

          # ## random sample
          # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
          
        m_sigmoid = self.decode(z)
        # m_binarized = self.binarize(m)
        return m_sigmoid, mu, logvar, z



    def generate(self,sample_num):
        p_z = Variable(torch.randn(sample_num ,self.hidden_m)).to(self.device)
        melody= self.decode(p_z)
        m_binarized = self.binarize(melody)
        predict_m = m_binarized.cpu().detach()

        gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0],self.bar,  self.freq_range,self.ts_per_bar) 
        for i in range(sample_num-1):
            gen_m, gen_mr = shape_to_pianoroll(predict_m[i+1],self.bar,  self.freq_range,self.ts_per_bar) 
            gen_m1 = np.concatenate((gen_m1,gen_m),axis=0)
            gen_mr1 = np.concatenate((gen_mr1,gen_mr),axis=0)

        return gen_m1, gen_mr1

    def interpolation(self,device, m, interp_num=5):
        ### encode
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
        ### only take mean from encoder
        z = mu
        
        a = 0
        b = 1
        z = z.cpu().detach().numpy()
        st_clip = z[a].reshape(1, z.shape[1]) # a: the start clip
        interp = np.array(slerp(z[a], z[b], interp_num))
        ed_clip = z[b].reshape(1, z.shape[1])#b: the end clip
        

        whole_piece = np.concatenate([st_clip, interp, ed_clip]) # passing clips
        whole_piece = torch.from_numpy(whole_piece).type(torch.FloatTensor)
        z = whole_piece.to(device)

        ### decode  
        m= self.decode(z)
        return m



class VAE_one_hot(nn.Module):
    """Class that defines the model."""
    def __init__(self,DATA_CONFIG,MODEL_CONFIG,device):
        super(VAE_one_hot, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.ts_per_bar = DATA_CONFIG['ts_per_bar']
        self.feature_size = DATA_CONFIG['feature_size']
        self.freq_range = DATA_CONFIG['freq_range']
        self.primary_event = self.feature_size - 1
        self.hidden_m = MODEL_CONFIG['vae']['encoder']['hidden_m']
        self.Bi = 2 if MODEL_CONFIG['vae']['encoder']['direction'] else 1
        self.Bi_de = 2 if MODEL_CONFIG['vae']['decoder']['direction'] else 1
        self.num_layers_en = MODEL_CONFIG['vae']['encoder']['num_of_layer']
        self.gru_dropout_en = MODEL_CONFIG['vae']['encoder']['gru_dropout_en']
        self.num_layers_de = MODEL_CONFIG['vae']['decoder']['num_of_layer']
        self.gru_dropout_de = MODEL_CONFIG['vae']['decoder']['gru_dropout_de']
        self.teacher_forcing_ratio = MODEL_CONFIG['vae']['decoder']['teacher_forcing_ratio']

        self.BGRUm      = nn.GRU(input_size=self.feature_size, 
                                hidden_size=self.hidden_m, num_layers=self.num_layers_en, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['encoder']['direction'],
                                # dropout=self.gru_dropout_en
                                )
        self.BGRUm2     = nn.GRU(input_size=self.hidden_m*self.Bi_de, 
                                hidden_size=self.hidden_m, 
                                num_layers=self.num_layers_de, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
                                # dropout=self.gru_dropout_de
                                )

        # self.BGRUm2     = nn.GRU(input_size=self.feature_size, 
        #                         hidden_size=self.hidden_m, 
        #                         num_layers=self.num_layers_de, 
        #                         batch_first=True, 
        #                         bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
        #                         dropout=self.gru_dropout_de
        #                         )
        


        
        self.hid2mean   = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.hid2var    = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.lat2hidm   = nn.Linear(self.hidden_m + 2 , self.hidden_m)
        
        # self.outm     = nn.Linear(self.hidden_m*self.Bi_de + self.hidden_m  , self.feature_size)
        self.outm     = nn.Linear(self.hidden_m*self.Bi_de  , self.feature_size)

    def encode(self, m):
        batch_size = m.shape[0]
        m, hn = self.BGRUm(m)  #hn:(num_layers * num_directions, batch, hidden_size):
        # h1 = hn.contiguous().view(batch_size, self.hidden_m*self.hidden_factor)
        h1 =  m.contiguous().view(batch_size,self.hidden_m*self.Bi*self.bar)
        mu = self.hid2mean(h1)
        var = self.hid2var(h1)
        return mu, var

    def reparameterize(self, mu, logvar, y):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mu.shape).cuda()
        z = eps * std + mu
        
        label_z = torch.cat((z,y),1)
        return label_z

    def decode(self, label_z):
        melody = torch.zeros((label_z.shape[0], self.bar, self.feature_size))
        melody = melody.to(self.device) 

        m = self.lat2hidm(label_z)
        m = m.view(m.shape[0], 1, m.shape[1])

        for i in range(self.bar):
            m, _ = self.BGRUm2(m)
            out_m = self.outm(m[:,0,:])
            melody[:,i,:] = torch.sigmoid(out_m)
        return melody


    def binarize(self,test_m):
        m_binarized = torch.zeros(test_m.shape)
        m = test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49)

        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        m_idx = m.reshape(m.shape[0], self.bar, -1) 
        sample_num = test_m.shape[0]
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    pitch = m_idx[i,j,k]
                    m_binarized[i,j,49*k + pitch] = 1

        
        pmr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*self.ts_per_bar)
        pmr = pmr.cpu().detach().numpy()
       
        pm2_binarized = np.zeros_like(pmr, dtype=bool)
        pm2_binarized[pmr>0.55] = 1

        # _,pm2_binarized = cv2.threshold(pmr,0.55,1,cv2.THRESH_BINARY)

        pmr_idx = pm2_binarized.reshape(pmr.shape[0], self.bar, -1) 
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    mr = pmr_idx[i,j,k]
                    m_binarized[i,j,-16+k] = int(mr)
        return m_binarized.to(self.device)

    def forward(self, m, y):
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
          
          ### reparameter
        z = self.reparameterize(mu, logvar, y)

          # ## random sample
          # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
          
        m_sigmoid = self.decode(z)
        # m_binarized = self.binarize(m)
        return m_sigmoid, mu, logvar, z



    def generate(self,sample_num, y):
        p_z = Variable(torch.randn(sample_num ,self.hidden_m)).to(self.device)
        y   = Variable(torch.from_numpy(y).type(torch.FloatTensor)).to(self.device)
        label_z = torch.cat((p_z,y), 1)
        melody= self.decode(label_z)
        m_binarized = self.binarize(melody)
        predict_m = m_binarized.cpu().detach()

        gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0],self.bar,  self.freq_range,self.ts_per_bar) 
        for i in range(sample_num-1):
            gen_m, gen_mr = shape_to_pianoroll(predict_m[i+1],self.bar,  self.freq_range,self.ts_per_bar) 
            gen_m1 = np.concatenate((gen_m1,gen_m),axis=0)
            gen_mr1 = np.concatenate((gen_mr1,gen_mr),axis=0)

        return gen_m1, gen_mr1

    def interpolation(self,device, m, y, interp_num=5):
        ### encode
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
        ### only take mean from encoder
        z = mu
        
        a = 0
        b = 1
        z = z.cpu().detach().numpy()
        st_clip = z[a].reshape(1, z.shape[1]) # a: the start clip
        interp = np.array(slerp(z[a], z[b], interp_num))
        ed_clip = z[b].reshape(1, z.shape[1])#b: the end clip
        

        whole_piece = np.concatenate([st_clip, interp, ed_clip]) # passing clips
        whole_piece = torch.from_numpy(whole_piece).type(torch.FloatTensor)
        whole_piece = torch.cat((whole_piece, y), 1)
        label_z = whole_piece.to(device)

        ### decode  
        m= self.decode(label_z)
        return m



class VAE(nn.Module):
    """Class that defines the model."""
    def __init__(self,DATA_CONFIG,MODEL_CONFIG,device):
        super(VAE, self).__init__()
        self.device = device
        self.bar = DATA_CONFIG['bar']
        self.ts_per_bar = DATA_CONFIG['ts_per_bar']
        self.feature_size = DATA_CONFIG['feature_size']
        self.freq_range = DATA_CONFIG['freq_range']
        self.primary_event = self.feature_size - 1
        self.hidden_m = MODEL_CONFIG['vae']['encoder']['hidden_m']
        self.Bi = 2 if MODEL_CONFIG['vae']['encoder']['direction'] else 1
        self.Bi_de = 2 if MODEL_CONFIG['vae']['decoder']['direction'] else 1
        self.num_layers_en = MODEL_CONFIG['vae']['encoder']['num_of_layer']
        self.gru_dropout_en = MODEL_CONFIG['vae']['encoder']['gru_dropout_en']
        self.num_layers_de = MODEL_CONFIG['vae']['decoder']['num_of_layer']
        self.gru_dropout_de = MODEL_CONFIG['vae']['decoder']['gru_dropout_de']
        self.teacher_forcing_ratio = MODEL_CONFIG['vae']['decoder']['teacher_forcing_ratio']

        self.BGRUm      = nn.GRU(input_size=self.feature_size, 
                                hidden_size=self.hidden_m, num_layers=self.num_layers_en, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['encoder']['direction'],
                                # dropout=self.gru_dropout_en
                                )
        self.BGRUm2     = nn.GRU(input_size=self.hidden_m*self.Bi_de, 
                                hidden_size=self.hidden_m, 
                                num_layers=self.num_layers_de, 
                                batch_first=True, 
                                bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
                                # dropout=self.gru_dropout_de
                                )

        # self.BGRUm2     = nn.GRU(input_size=self.feature_size, 
        #                         hidden_size=self.hidden_m, 
        #                         num_layers=self.num_layers_de, 
        #                         batch_first=True, 
        #                         bidirectional=MODEL_CONFIG['vae']['decoder']['direction'],
        #                         dropout=self.gru_dropout_de
        #                         )
        


        
        self.hid2mean   = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.hid2var    = nn.Linear(self.hidden_m*self.Bi*self.bar , self.hidden_m)
        self.lat2hidm   = nn.Linear(self.hidden_m + 1 , self.hidden_m)
        
        # self.outm     = nn.Linear(self.hidden_m*self.Bi_de + self.hidden_m  , self.feature_size)
        self.outm     = nn.Linear(self.hidden_m*self.Bi_de  , self.feature_size)

    def encode(self, m):
        batch_size = m.shape[0]
        m, hn = self.BGRUm(m)  #hn:(num_layers * num_directions, batch, hidden_size):
        # h1 = hn.contiguous().view(batch_size, self.hidden_m*self.hidden_factor)
        h1 =  m.contiguous().view(batch_size,self.hidden_m*self.Bi*self.bar)
        mu = self.hid2mean(h1)
        var = self.hid2var(h1)
        return mu, var

    def reparameterize(self, mu, logvar, y):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mu.shape).cuda()
        z = eps * std + mu

        y = y.reshape(y.shape[0],1)
        label_z = torch.cat((z,y),1)
        return label_z

    def decode(self, label_z):
        melody = torch.zeros((label_z.shape[0], self.bar, self.feature_size))
        melody = melody.to(self.device) 

        m = self.lat2hidm(label_z)
        m = m.view(m.shape[0], 1, m.shape[1])

        for i in range(self.bar):
            m, _ = self.BGRUm2(m)
            out_m = self.outm(m[:,0,:])
            melody[:,i,:] = torch.sigmoid(out_m)
        return melody


    def binarize(self,test_m):
        m_binarized = torch.zeros(test_m.shape)
        m = test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49)

        m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
        m_idx = m.reshape(m.shape[0], self.bar, -1) 
        sample_num = test_m.shape[0]
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    pitch = m_idx[i,j,k]
                    m_binarized[i,j,49*k + pitch] = 1

        
        pmr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*self.ts_per_bar)
        pmr = pmr.cpu().detach().numpy()
       
        pm2_binarized = np.zeros_like(pmr, dtype=bool)
        pm2_binarized[pmr>0.55] = 1

        # _,pm2_binarized = cv2.threshold(pmr,0.55,1,cv2.THRESH_BINARY)

        pmr_idx = pm2_binarized.reshape(pmr.shape[0], self.bar, -1) 
        for i in range(sample_num):
            for j in range(self.bar):
                for k in range(self.ts_per_bar):
                    mr = pmr_idx[i,j,k]
                    m_binarized[i,j,-16+k] = int(mr)
        return m_binarized.to(self.device)

    def forward(self, m, y):
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
          
          ### reparameter
        z = self.reparameterize(mu, logvar, y)

          # ## random sample
          # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
          
        m_sigmoid = self.decode(z)
        # m_binarized = self.binarize(m)
        return m_sigmoid, mu, logvar, z



    def generate(self,sample_num, y):
        p_z = Variable(torch.randn(sample_num ,self.hidden_m)).to(self.device)
        y   = Variable(torch.from_numpy(y).type(torch.FloatTensor)).to(self.device)
        label_z = torch.cat((p_z,y), 1)
        melody= self.decode(label_z)
        m_binarized = self.binarize(melody)
        predict_m = m_binarized.cpu().detach()

        gen_m1, gen_mr1 = shape_to_pianoroll(predict_m[0],self.bar,  self.freq_range,self.ts_per_bar) 
        for i in range(sample_num-1):
            gen_m, gen_mr = shape_to_pianoroll(predict_m[i+1],self.bar,  self.freq_range,self.ts_per_bar) 
            gen_m1 = np.concatenate((gen_m1,gen_m),axis=0)
            gen_mr1 = np.concatenate((gen_mr1,gen_mr),axis=0)

        return gen_m1, gen_mr1

    def interpolation(self,device, m, y, interp_num=5):
        ### encode
        mu, logvar = self.encode(m.view(-1, self.bar, self.feature_size))
        ### only take mean from encoder
        z = mu
        
        a = 0
        b = 1
        z = z.cpu().detach().numpy()
        st_clip = z[a].reshape(1, z.shape[1]) # a: the start clip
        interp = np.array(slerp(z[a], z[b], interp_num))
        ed_clip = z[b].reshape(1, z.shape[1])#b: the end clip
        

        whole_piece = np.concatenate([st_clip, interp, ed_clip]) # passing clips
        whole_piece = torch.from_numpy(whole_piece).type(torch.FloatTensor)
        whole_piece = torch.cat((whole_piece, y), 1)
        label_z = whole_piece.to(device)

        ### decode  
        m= self.decode(label_z)
        return m










