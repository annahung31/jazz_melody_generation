import numpy as np
import random
import torch
import torch.utils.data as Data
import ipdb

from utils.pianoroll2midi import *





class get_dataloader(object):
    def __init__(self, data, y):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.y   = torch.from_numpy(y).type(torch.FloatTensor)

         # self.label = np.array(label)
    def __getitem__(self, index):
        return self.data[index], self.y[index]

    def __len__(self):
        return self.size



class get_dataloader_no_label(object):
    def __init__(self, data):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).type(torch.FloatTensor)

         # self.label = np.array(label)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size



class data(object):
    def get_data(self,data_path,label):
        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        if label == 0:
            train_y = np.zeros(train_m.shape[0])
            test_y = np.zeros(test_m.shape[0])
        else:
            train_y = np.ones(train_m.shape[0])
            test_y = np.ones(test_m.shape[0])
        return train_m, test_m, train_y, test_y

    def parse_data(self):
        ratio = DATA_CONFIG['testset_ratio']
        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'
        tt_train_m, tt_test_m, tt_train_y, tt_test_y = self.get_data(data_path_tt,0)
        cy_train_m, cy_test_m, cy_train_y, cy_test_y = self.get_data(data_path_cy,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)


        train_m = np.concatenate((tt_train_m,cy_train_m,rb_train_m),axis=0)
        test_m = np.concatenate((tt_test_m,cy_test_m,rb_test_m),axis=0)

        train_y = np.concatenate((tt_train_y, cy_train_y, rb_train_y),axis=0)
        test_y = np.concatenate((tt_test_y, cy_test_y, rb_test_y),axis=0)


        # train_m = torch.from_numpy(train_m).type(torch.FloatTensor)
        # test_m = torch.from_numpy(test_m).type(torch.FloatTensor)
        # train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
        # test_y = torch.from_numpy(test_y).type(torch.FloatTensor)
        return train_m, test_m, train_y, test_y

    def get_train_test_loader(self):
        train_m, test_m, train_y, test_y = self.parse_data()
        train_iter = get_dataloader(train_m,train_y)
        test_iter = get_dataloader(test_m,test_y)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = 256,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = 256,
            shuffle=True,
            num_workers=1
        )
        return train_loader, test_loader


class paprallel_data(object):
    def get_data(self,data_path,label):

        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        if label == 0:
            train_y = np.zeros(train_m.shape[0])
            test_y = np.zeros(test_m.shape[0])
        else:
            train_y = np.ones(train_m.shape[0])
            test_y = np.ones(test_m.shape[0])
        return train_m, test_m, train_y, test_y

    def parse_data_c(self):
        
        ratio = DATA_CONFIG['testset_ratio']
        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'

        tt_train_m, tt_test_m, tt_train_y, tt_test_y = self.get_data(data_path_tt,0)
        cy_train_m, cy_test_m, cy_train_y, cy_test_y = self.get_data(data_path_cy,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)


        tt_train_m = np.asarray(random.sample(list(tt_train_m[10:]), cy_train_m.shape[0] + rb_train_m.shape[0]))
        tt_test_m = np.asarray(random.sample(list(tt_test_m), cy_test_m.shape[0] + rb_test_m.shape[0]))
        tt_train_y = tt_train_y[:tt_train_m.shape[0]]
        tt_test_y =  tt_test_y[:tt_test_m.shape[0]]

        sample_num = int(1446)
        test_sample_num = int(sample_num * 0.1)

        train_m = np.concatenate((tt_train_m[:sample_num],cy_train_m[:sample_num],rb_train_m[:sample_num]),axis=0)
        test_m = np.concatenate((tt_test_m[:test_sample_num],cy_test_m[:test_sample_num],rb_test_m[:test_sample_num]),axis=0)
        train_y = np.concatenate((tt_train_y[:sample_num], cy_train_y[:sample_num], rb_train_y[:sample_num]),axis=0)
        test_y = np.concatenate((tt_test_y[:test_sample_num], cy_test_y[:test_sample_num], rb_test_y[:test_sample_num]),axis=0)

        eval_m = np.concatenate((tt_train_m[:10],cy_train_m[:10], rb_train_m[:10]), axis=0)
        print('train_m shape:{}, test_m shape:{}'.format(train_m.shape, test_m.shape))
        return train_m, test_m, train_y, test_y, eval_m


    def get_train_test_loader(self, batch_size):
        train_m, test_m, train_y, test_y, eval_m = self.parse_data_c()
        train_iter = get_dataloader(train_m,train_y)
        test_iter = get_dataloader(test_m,test_y)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )
        return train_loader, test_loader


class jazz_only_data(object):
    def get_data(self,data_path):

        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        return train_m, test_m

    def parse_data(self):
        
        ratio = DATA_CONFIG['testset_ratio']
        
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'

        cy_train_m, cy_test_m = self.get_data(data_path_cy)
        rb_train_m, rb_test_m = self.get_data(data_path_rb)


        train_m = np.concatenate((cy_train_m[10:],rb_train_m[10:]),axis=0)
        test_m = np.concatenate((cy_test_m,rb_test_m),axis=0)

        eval_m = np.concatenate((cy_train_m[:10], rb_train_m[:10]), axis=0)
        print('train_m shape:{}, test_m shape:{}'.format(train_m.shape, test_m.shape))
        return train_m, test_m, eval_m


    def get_train_test_loader(self, batch_size):
        train_m, test_m, eval_m = self.parse_data()
        train_iter = get_dataloader_no_label(train_m)
        test_iter = get_dataloader_no_label(test_m)
        eval_iter = get_dataloader_no_label(eval_m)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )

        eval_loader = Data.DataLoader(
            dataset = eval_iter,
            batch_size = batch_size,
            shuffle=False,
            num_workers=1
        )
        return train_loader, test_loader, eval_loader





class tt_only_data(object):
    def __init__(self, sample_ratio):
        self.sample_ratio = sample_ratio
    def get_data(self,data_path):

        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        return train_m, test_m

    def parse_data(self):
        
        ratio = DATA_CONFIG['testset_ratio']

        sample_num = int(self.sample_ratio * 1446)
        test_sample_num = int(sample_num * 0.1)

        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        tt_train_m, tt_test_m = self.get_data(data_path_tt)

        tt_train_m = np.asarray(random.sample(list(tt_train_m[10:]), sample_num))
        tt_test_m = np.asarray(random.sample(list(tt_test_m[10:]), test_sample_num))


        return tt_train_m, tt_test_m


    def get_train_test_loader(self, batch_size):
        train_m, test_m = self.parse_data()
        train_iter = get_dataloader_no_label(train_m)
        test_iter = get_dataloader_no_label(test_m)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=1
        )
        return train_loader, test_loader



class ratio_one_hot_data(object):
    def __init__(self, sample_ratio):
        self.sample_ratio = sample_ratio

    def get_data(self,data_path,label):

        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        train_y = np.zeros((train_m.shape[0],2))
        test_y = np.zeros((test_m.shape[0],2))
        if label == 0:
            train_y[:,0] = 1
            test_y[:,0]  = 1
        else:
            train_y[:,1] = 1
            test_y[:,1]  = 1
        return train_m, test_m, train_y, test_y

    def parse_data(self):
        
        ratio = DATA_CONFIG['testset_ratio']
        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'

        tt_train_m, tt_test_m, tt_train_y, tt_test_y = self.get_data(data_path_tt,0)
        cy_train_m, cy_test_m, cy_train_y, cy_test_y = self.get_data(data_path_cy,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)


        
        sample_num = int(self.sample_ratio * 1446)
        test_sample_num = int(sample_num * 0.1)

        tt_train_m = np.asarray(random.sample(list(tt_train_m[10:]), sample_num))
        tt_test_m = np.asarray(random.sample(list(tt_test_m[10:]), test_sample_num))


        train_m = np.concatenate((tt_train_m,cy_train_m[10:],rb_train_m[10:]),axis=0)
        test_m = np.concatenate((tt_test_m,cy_test_m,rb_test_m),axis=0)

        train_y = np.concatenate((tt_train_y[:sample_num], cy_train_y[10:], rb_train_y[10:]),axis=0)
        test_y = np.concatenate((tt_test_y, cy_test_y, rb_test_y),axis=0)


        eval_m = np.concatenate((cy_train_m[:10], rb_train_m[:10]), axis=0)
        eval_y = np.concatenate((cy_train_y[:10], rb_train_y[:10]),axis=0)
        print('train_m shape:{}, test_m shape:{}'.format(train_m.shape, test_m.shape))
        print('train_y shape:{}, test_y shape:{}'.format(train_y.shape, test_y.shape))
        return train_m, test_m, train_y, test_y, eval_m, eval_y


    def get_train_test_loader(self, batch_size):
        train_m, test_m, train_y, test_y, eval_m, eval_y = self.parse_data()
        train_iter = get_dataloader(train_m, train_y)
        test_iter = get_dataloader(test_m, test_y)
        eval_iter = get_dataloader(eval_m, eval_y)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0
        )

        eval_loader = Data.DataLoader(
            dataset = eval_iter,
            batch_size = batch_size,
            shuffle=False,
            num_workers=0
        )
        return train_loader, test_loader, eval_loader




class ratio_data(object):
    def __init__(self, sample_ratio):
        self.sample_ratio = sample_ratio

    def get_data(self,data_path,label):

        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        if label == 0:
            train_y = np.zeros(train_m.shape[0])
            test_y = np.zeros(test_m.shape[0])
        else:
            train_y = np.ones(train_m.shape[0])
            test_y = np.ones(test_m.shape[0])
        return train_m, test_m, train_y, test_y

    def parse_data(self):
        
        ratio = DATA_CONFIG['testset_ratio']
        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'

        tt_train_m, tt_test_m, tt_train_y, tt_test_y = self.get_data(data_path_tt,0)
        cy_train_m, cy_test_m, cy_train_y, cy_test_y = self.get_data(data_path_cy,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)


        
        sample_num = int(self.sample_ratio * 1446)
        test_sample_num = int(sample_num * 0.1)

        tt_train_m = np.asarray(random.sample(list(tt_train_m[10:]), sample_num))
        tt_test_m = np.asarray(random.sample(list(tt_test_m[10:]), test_sample_num))


        train_m = np.concatenate((tt_train_m,cy_train_m[10:],rb_train_m[10:]),axis=0)
        test_m = np.concatenate((tt_test_m,cy_test_m,rb_test_m),axis=0)

        train_y = np.concatenate((tt_train_y[:sample_num], cy_train_y[10:], rb_train_y[10:]),axis=0)
        test_y = np.concatenate((tt_test_y, cy_test_y, rb_test_y),axis=0)


        eval_m = np.concatenate((cy_train_m[:10], rb_train_m[:10]), axis=0)
        eval_y = np.concatenate((cy_train_y[:10], rb_train_y[:10]),axis=0)
        print('train_m shape:{}, test_m shape:{}'.format(train_m.shape, test_m.shape))
        return train_m, test_m, train_y, test_y, eval_m, eval_y


    def get_train_test_loader(self, batch_size):
        train_m, test_m, train_y, test_y, eval_m, eval_y = self.parse_data()
        train_iter = get_dataloader(train_m, train_y)
        test_iter = get_dataloader(test_m, test_y)
        eval_iter = get_dataloader(eval_m, eval_y)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = batch_size,
            shuffle=True,
            num_workers=0
        )

        eval_loader = Data.DataLoader(
            dataset = eval_iter,
            batch_size = batch_size,
            shuffle=False,
            num_workers=0
        )
        return train_loader, test_loader, eval_loader












class audio_data(object):
    def get_data(self,data_path,label):
        ratio = DATA_CONFIG['testset_ratio']
        m = np.load(data_path).astype(int)
        T = int(m.shape[0]*ratio)
        train_m = m[:-T]
        test_m = m[-T:]
        if label == 0:
            train_y = np.zeros(train_m.shape[0])
            test_y = np.zeros(test_m.shape[0])
        else:
            train_y = np.ones(train_m.shape[0])
            test_y = np.ones(test_m.shape[0])
        return train_m, test_m, train_y, test_y

    def parse_data_c(self):
        ratio = DATA_CONFIG['testset_ratio']
        data_path_tt = '/nas2/annahung/project/anna_jam/data/TT_non_intro_m.npy'
        data_path_cy = '/nas2/annahung/project/anna_jam/data/cy_m.npy'
        data_path_rb = '/nas2/annahung/project/anna_jam/data/rb_m.npy'
        data_path_au = '/nas2/annahung/project/anna_jam/data/audio_m.npy'

        tt_train_m, tt_test_m, tt_train_y, tt_test_y = self.get_data(data_path_tt,0)
        cy_train_m, cy_test_m, cy_train_y, cy_test_y = self.get_data(data_path_cy,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)
        rb_train_m, rb_test_m, rb_train_y, rb_test_y = self.get_data(data_path_rb,1)

        tt_train_m = np.asarray(random.sample(list(tt_train_m[10:]), cy_train_m[10:].shape[0] + rb_train_m[10:].shape[0]))
        tt_test_m = np.asarray(random.sample(list(tt_test_m), cy_test_m.shape[0] + rb_test_m.shape[0]))
        tt_train_y = tt_train_y[:tt_train_m.shape[0]]
        tt_test_y =  tt_test_y[:tt_test_m.shape[0]]

        train_m = np.concatenate((tt_train_m,cy_train_m,rb_train_m),axis=0)
        test_m = np.concatenate((tt_test_m,cy_test_m,rb_test_m),axis=0)
        train_y = np.concatenate((tt_train_y, cy_train_y, rb_train_y),axis=0)
        test_y = np.concatenate((tt_test_y, cy_test_y, rb_test_y),axis=0)

        eval_m = np.concatenate((tt_train_m[:10],cy_train_m[:10], rb_train_m[:10]), axis=0)

        return train_m, test_m, train_y, test_y, eval_m


    def get_train_test_loader(self):
        train_m, test_m, train_y, test_y = self.parse_data_c()
        train_iter = get_dataloader(train_m,train_y)
        test_iter = get_dataloader(test_m,test_y)
        train_loader = Data.DataLoader(
            dataset = train_iter,
            batch_size = 256,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset = test_iter,
            batch_size = 256,
            shuffle=True,
            num_workers=1
        )
        return train_loader, test_loader








training_data = data()

train_loader, test_loader = training_data.get_train_test_loader()