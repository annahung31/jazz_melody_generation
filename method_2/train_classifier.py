import numpy as np
import os
import random
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from matplotlib import pyplot as plt
import ipdb
import time
import importlib
from argparse import ArgumentParser

from utils.get_data_loader import ratio_one_hot_data, paprallel_data
from utils.util import torch_summarize


def draw_loss(train_loss_list, test_loss_list, dir_name, label):
    loss_path = dir_name + '/loss/'
    train_loss_list = np.asarray(train_loss_list)
    test_loss_list  = np.asarray(test_loss_list)
    length = train_loss_list.shape[0]

    x = np.linspace(0, length-1, length)
    x = np.asarray(x)

    plt.figure()
    plt.plot(x, train_loss_list,label=label + '_train',linewidth=1.5)
    plt.plot(x, test_loss_list,label=label+'_test',linewidth=1.5)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(loss_path  + label +'.png')
    plt.close()

def train(model_c,trainer,dir_name, sample_ratio):

    '''
    import data
    '''

    BATCH_SIZE = TRAIN_CONFIG['batch_size']
    epochs = TRAIN_CONFIG['epochs']
    bar = DATA_CONFIG['bar']

    training_data = ratio_one_hot_data(sample_ratio)
    train_loader, test_loader, _ = training_data.get_train_test_loader(BATCH_SIZE)

    # training_data = paprallel_data()
    # train_loader, test_loader = training_data.get_train_test_loader(BATCH_SIZE)


    min_loss = 1000000

    model_c = model_c.to(device)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_list = []
    test_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_classifier(model_c,epoch,train_loader)
        test_loss , test_acc  = trainer.test_classifier(model_c,epoch,test_loader)
        train_list.append(train_loss)
        test_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model_c.state_dict(), dir_name+'/presents/loss_min.pt')

        draw_loss(train_list, test_list, dir_name,'loss')
        draw_loss(train_acc_list, test_acc_list, dir_name,'accuracy')





    loss_path = dir_name + '/loss/'
    np.save(loss_path +'train_loss.npy', np.asarray(train_list))
    np.save(loss_path + 'test_loss.npy', np.asarray(test_list))  




if __name__ == '__main__':

    parser = ArgumentParser()
    
    # args = argparse.ArgumentParser(description='train')
    parser.add_argument('-c', '--config', default= 'configs.config_classifier', type=str,
                        help='config file path')
    parser.add_argument('-data', '--data_config', default= 'configs.data_config', type=str,
                        help='data config file path')
    parser.add_argument('-ro', '--sample_ratio', default= '1', type=float,
                        help='Non-jazz/jazz data number ratio')
    parser.add_argument('-d', '--device', default='5', type=str,
                        help='indices of GPUs to enable')

    args = parser.parse_args()
   


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_config_name = args.config.split('.')[-1]



    '''
    import config # TODO: autochange
    '''
    config_module = importlib.import_module(args.data_config, __package__) 
    DATA_CONFIG = getattr(config_module, 'DATA_CONFIG')


    config_module = importlib.import_module(args.config, __package__)
    MAIN_CONFIG = getattr(config_module, 'MAIN_CONFIG')
    MODEL_CONFIG = getattr(config_module, 'MODEL_CONFIG')
    TRAIN_CONFIG = getattr(config_module, 'TRAIN_CONFIG')



    '''
    import module
    '''
    
    #assign a model


    classifier_module = importlib.import_module(MAIN_CONFIG['classifier_module'], __package__)
    classifier = getattr(classifier_module, MAIN_CONFIG['classifier'])
    classifier = classifier(DATA_CONFIG,MODEL_CONFIG,device)



    #assign a trainer
    trainer_module = importlib.import_module(MAIN_CONFIG['trainer_module'] , __package__)
    trainer = getattr(trainer_module, MAIN_CONFIG['trainer'])
    trainer = trainer(TRAIN_CONFIG, device)


    #assign a folder to save (timestemp) 
    # foldername = '_en_layer_' + str(MODEL_CONFIG['classifier']['num_layers_en']) +'_ratio_' + str(args.sample_ratio)
    # dir_name = 'classifier/'+ foldername

    dir_name = 'classifier/'+ time.strftime("%Y_%m_%d__%H_%M", time.localtime())
    print('output saved to', dir_name)
    subfolder_names = ['presents','loss']
    for subfolder_name in subfolder_names:
        if not os.path.exists(os.path.join(dir_name, subfolder_name)):
            os.makedirs(os.path.join(dir_name, subfolder_name))


    #save config to the folder
    with open(dir_name + '/data_config.py', 'w') as outfile: 
        outfile.write('DATA_CONFIG=' + str(DATA_CONFIG))


    with open(dir_name+ '/model_config.py', 'w') as outfile:  
        outfile.write('MAIN_CONFIG=' + str(MAIN_CONFIG))
        outfile.write('\n')
        outfile.write('MODEL_CONFIG=' + str(MODEL_CONFIG))
        outfile.write('\n')
        outfile.write('TRAIN_CONFIG=' + str(TRAIN_CONFIG))


    #write model structure and trainer detail to txt
    with open(dir_name +"/model_structure.py", "w") as text_file:
        text_file.write(torch_summarize(classifier))

    with open(dir_name +"/notes.py", "w") as text_file:
        text_file.write('')


    train(classifier,trainer,dir_name, args.sample_ratio)






