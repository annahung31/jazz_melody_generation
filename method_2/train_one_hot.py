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
import json
from argparse import ArgumentParser

from utils.get_data_loader import ratio_data, ratio_one_hot_data
from utils.pianoroll2midi import *
from utils.midi2pianoroll import midi2pianoroll
from utils.data_processing import shape_to_bar_sample
from utils.util import write_json, torch_summarize
from generating import generating_while_train


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


def reconstruct(device,DATA_CONFIG,model,dir_name,epoch):
    for piece in range(4):
        filename =  '/nas2/ai_music_database/jazz_freejammaster/split/AI_Jazz_freejammaster_01_split_'+str(piece+2)+'.mid'
        # filename = '/nas2/annahung/project/happy.mid'
        m,mr = midi2pianoroll(filename)
        bar_sample = shape_to_bar_sample(m,mr,DATA_CONFIG)
        bar_sample = bar_sample.reshape(1,bar_sample.shape[0],bar_sample.shape[1])
        batch_m = torch.from_numpy(bar_sample).type(torch.FloatTensor)
        batch_m = Variable(batch_m).to(device)
        
        y = np.zeros((1,2))
        y[:,1] = 1
        batch_y = torch.from_numpy(y).type(torch.FloatTensor)
        batch_y = Variable(batch_y).to(device)

        BATCH_SIZE = 1
        bar = DATA_CONFIG['bar']

        predict_m,  mu, logvar, z = model(batch_m, batch_y)
        m_binarized = model.binarize(predict_m)
        m_binarized = m_binarized.cpu().detach()

        gen_m1, gen_mr1 = shape_to_pianoroll(m_binarized[0],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
        midi_path = dir_name + '/reconstruction/'
        # filename = 'test451_AI_Jazz_freejammaster_01_split_'+str(piece)
        filename = 'AI_Jazz_freejammaster_01_split_'+str(piece) + '_ep_' + str(epoch)

        pianoroll2midi(gen_m1, gen_mr1, midi_path, filename)


def generate(device,model, dir_name,epoch,sample_num=5):
    jazz_label = True
    midi_path = dir_name + '/jamming/'

    for idx in range(sample_num):
        m, mr = generating_while_train(device, model,jazz_label, dir_name)
        filename ='ep_'+ str(epoch) + '_melody_' + str(idx) 
        pianoroll2midi(m, mr, midi_path, filename)


def interpolate(device, epoch,DATA_CONFIG,model, dir_name, interp_num=5):
    
    piece = 4
    # filename = '/nas2/annahung/project/happy.mid'
    filename =  '/nas2/ai_music_database/jazz_freejammaster/split/AI_Jazz_freejammaster_01_split_'+str(2)+'.mid'
    filename2 = '/nas2/ai_music_database/jazz_freejammaster/split/AI_Jazz_freejammaster_01_split_'+str(8)+'.mid'
    # filename2 = '/nas2/annahung/project/data/TT/Pop/c/carly-rae-jepsen/call-me-maybe/chorus_symbol_nokey_melody.mid'
    # filename = '/nas2/annahung/project/data/TT/Unlabeled/m/mayday/youre-not-truly-happy/intro_symbol_nokey_melody.mid'
    m,mr   = midi2pianoroll(filename)
    m2,mr2 = midi2pianoroll(filename2)

    bar_sample1 = shape_to_bar_sample(m,mr,DATA_CONFIG)
    bar_sample2 = shape_to_bar_sample(m2,mr2,DATA_CONFIG)
    batch_m  = np.concatenate((np.expand_dims(bar_sample1, axis=0), np.expand_dims(bar_sample2, axis=0)))
    
    batch_m = torch.from_numpy(batch_m).type(torch.FloatTensor)
    batch_m = Variable(batch_m).to(device)

    y = np.zeros((interp_num+2,2))
    y[:,1] = 1
    batch_y = torch.from_numpy(batch_y).type(torch.FloatTensor)


    BATCH_SIZE = 2
    bar = DATA_CONFIG['bar']

    predict_m = model.interpolation(device, batch_m, batch_y, interp_num)
    m_binarized = model.binarize(predict_m)
    m_binarized = m_binarized.cpu().detach()

    for i in range(m_binarized.shape[0]):
        gen_m1, gen_mr1 = shape_to_pianoroll(m_binarized[i],DATA_CONFIG['bar'],  DATA_CONFIG['freq_range'],DATA_CONFIG['ts_per_bar']) 
        midi_path = dir_name + '/interpolation/'
        # filename = 'test451_AI_Jazz_freejammaster_01_split_'+str(piece)
        filename = 'AI_Jazz_freejammaster_01_split_1to7'+ '_ep_' + str(epoch) +'_'+str(i) 
        pianoroll2midi(gen_m1, gen_mr1, midi_path, filename)
    print('interpolation save to', midi_path)

def train(model,model_c,trainer,dir_name, sample_ratio, is_reconstruction, is_generating, is_interpolation):

    '''
    import data
    '''

    BATCH_SIZE = TRAIN_CONFIG['batch_size']
    epochs = TRAIN_CONFIG['epochs']
    bar = DATA_CONFIG['bar']

    training_data = ratio_one_hot_data(sample_ratio)
    train_loader, test_loader, _ = training_data.get_train_test_loader(BATCH_SIZE)
    min_loss = 1000000
    min_BCE  = 1000000
    min_KLD  = 1000000


    model = model.to(device)
    model_c = model_c.to(device)



    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_list = [[] for i in range(3)]
    test_list = [[] for i in range(3)]
    for epoch in range(epochs):
        train_loss, train_BCE_loss, train_KLD_loss, train_sample_num = trainer.train_vae(model,model_c,epoch,train_loader)
        test_loss , test_BCE_loss , test_KLD_loss, test_sample_num  = trainer.test_vae(model,model_c,epoch,test_loader)
        train_list[0].append(train_loss)
        train_list[1].append(train_BCE_loss)
        train_list[2].append(train_KLD_loss)
        test_list[0].append(test_loss)
        test_list[1].append(test_BCE_loss)
        test_list[2].append(test_KLD_loss)



        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), dir_name+'/presents/loss_min.pt')
        if test_BCE_loss < min_BCE:
            min_BCE = test_BCE_loss
            torch.save(model.state_dict(), dir_name+'/presents/BCE_min.pt')
        if test_KLD_loss < min_KLD:
            min_KLD = test_KLD_loss
            torch.save(model.state_dict(), dir_name+'/presents/KLD_min.pt')







        if (epoch+1) % 20 == 0 or (epoch+1) == epochs:
            
            if is_reconstruction:
                reconstruct(device,DATA_CONFIG,model,dir_name,epoch)
            if is_generating:
                generate(device,model, dir_name,epoch,sample_num=5)
            if is_interpolation:
                interpolate(device, epoch, DATA_CONFIG, model, dir_name, interp_num=5)

        draw_loss(train_list[0], test_list[0], dir_name,'loss')
        draw_loss(train_list[1], test_list[1], dir_name, 'BCE')
        draw_loss(train_list[2], test_list[2], dir_name, 'KLD')



    with open(dir_name +"/sample_num.py", "w") as text_file:
        text_file.write('train_sample_num:')
        text_file.write(str(train_sample_num))
        text_file.write('\ntest_sample_num:')
        text_file.write(str(test_sample_num))


    loss_label = [['loss_', 'BCE_', 'KLD_'], ['train', 'test']]
    for i, train_l in  enumerate(train_list):
        loss_path = dir_name + '/loss/'
        train_loss_list = np.asarray(train_l)
        np.save(loss_path + loss_label[0][i]+loss_label[1][0]+'.npy', train_loss_list)
              

    for i, test_l in  enumerate(test_list):
        loss_path = dir_name + '/loss/'
        train_loss_list = np.asarray(test_l)
        np.save(loss_path + loss_label[0][i]+loss_label[1][1]+'.npy', train_loss_list)  








if __name__ == '__main__':

    parser = ArgumentParser()
    
    # args = argparse.ArgumentParser(description='train')
    parser.add_argument('-c', '--config', default= 'configs.config_vae_one_hot', type=str,
                        help='config file path')
    parser.add_argument('-data', '--data_config', default= 'configs.data_config', type=str,
                        help='data config file path')
    parser.add_argument('-ro', '--sample_ratio', default= '1', type=float,
                        help='Non-jazz/jazz data number ratio')
    parser.add_argument('-b', '--loss_beta', default= '0.7', type=float,
                        help='loss_beta')
    parser.add_argument('-enl', '--num_of_layer', default= '2', type=float,
                        help='num_of_layer of encoder')


    parser.add_argument('-d', '--device', default='5', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('-r', '--reconstruction', default=False, type=bool,
                        help='check the reconstruction while training')
    parser.add_argument('-g', '--generating', default=False, type=bool,
                        help='check the generation while training')
    parser.add_argument('-i', '--interpolation', default=False, type=bool,
                        help='check the interpolation while training')
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

    TRAIN_CONFIG['vae']['loss_beta'] = float(args.loss_beta)
    MODEL_CONFIG['vae']['encoder']['num_of_layer'] = int(args.num_of_layer)



    '''
    import module
    '''
    
    #assign a model
    VAE_module = importlib.import_module(MAIN_CONFIG['model_module'] , __package__) 
    VAE =  getattr(VAE_module, MAIN_CONFIG['model'])
    model = VAE(DATA_CONFIG, MODEL_CONFIG,device)
    # model.load_state_dict(torch.load(MAIN_CONFIG['pertrain_model']))

    classifier_module = importlib.import_module(MAIN_CONFIG['classifier_module'], __package__)
    classifier = getattr(classifier_module, MAIN_CONFIG['classifier'])
    classifier = classifier(DATA_CONFIG,MODEL_CONFIG,device)
    classifier.load_state_dict(torch.load(MAIN_CONFIG['pretrain_model_c']))


    #assign a trainer
    trainer_module = importlib.import_module(MAIN_CONFIG['trainer_module'] , __package__)
    trainer = getattr(trainer_module, MAIN_CONFIG['trainer'])
    trainer = trainer(MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device)


    #assign a folder to save (timestemp) 
    foldername = 'bata_' + str(TRAIN_CONFIG['vae']['loss_beta']) + '_en_layer_' + str(MODEL_CONFIG['vae']['encoder']['num_of_layer']) +'_ratio_' + str(args.sample_ratio)
    dir_name = 'paper_outputs_conditional_one_hot/'+ foldername

    # dir_name = 'outputs/'+ time.strftime("%Y_%m_%d__%H_%M", time.localtime())
    print('output saved to', dir_name)
    subfolder_names = ['presents','reconstruction','jamming','interpolation','loss']
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
        text_file.write(torch_summarize(model))
        text_file.write(torch_summarize(classifier))

    with open(dir_name +"/notes.py", "w") as text_file:
        text_file.write('')


    train(model,classifier,trainer,dir_name, args.sample_ratio, args.reconstruction, args.generating, args.interpolation)






