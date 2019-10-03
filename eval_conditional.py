import numpy as np
import os
import torch
from torch.autograd import Variable
import glob
from matplotlib import pyplot as plt
import ipdb
import importlib
from argparse import ArgumentParser
from utils.get_data_loader import ratio_data, jazz_only_data, ratio_one_hot_data
from utils.pianoroll2midi import pianoroll2midi
from configs.config_vae_one_hot import MAIN_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
from configs.data_config import DATA_CONFIG 
from generating import generating_conditional, generating_conditional_one_hot


def generate(device,model, folder, mode, epoch, jamming_num):
    jazz_label = True
    midi_path = folder + '/jamming_one_hot/'
    if not os.path.exists(midi_path):
        os.makedirs(midi_path)
    for idx in range(jamming_num):
        m, mr = generating_conditional_one_hot(model,jazz_label)
        filename = mode + '_melody_' + str(idx) 
        pianoroll2midi(m, mr, midi_path, filename)


def choose_model(mode, folder):
    presents = glob.glob(folder + '/presents/*' + str(mode) + '*')
    assert len(presents) != 0
    presents.sort()

    present = presents[-1]
    
    return present


###choose a model by..


def get_model(dir_name,mode, en_layer, bata):
    ###
    #mode = loss, BCE, KLD
    ###

    foldernames = glob.glob(dir_name+ 'bata*')
    
    models = []
    model_names = []
    for folder in foldernames:
        #get beta, layer
        model_name = folder.split('/')[-1].split('_')
        b = model_name[1]
        l = model_name[4]
        ratio = model_name[6]
        #choose a pretrain model
        # present = choose_model(mode, folder)
        present = folder + '/presents/'+ str(mode) + '_min.pt'
        
        #get saved model
        if b == str(bata) and l == str(en_layer):
            models.append((present, b, l, ratio))
        models.sort(key=lambda x:x[3])

    return models

def draw_loss(loss_list, ratio_idx, dir_name, bata, layer, mode):
    loss_path = dir_name
    loss_list = np.asarray(loss_list)
    length = loss_list.shape[0]
    x = np.asarray(ratio_idx)
    figure_name = mode+'_'+str(bata)+'_'+str(layer)
    plt.figure()
    plt.plot(x, loss_list,label=figure_name ,linewidth=1.5)
    plt.legend(loc='upper right')
    plt.xlabel('ratio')
    plt.ylabel(mode)
    plt.savefig(loss_path  + figure_name +'.png')
    plt.close()
    np.save(loss_path  +'ratio_idx.npy', ratio_idx)
    np.save(loss_path  + figure_name +'.npy', loss_list)


###use eval dataset to get a final score (loss,BCE,KLD)




if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('-l', '--en_layer', default= '2', type=int,
                        help='encoder_layer')
    parser.add_argument('-b', '--bata', default= '0.7', type=float,
                        help='bata')
    parser.add_argument('-num', '--generating_num', default= '100', type=int,
                        help='generate how many song')

    args = parser.parse_args()



    BATCH_SIZE = 5
    epoch = 0
    bar = 4


    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    training_data = ratio_one_hot_data(sample_ratio=1)
    _,_, eval_loader = training_data.get_train_test_loader(BATCH_SIZE)




    dir_name = '/nas2/annahung/project/anna_jam_v2/paper_outputs_conditional_one_hot/'


    # modes = ['loss', 'BCE', 'KLD']
    modes = ['loss']
    for mode in modes:
        models = get_model(dir_name, mode, args.en_layer, args.bata)
        print('mode:',mode)
        model_infos = []
        for i, (model_path, bata, layer, ratio) in enumerate(models):
            TRAIN_CONFIG['vae']['loss_beta'] = float(bata)
            MODEL_CONFIG['vae']['encoder']['num_of_layer'] = int(layer)

            VAE_module = importlib.import_module(MAIN_CONFIG['model_module'] , __package__) 
            VAE =  getattr(VAE_module, MAIN_CONFIG['model'])
            model = VAE(DATA_CONFIG, MODEL_CONFIG,device)
            model.load_state_dict(torch.load(model_path))

            trainer_module = importlib.import_module(MAIN_CONFIG['trainer_module'] , __package__)
            trainer = getattr(trainer_module, MAIN_CONFIG['trainer'])
            trainer = trainer(MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device)


            classifier_module = importlib.import_module(MAIN_CONFIG['classifier_module'], __package__)
            classifier = getattr(classifier_module, MAIN_CONFIG['classifier'])
            classifier = classifier(DATA_CONFIG,MODEL_CONFIG,device)
            classifier.load_state_dict(torch.load(MAIN_CONFIG['pretrain_model_c']))
            classifier = classifier.to(device)


            model = model.to(device)
            
            model_info = [0] * 4
            model_info[0] = [bata, layer, ratio]
            model_info[1] , model_info[2] , model_info[3], test_sample_num  = trainer.test_vae(model,classifier, epoch,eval_loader)

            model_infos.append(model_info)
            folder = dir_name + model_path.split('/')[-3]
            generate(device,model,  folder, mode,epoch,jamming_num= args.generating_num)
        
        ratio_idx = [float(model_infos[i][0][2]) for i in range(len(model_infos))]
        if mode == 'loss':
            loss_list = [float(model_infos[i][1]) for i in range(len(model_infos))]

        elif mode == 'BCE':
            loss_list = [float(model_infos[i][2]) for i in range(len(model_infos))]
        else:
            loss_list = [float(model_infos[i][3]) for i in range(len(model_infos))]

        print(loss_list)
        draw_loss(loss_list, ratio_idx, dir_name, bata, layer, mode)




















