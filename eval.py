import numpy as np
import os
import time
import torch
from torch.autograd import Variable
import glob
from matplotlib import pyplot as plt
import ipdb
import importlib
from argparse import ArgumentParser
from utils.get_data_loader import ratio_data, jazz_only_data
# from configs.config_jazz_only_adapt import MAIN_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
from configs.config_jazz_only_no_pcd import MAIN_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
from configs.data_config import DATA_CONFIG 
from generating import generating
from utils.pianoroll2midi import pianoroll2midi


def generate(device,model, folder, mode, jamming_num):
    midi_path = folder + '/jamming_big/'
    if not os.path.exists(midi_path):
        os.makedirs(midi_path)
    print('start generating')
    for idx in range(jamming_num):
        m, mr = generating(model)
        filename = mode + '_melody_' + str(idx) 
        pianoroll2midi(m, mr, midi_path, filename)



def choose_model(mode, folder):
    presents = glob.glob(folder + '/presents/*' + str(mode) + '*')
    assert len(presents) != 0
    presents.sort()

    present = presents[-1]
    ipdb.set_trace()
    return present


###choose a model by..


def get_model(dir_name,mode, en_layer):
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
        # ratio = 0
        #choose a pretrain model
        
        # present = choose_model(mode, folder)
        # present = folder + '/presents/adapt_'+ str(mode) + '_min.pt'
        # present = folder + '/presents/tt_'+ str(mode) + '_min.pt'
        present = folder + '/presents/'+ str(mode) + '_min.pt'
        #get saved model
        if b == '0.7' and l == str(en_layer):
            models.append((present, b, l, ratio))
        models.sort(key=lambda x:x[3])

    return models


def draw_loss(loss_list, ratio_idx, dir_name, bata, layer, mode):
    loss_path = dir_name
    loss_list = np.asarray(loss_list)
    length = loss_list.shape[0]
    # x = np.linspace(0, length-1, length)
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
    parser.add_argument('-num', '--generating_num', default= '575', type=int,
                        help='generate how many song')
    parser.add_argument('-p', '--project_name', default= 'paper_outputs_adaptation_no_pcd/', type=str,
                        help='project name')


    args = parser.parse_args()


    print('evaluating project:{}, {} layer'.format(args.project_name, args.en_layer))
    time.sleep(2)


    BATCH_SIZE = 5
    epoch = 0
    bar = 4


    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    training_data = jazz_only_data()
    _,_, eval_loader = training_data.get_train_test_loader(BATCH_SIZE)




    # dir_name = '/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/'
    dir_name = '/nas2/annahung/project/anna_jam_v2/' + args.project_name


    # modes = ['loss', 'BCE', 'KLD']
    modes = ['loss']
    for mode in modes:
        models = get_model(dir_name, mode, args.en_layer)
        # present_model = '/nas2/annahung/project/anna_jam_v2/paper_outputs_adaptation_no_pcd/bata_0.7_en_layer_2_ratio_6.0/presents/tt_'+ str(mode) + '_min.pt'
        # models = [(present_model, '0.7', '2', '6.0')]
        print('mode:',mode)
        print(models)
        model_infos = []
        for i, (model_path, bata, layer, ratio) in enumerate(models):
            print(i, model_path)
            TRAIN_CONFIG['vae']['loss_beta'] = float(bata)
            MODEL_CONFIG['vae']['encoder']['num_of_layer'] = int(layer)

            VAE_module = importlib.import_module(MAIN_CONFIG['model_module'] , __package__) 
            VAE =  getattr(VAE_module, MAIN_CONFIG['model'])
            model = VAE(DATA_CONFIG, MODEL_CONFIG,device)
            model.load_state_dict(torch.load(model_path))

            trainer_module = importlib.import_module(MAIN_CONFIG['trainer_module'] , __package__)
            trainer = getattr(trainer_module, MAIN_CONFIG['trainer'])
            trainer = trainer(MODEL_CONFIG,TRAIN_CONFIG,DATA_CONFIG,device)


            model = model.to(device)
            
            model_info = [0] * 4
            model_info[0] = [bata, layer, ratio]
            model_info[1] , model_info[2] , model_info[3], test_sample_num  = trainer.test_vae(model, epoch,eval_loader)
            folder = dir_name + model_path.split('/')[-3]
            generate(device,model,  folder, mode,jamming_num=args.generating_num)
            model_infos.append(model_info)

        
        ratio_idx = [float(model_infos[i][0][2]) for i in range(len(model_infos))]
        if mode == 'loss':
            loss_list = [float(model_infos[i][1]) for i in range(len(model_infos))]

        elif mode == 'BCE':
            loss_list = [float(model_infos[i][2]) for i in range(len(model_infos))]
        else:
            loss_list = [float(model_infos[i][3]) for i in range(len(model_infos))]

        # print(loss_list)
        # draw_loss(loss_list, ratio_idx, dir_name, bata, layer, mode)





















