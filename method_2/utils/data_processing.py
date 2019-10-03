import mido
import numpy as np
import torch
from torchvision.utils import save_image
from utils.midi_io.src.core_midi import midi_parser
from utils.write_midi import *
import glob
import ipdb
from utils.midi2pianoroll import get_notes, resolution_transfer, note2pianoroll
from utils.pianoroll2midi import p2m_get_notes, save2midi, pianoroll2midi
# from configs.data_config import DATA_CONFIG

def shape_to_bar_sample(m,mr,DATA_CONFIG):
    #reshape to leadsheetVAE
    #each sample reshape to (4,800)  4 bar, each_beat=200, concat 4 beat

    ts_per_bar = DATA_CONFIG['ts_per_bar']
    freq_range = DATA_CONFIG['freq_range']
    #reshape to (1,64*freq_range)
    m1 = m.reshape((1,-1), order='C')
    #reshape to (1,64)
    mr1 = mr.reshape((1,-1), order='C')
    bar_sample = []
    for i in range(DATA_CONFIG['bar']):
        st = freq_range*ts_per_bar*i
        ed = freq_range*ts_per_bar*(i+1)
        st_r = 1*ts_per_bar*i
        ed_r = 1*ts_per_bar*(i+1)
        one_bar = np.concatenate((m1[:,st:ed],mr1[:,st_r:ed_r]),axis=1)
        bar_sample.append(one_bar)
    bar_sample = np.asarray(bar_sample)
    bar_sample = bar_sample.reshape(bar_sample.shape[0],bar_sample.shape[2])
    return bar_sample

def shape_to_pianoroll(bar_sample, freq_range,ts_per_bar):
        # reshape back to pianoroll
        m = bar_sample[:,:freq_range*ts_per_bar]
        m1 = m.reshape((bar,ts_per_bar,freq_range),order='C')

        mr = bar_sample[:,freq_range*ts_per_bar:]
        mr1 = mr.reshape((bar,ts_per_bar,1),order='C')

        return m1, mr1


def PATH_TT():
    train_data_path1 = "/mnt/md0/user_annahung/leadsheet_challenge_tt/train/*/*/*/*/verse*.mid"
    train_data_path2 = "/mnt/md0/user_annahung/leadsheet_challenge_tt/train/*/*/*/*/chorus*.mid"
    train_data_path3 = "/mnt/md0/user_annahung/leadsheet_challenge_tt/train/*/*/*/*/pre*.mid"
    train_data_path4 = "/mnt/md0/user_annahung/leadsheet_challenge_tt/train/*/*/*/*/chorus*.mid"
    train_data_path5 = "/mnt/md0/user_annahung/leadsheet_challenge_tt/train/*/*/*/*/bridge*.mid"

    pieces_file_path1 = glob.glob(train_data_path1)
    pieces_file_path2 = glob.glob(train_data_path2)
    pieces_file_path3 = glob.glob(train_data_path3)
    pieces_file_path4 = glob.glob(train_data_path4)
    pieces_file_path5 = glob.glob(train_data_path5)
    pieces_file_path = pieces_file_path1 + pieces_file_path2 + pieces_file_path3 + pieces_file_path4 + pieces_file_path5
    print(len(pieces_file_path))
    return pieces_file_path



def PATH_CY():
    train_data_path = '/nas2/ai_music_database/jazz_freejammaster/split/*.mid'
    pieces_file_path = glob.glob(train_data_path)
    return pieces_file_path


def PATH_RB():
    train_data_path = '/nas2/annahung/project/data/Jazz-Data-Real-book_instrument/M/*split/*split_[0-9].mid'
    pieces_file_path = glob.glob(train_data_path)
    return pieces_file_path




def get_data(bar,
            pieces_file_path,
            save_path,
            save_filename,
            ts_per_bar,
            rest_dim,
            freq_up,
            freq_low, 
            freq_range,
            data_tpb):
    data_x = []
    for filename in pieces_file_path:
        song_name = filename.split('.')[0].split('/')[5:]
        seperator = '_'
        song_name = seperator.join(song_name)
        try:
            midi = midi_parser.MidiFile(filename)
            original_tpb = midi.ticks_per_beat

            note_group = get_notes(midi)
            where_note_group, dur_group, pitch_group = resolution_transfer(note_group, original_tpb, data_tpb, bar, ts_per_bar)
            m, mr = note2pianoroll(note_group,where_note_group,pitch_group,freq_range, freq_low, freq_up, rest_dim, bar, ts_per_bar)
            bar_sample = shape_to_bar_sample(m,mr)
            m1, mr1 = shape_to_pianoroll(bar_sample, freq_range,ts_per_bar)
            mr1 = mr1.reshape(-1,1)
            m1 = m1.reshape(-1,49)
            save_image(torch.from_numpy(m).type(torch.FloatTensor),'./output/TT_m_'+ song_name +'.png')
            save_image(torch.from_numpy(m1).type(torch.FloatTensor),'./output/TT_m1_'+ song_name +'.png')
            # pianoroll2midi(m,mr,'./output/TT_m_', song_name +'.mid')
            # pianoroll2midi(m1,mr1,'./output/TT_m1_', song_name +'.mid')
            data_x.append(bar_sample)
        except:
            continue
    data_x = np.asarray(data_x)
    print(data_x.shape, ' saved.')
    np.save(save_path + save_filename, data_x)




def get_data_CNN(bar,
            pieces_file_path,
            save_path,
            save_filename,
            ts_per_bar,
            rest_dim,
            freq_up,
            freq_low, 
            freq_range,
            data_tpb):
    data_x = []
    for filename in pieces_file_path:
        song_name = filename.split('.')[0].split('/')[5:]
        seperator = '_'
        song_name = seperator.join(song_name)
        try:
            midi = midi_parser.MidiFile(filename)
            original_tpb = midi.ticks_per_beat

            note_group = get_notes(midi)
            where_note_group, dur_group, pitch_group = resolution_transfer(note_group, original_tpb, data_tpb, bar, ts_per_bar)
            m, mr = note2pianoroll(note_group,where_note_group,pitch_group,freq_range, freq_low, freq_up, rest_dim, bar, ts_per_bar)
            m1, mr1 = shape_to_pianoroll(bar_sample, freq_range,ts_per_bar)
            mr1 = mr1.reshape(-1,1)
            m1 = m1.reshape(-1,49)
            save_image(torch.from_numpy(m1).type(torch.FloatTensor),'./output/TT_m1_'+ song_name +'.png')
            pianoroll2midi(m,mr,'./output/TT_m_', song_name +'.mid')
            pianoroll2midi(m1,mr1,'./output/TT_m1_', song_name +'.mid')
            data_x.append(m1)
        except:
            continue
    data_x = np.asarray(data_x)
    print(data_x.shape, ' saved.')
    np.save(save_path + save_filename, data_x)







if __name__ == '__main__':

    bar = 4
    ts_per_bar = 16
    rest_dim = 48
    freq_up = 95
    freq_low = 48   #48 ~ 95
    freq_range = freq_up - freq_low + 2
    data_tpb = 4
    save_path = '/nas2/annahung/project/anna_jam/data/'
    save_filename = 'cy_m.npy'

    # pieces_file_path = PATH_TT()
    pieces_file_path = PATH_CY()

    get_data(bar,pieces_file_path,save_path,save_filename,
                ts_per_bar,rest_dim,freq_up,freq_low, freq_range,data_tpb)
