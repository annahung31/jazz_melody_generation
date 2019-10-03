import mido
import numpy as np
import pypianoroll as pypiano
from pypianoroll import Multitrack, Track
import ipdb

from torchvision.utils import save_image
import torch

from utils.midi_io.src.core_midi import midi_parser
from utils.write_midi import *
from configs.data_config import DATA_CONFIG


def batch_shape_to_pianoroll(batch_bar_sample, bar, freq_range,ts_per_bar):
    batch_m = torch.zeros((batch_bar_sample.shape[0],bar*ts_per_bar,freq_range))
    batch_mr = torch.zeros((batch_bar_sample.shape[0],bar*ts_per_bar,1))
    batch_bar_sample = batch_bar_sample.cpu().numpy()
    for i in range(batch_bar_sample.shape[0]):
        bar_sample = batch_bar_sample[i]
        m = bar_sample[:,:freq_range*ts_per_bar]
        m1 = m.reshape((bar*ts_per_bar,freq_range),order='C')
        batch_m[i] = torch.from_numpy(m1).type(torch.FloatTensor)

        mr = bar_sample[:,freq_range*ts_per_bar:]
        mr1 = mr.reshape((bar*ts_per_bar,1),order='C')
        batch_mr[i] = torch.from_numpy(mr1).type(torch.FloatTensor)

    return batch_m, batch_mr




def shape_to_pianoroll(bar_sample, bar, freq_range,ts_per_bar):
        # reshape back to pianoroll
        
        bar_sample = bar_sample.cpu().numpy() 
        m = bar_sample[:,:freq_range*ts_per_bar]
        m1 = m.reshape((bar,ts_per_bar,freq_range),order='C')

        mr = bar_sample[:,freq_range*ts_per_bar:]
        mr1 = mr.reshape((bar,ts_per_bar,1),order='C')

        return m1, mr1


def p2m_get_notes(m,mr,rest_dim, freq_low,ts_per_bar,bar):
    p2n_where_note = np.where(mr[:,0] == 1)[0]
    back_dur = [p2n_where_note[i] - p2n_where_note[i-1] for i in range(len(p2n_where_note))][1:] + [ts_per_bar*bar - p2n_where_note[-1]]

    back_pitch = []
    for i in range(len(p2n_where_note)):
        where_note = p2n_where_note[i]
        pitch = np.where(m[where_note,:]==1)[0][0]
        if pitch == rest_dim:
            back_pitch.append(0)
        else:
            back_pitch.append(pitch + freq_low)

    return p2n_where_note, back_dur, back_pitch

def save2midi(velocity, p2n_where_note, back_dur, back_pitch, transfer_ratio, filename,ticks_per_beat):
    midi = mido.MidiFile()
    track_list = []
    notes_stream = []

    for i in range(p2n_where_note.shape[0]):
        if back_pitch[i] != 0:
            notes_stream.append([p2n_where_note[i]*transfer_ratio , back_dur[i]*transfer_ratio, back_pitch[i], velocity])
        else:
            continue
    track_list.append({
        'note_stream': notes_stream,
        'name': 'melody',
        'is_drum': False,
        'program': 0
    })
    write_midi_notestream(track_list, ticks_per_beat,filename+'.mid')

def check_shape(m,mr):
    m = m.reshape(-1, 49)
    mr = mr.reshape(-1,1)
    return m, mr


def save_midi_pypiano(m, midi_path, filename):
    # velocity = 100

    # zero1 = np.zeros((m.shape[0],47))
    # zero2 = np.zeros((m.shape[0],128 - 95))
    # m1 = np.concatenate((zero1,m[:,:-1],zero2),axis=1)
    m1 = np.zeros((m.shape[0], 128))
    m1[:,48:96] = m[:,:-1]
    velocity_matrix = get_velocity_matrix(m1)
    m1 = m1 * velocity_matrix
    # m1 = m1 * velocity
    resolution = DATA_CONFIG['data_tpb']
    track_m = Track(pianoroll=m1, program=0, is_drum=False, name='Melody')
    multitrack = Multitrack(tracks=[track_m], tempo=80.0, beat_resolution=resolution)
    pypiano.write(multitrack, midi_path+filename+".mid")


def pianoroll2midi(m,mr,midi_path,filename):
    bar = 4
    ts_per_bar = 16
    freq_range = 49
    freq_low = 48
    rest_dim = 48
    original_tpb = 480
    velocity=84
    data_tpb = 4


    transfer_ratio = round(original_tpb/data_tpb)
    
    ticks_per_beat = original_tpb
    m, mr = check_shape(m,mr)
    # save_image(torch.from_numpy(m).type(torch.FloatTensor), midi_path + filename + '.png')
    save_midi_pypiano(m, midi_path, filename )

    # p2n_where_note, back_dur, back_pitch = p2m_get_notes(m,mr,rest_dim, freq_low,ts_per_bar,bar)
    # print('p2n_where_note:', p2n_where_note)
    # print('pitch         :', back_pitch)
    # save2midi(velocity, p2n_where_note, back_dur, back_pitch, transfer_ratio,midi_path + filename,ticks_per_beat)

def get_velocity_matrix(melody):

    velocity_matrix = np.zeros_like(melody)
    velocity_matrix = np.random.randint(30,90, size=melody.shape)

    return velocity_matrix




if __name__ == '__main__':
    bar = 4
    ts_per_bar = 16
    rest_dim = 48
    freq_up = 95
    freq_low = 48   #48 ~ 95
    freq_range = freq_up - freq_low + 2
    data_tpb = 4
    data = np.load('/home/annahung/project/anna_jam/data/cy_m.npy')
    bar_sample = data[0]
    print(bar_sample.shape)
    m, mr = shape_to_pianoroll(bar_sample, bar, freq_range,ts_per_bar)
    print(m.shape)
    midi_path = '/home/annahung/project/anna_jam_v2/outputs/'
    filename = 'sample_cy.mid'
    pianoroll2midi(m, mr, midi_path, filename)







