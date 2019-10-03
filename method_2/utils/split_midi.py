import mido
import numpy as np
from utils.midi_io.src.core_midi import midi_parser
from utils.write_midi import *

def split(filename,
          split_file_path,
          bar=4):

    song_name = filename.split('/')[-2] + filename.split('/')[-1].split('.')[0]
    midi_ori = midi_parser.MidiFile(filename)

    beats_per_bar = 4
    original_tpb = midi_ori.ticks_per_beat
    length = original_tpb * beats_per_bar * bar

    for tidx, track in enumerate(midi_ori.instruments):
        notes_stream = []
        i = 0
        for note in track.notes:
            st = length*i
            ed = length*(i+1)        
            track_name = song_name + '_split_'+str(i)
            if st <= note.start <= ed and st <= note.end <= ed:
                notes_stream.append([note.start-st, note.end-note.start, note.pitch, note.velocity])
            elif note.start < st and st < note.end < ed :
                notes_stream.append([st, note.end-note.start, note.pitch, note.velocity])
            elif st < note.start < ed and ed < note.end:
                notes_stream.append([note.start-st, ed-note.start, note.pitch, note.velocity])
                track_list = []
                track_list.append({
                    'note_stream': notes_stream,
                    'name': 'Melody',
                    'is_drum': track.is_drum,
                    'program': track.program
                })
                ticks_per_beat=original_tpb
                filename = split_file_path + track_name + '.mid'
                write_midi_notestream(track_list, ticks_per_beat,filename)
                notes_stream = []
                i += 1
     
            else:     
                track_list = []
                track_list.append({
                    'note_stream': notes_stream,
                    'name': 'Melody',
                    'is_drum': track.is_drum,
                    'program': track.program
                })
                ticks_per_beat=original_tpb
                filename = split_file_path + track_name + '.mid'
                write_midi_notestream(track_list, ticks_per_beat,filename)
                
                notes_stream = []
                i += 1
                st = length*i
                ed = length*(i+1)  
                notes_stream.append([note.start-st, note.end-note.start, note.pitch, note.velocity])





if __name__ == '__main__':
    filename = '/home/annahung/project/data/TT/Pop/a/adele/hello/chorus_symbol_nokey_melody.mid'
    split_file_path = '/home/annahung/project/data/TT/split/'
    bar = 4
    split(filename, split_file_path, bar)


