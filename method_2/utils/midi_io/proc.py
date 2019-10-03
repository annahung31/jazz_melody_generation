from midi_io.src.core_midi import midi_parser
import json
from argparse import ArgumentParser

def proc(midi_file):
    midi = midi_parser.MidiFile(midi_file)
    ticks_per_beat = midi.ticks_per_beat_ori
    
    proc_instrs = []
    for instr in midi.instruments:
        n_proc = []
        for n in instr.notes:
            duration = n.end - n.start
            n_proc.append({
                'pitch': int(n.pitch),
                'start': int(n.start),
                'duration': int(duration),        
            })

        track_info = {
            'name': instr.name,
            'program': int(instr.program),
            'is_drum': instr.is_drum,
            'notes': n_proc,
        }
        proc_instrs.append(track_info)

    proc_data = {
        'instruments': proc_instrs,
        'ticks_per_beat': ticks_per_beat
    }
    return proc_data

if __name__ == '__main__':
    filename = 'test4.mid'
    res = proc(filename)

    with open(filename.split('.')[0]+'.json', 'w') as f:
        json.dump(res, f)




