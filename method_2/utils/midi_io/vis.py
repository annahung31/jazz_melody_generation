from src.midi2pianoroll import TrackPianoroll
from src.core_midi import midi_parser
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# load file
filename = 'AI_Realbook_091_split_1.mid'
midi = midi_parser.MidiFile(filename)

# visualize the 1st track
tidx = 0
track = TrackPianoroll(midi.instruments[tidx], midi.max_tick, midi.ticks_per_beat, midi.tick_to_time)

# symbolic
track.time_range = (0, 6600)
track.pitch_tange = (40, 60)
track.plot_pianoroll_midi('symbolic.png')
print(track, '\n\n')

# control change
track.plot_control_change(64, 'cc_64')
print(track.pianoroll.shape)
