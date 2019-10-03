import mido
from utils.midi_io.src.core_midi import midi_parser


def write_midi_notestream(
        tracks,
        ticks_per_beat,
        filename='result.mid',
        numerator=4,
        denominator=4,
        key=None):

    """Write note stream into midi file.

    Parameters
    ----------
    track_list : list
        list of dict containing necessary info of a track. The format should be:
            {
                'note_stream': notes_stream,
                'name': track.name,
                'is_drum': track.is_drum,
                'program': track.program
            }
    ticks_per_beat : int
        beat resolution of a beat
    numerator : int
        time signature (numerator)
    denominator : int
        time signature (denominator)
    key : str
        key signature
    """

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # meta data track
    meta_track = mido.MidiTrack()
    meta_track.append(
        mido.MetaMessage(
            'time_signature',
            time=0,
            numerator=numerator,
            denominator=denominator))

    if key is not None:
        meta_track.append(
            mido.MetaMessage(
                'key_signature',
                time=0,
                key=key))

    meta_track.append(mido.MetaMessage('end_of_track', time=meta_track[-1].time + 1))
    mid.tracks.append(meta_track)

    for tidx, track in enumerate(tracks):
        track = write_track_notestream(
            track['note_stream'],
            program=track['program'],
            is_drum=track['is_drum'],
            name=track['name'],
            tidx=tidx)
        mid.tracks.append(track)
    mid.save(filename=filename)


def write_track_notestream(notes, program=0, is_drum=False, name=None, tidx=0):
    mtrack = mido.MidiTrack()
    if name is not None:
        mtrack.append(mido.MetaMessage(
            'track_name', time=0, name=name))

    # note to message
    event_list = []
    for idx, note in enumerate(notes):
        st, duration, p, v = note
        if duration == 0:
            # warning
            event_list.append(('note_on', p, 0, st))
        else:
            event_list.extend([
                ('note_on', p, v, st),
                ('note_on', p, 0, st+duration)])
    # sort by time
    event_list = sorted(event_list, key=lambda x: x[-1])

    # set channel
    channels = list(range(16))
    channels.remove(9)
    if is_drum:
        channel = 9
    else:
        channel = channels[tidx % len(channels)]

    # set program
    mtrack.append(mido.Message('program_change', channel=channel, program=program, time=0))

    # set notes
    current = 0
    for idx, e in enumerate(event_list):
        delta = e[-1] - current
        current = e[-1]
        mtrack.append(mido.Message(e[-0], channel=channel, note=e[1], velocity=e[2], time=delta))

    # add end of track
    mtrack.append(mido.MetaMessage('end_of_track', time=mtrack[-1].time + 1))
    return mtrack


if __name__ == '__main__':
    filename = 'test.mid'
    midi = midi_parser.MidiFile(filename)

    # loading for example
    track_list = []
    for tidx, track in enumerate(midi.instruments):
        notes_stream = []
        for note in track.notes:
            notes_stream.append([note.start, note.end-note.start, note.pitch, note.velocity])

        track_list.append({
            'note_stream': notes_stream,
            'name': track.name,
            'is_drum': track.is_drum,
            'program': track.program
        })

    # write back
    write_midi_notestream(track_list, midi.ticks_per_beat)
