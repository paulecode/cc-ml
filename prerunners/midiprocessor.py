import mido
import os

import pandas as pd

def midi_preprocess(filename):

    print(f'Processing {filename}')
    rootDir = 'maestro-v3.0.0'
    mid = mido.MidiFile(os.path.join(rootDir, filename))

    meta_data_list = []
    note_list = []

    for track_index, track in enumerate(mid.tracks):
        absolute_time = 0
        for message_index, message in enumerate(track):
            absolute_time += message.time
            if message.is_meta:
                meta_message = message.dict()
                if len(meta_message) == 2:
                    meta_data_list.append([
                        filename, track_index, absolute_time, str(
                            message.type), '-'
                    ])
                elif len(meta_message) == 3:
                    meta_message.pop('type')
                    meta_message.pop('time')
                    for key, value in meta_message.items():
                        meta_data_list.append([
                            filename, track_index, absolute_time, str(
                                message.type), str(value)
                        ])
                else:
                    meta_message.pop('type')
                    meta_message.pop('time')
                    for key, value in meta_message.items():
                        meta_data_list.append([
                            filename, track_index, absolute_time, f"{message.type}_{key}", str(
                                value)
                        ])
            elif message.type == 'control_change':
                if 'control=64' in str(message):
                    meta_message = message.dict()
                    meta_data_list.append([
                        filename, track_index, absolute_time, 'pedal', meta_message['value']
                    ])
            elif message.type == 'note_on' or message.type == 'note_off':
                meta_message = message.dict()
                note_list.append([
                    filename, track_index, absolute_time, message.type, meta_message[
                        'note'], meta_message['velocity']
                ])
            note_frame = pd.DataFrame(
                note_list, columns=['midi_filename', 'track', 'time', 'type', 'note', 'velocity'])
            meta_frame = pd.DataFrame(meta_data_list, columns=[
                'midi_filename', 'track', 'time', 'key', 'value'])
    print(f'Processed {filename}')
    return note_frame, meta_frame