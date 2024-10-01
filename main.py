import os

import pandas as pd

from experiments.midi.experiment01_randomforest_midi import train_random_forest_midi
from experiments.wav.experiment01_randomforest_wav import train_random_forest_wav
from preprocessors.midi_preprocessor import preprocess_midi
from preprocessors.wav_preprocessor import preprocess_wav
from prerunners.assembler import frame_assembly
from prerunners.dataset_loader import dataset_loader


if __name__ == "__main__":
    df = dataset_loader("maestro-v3.0.0/maestro-v3.0.0.csv")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/midi", exist_ok=True)
    os.makedirs("checkpoints/wav", exist_ok=True)

    if not os.path.exists(
        "checkpoints/midi/checkpoint1-note.csv"
    ) or not os.path.exists("checkpoints/midi/checkpoint1-meta.csv"):
        print("Checkpoint files not found. Running assembler...")
        note_df, meta_df = frame_assembly(df)
    else:
        print("Checkpoint files found. Loading...")
        note_df = pd.read_csv("checkpoints/midi/checkpoint1-note.csv")
        meta_df = pd.read_csv("checkpoints/midi/checkpoint1-meta.csv")

    wav_df = preprocess_wav(df)

    train_random_forest_wav(wav_df)

    midi_df = preprocess_midi(note_df, df)

    train_random_forest_midi(midi_df)
