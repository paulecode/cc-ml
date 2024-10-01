import os
import librosa
import numpy as np
import pandas as pd

from prerunners.form_getter import form_getter


def compute_audio_features(audio_filename):
    y, sr = librosa.load("maestro-v3.0.0/" + audio_filename)

    rms = librosa.feature.rms(y=y).flatten().tolist()

    return rms


def preprocess_wav(df: pd.DataFrame):

    df["form"] = df["canonical_title"].apply(lambda x: form_getter(x))

    df["rms"] = df["audio_filename"].apply(compute_audio_features)

    max_length_rms = max(len(rms) for rms in df["rms"])

    df["rms"] = df["rms"].apply(lambda x: x + [0] * (max_length_rms - len(x)))

    return df
