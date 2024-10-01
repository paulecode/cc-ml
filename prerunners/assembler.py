import pandas as pd
from multiprocessing import Pool, cpu_count

from prerunners.midiprocessor import midi_preprocess


def frame_assembly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function assembles the dataframes from the midiprocessor and dataset_loader modules.
    Warning, CPU Heavy and takes very long to run.
    """

    with Pool(cpu_count()) as p:
        results = p.map(midi_preprocess, df["midi_filename"].tolist())

    note_df = pd.concat([result[0] for result in results])
    meta_df = pd.concat([result[1] for result in results])

    note_df.to_csv("checkpoints/midi/checkpoint1-note.csv", index=False)
    meta_df.to_csv("checkpoints/midi/checkpoint1-meta.csv", index=False)

    return note_df, meta_df
