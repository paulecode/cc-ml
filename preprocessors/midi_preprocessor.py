import pandas as pd

from prerunners.form_getter import form_getter


def preprocess_midi(note_df: pd.DataFrame, mainframe: pd.DataFrame):
    mainframe["form"] = mainframe["canonical_title"].apply(lambda x: form_getter(x))

    grouped_frame = note_df.groupby("midi_filename")["note"].apply(list).reset_index()

    merged_frame = pd.merge(
        mainframe,
        grouped_frame,
        left_on="midi_filename",
        right_on="midi_filename",
        how="left",
    )

    merged_frame.drop(columns=["midi_filename"], inplace=True)

    max_length_notes = max(len(notes) for notes in merged_frame["note"])
    merged_frame["note"] = merged_frame["note"].apply(
        lambda x: x + [0] * (max_length_notes - len(x))
    )

    return merged_frame
