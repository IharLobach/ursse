import numpy as np
import pandas as pd


def split_into_chunks(df, n_of_chunks, n_revolutions, gate):
    chunk_periods = n_revolutions // n_of_chunks
    new_length = n_of_chunks * chunk_periods
    truncated_df = df[df['revolution'] < new_length]
    truncated_df['chunk_number'] = truncated_df['revolution'] // chunk_periods
    return truncated_df.groupby("chunk_number")["delay"].apply(np.asarray)

