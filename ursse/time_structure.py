import numpy as np
import pandas as pd
from config_ursse import get_from_config
iota_period_sec = get_from_config("IOTA_revolution_period")
dt_sec_hydra_harp = get_from_config("dt")
iota_period_au = iota_period_sec/dt_sec_hydra_harp


def split_into_chunks(df, n_of_chunks, n_revolutions, gate):
    chunk_periods = n_revolutions // n_of_chunks
    new_length = n_of_chunks * chunk_periods
    truncated_df = df[df['revolution'] < new_length]
    truncated_df['chunk_number'] = truncated_df['revolution'] // chunk_periods
    return truncated_df.groupby("chunk_number")["delay"].apply(np.asarray)


def get_rate_in_gate_Hz(df, n_revolutions, gate):
    t_delays = df.delay
    t_delays_in_gate = t_delays[t_delays.between(gate[0], gate[1])]
    return len(t_delays_in_gate)/n_revolutions/iota_period_sec


def get_bucket_gates(gate):
    buckets = [iota_period_au/4*i for i in range(4)]
    bucket_gates = [((gate[0]+b) % iota_period_au,
                     (gate[1]+b) % iota_period_au) for b in buckets]
    return bucket_gates


def reduce_df_to_one_gate(df, gate):
    return df[(df.delay > gate[0]) & (df.delay < gate[1])]


def divide_df_into_time_bins(df, dt):
    revolutions_per_dt = dt/iota_period_sec
    df['index_of_dt_bin'] = (df.revolution/revolutions_per_dt).astype(int)
    df['bin_time'] = dt*df['index_of_dt_bin']
    df['time_sec'] = df['revolution']*iota_period_sec
    return df[df['index_of_dt_bin'] < df['index_of_dt_bin'].max()]


def get_properties_in_time_bins(df):
    grouped = df.groupby('index_of_dt_bin')
    res_df = pd.DataFrame({
        'time_sec': grouped.bin_time.mean(),
        'count': grouped.delay.count(),
        'std': grouped.delay.std()})
    return res_df
