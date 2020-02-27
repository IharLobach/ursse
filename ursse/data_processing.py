import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ursse.hydra_harp_file_reader import HydraHarpFile
from ursse.LED_tests.data_analyzis import \
 calc_Fano, get_time_window_hist, calc_Fano_from_counts_per_time_window


def get_event_delays(f, channel=1):
    """caclulates delays of detection events with respect to iota clock events
    f : instance of HydraHarpFile
    returns a pandas series"""
    # find first index where it's an iota clock event (most likely first)
    idx0 = 0
    while True:
        if f.TimeTags.iloc[idx0, 0] == 0:
            break
        else:
            idx0 += 1
    df0 = f.TimeTags.iloc[idx0::]
    df = df0[(df0.Channel == 0) | (df0.Channel == channel)]
    t = df.TimeTag
    ch04 = df.Channel.values
    t_with_nan = t.where(ch04 == 0)
    t_iota_clock = t_with_nan.fillna(method='ffill')
    df_counts_only = df[ch04 == channel]
    t_delays = df_counts_only.TimeTag-t_iota_clock[ch04 == channel]
    ch10 = np.where(ch04 == 0, 1, 0)
    revolution = ch10.cumsum()
    revolution_counts_only = revolution[ch04 == channel]
    return pd.DataFrame(
        {"revolution": revolution_counts_only, "delay": t_delays.values},
        index=t_delays.index), revolution[-1]+1


def plot_arrival_time_hist(t_delays, gate, bins=None,
                           yscale='log', saveas=None):
    ax = sns.distplot(t_delays, kde=False, bins=bins)
    ax.set_yscale(yscale)
    ax.set_ylabel('Occurrences of photocounts')
    ax.set_xlabel('Time relaltive to IOTA clock, ps')
    plt.axvline(gate[0])
    plt.axvline(gate[1])
    if saveas:
        plt.savefig(saveas)
    plt.show()


def get_events_array(df, n_revolutions, gate):
    counts_revolutions = df.revolution.values[df.delay.between(gate[0], gate[1])]
    if np.any(np.diff(counts_revolutions)==0):
        raise Exception("More than one event per revolution within the gate"
                        " encountered. But it should never happen!")
    events = np.zeros(n_revolutions, dtype=np.uint8)
    np.put(events, counts_revolutions, 1)
    return events


def get_fanos(events, n_revolutions, n_of_chunks=50,
              stat_interval=(0.16, 0.84), print_report=True):
    report = {}
    p_measured = sum(events)/n_revolutions
    report['p_measured'] = p_measured
    chunk_length = n_revolutions // n_of_chunks
    new_length = n_of_chunks * chunk_length
    chunks = np.reshape(events[:new_length], (n_of_chunks, chunk_length))
    report['chunck_length'] = chunk_length
    n_events = sum(events)
    report['n_events'] = n_events
    fanos = np.apply_along_axis(calc_Fano_from_counts_per_time_window,
                                1, chunks)
    fanos = np.sort(fanos)
    i1 = int(stat_interval[0]*len(fanos))
    i2 = int(stat_interval[1]*len(fanos))
    f1 = fanos[i1]
    f2 = fanos[i2]
    fano_interval = (f1, f2)
    report['fano_interval'] = fano_interval
    report['fnao_interval_percentiles'] = stat_interval
    fano_median = np.median(fanos)
    report['fano_median'] = fano_median
    fano_mean = np.mean(fanos)
    report['fano_mean'] = fano_mean
    error = (f2-f1)/2
    report['absolute_fano_error'] = error
    if print_report:
        for r in report:
            print("{} = {}".format(r, report[r]))
    return fanos, report


def plot_fanos_hist(fanos, fano_interval=None, bins=None):
    sns.distplot(fanos, kde=False, bins=bins)
    plt.xlabel("F-1")
    plt.ylabel("Occurences")
    plt.title("Sampling distribution of Fano factor")
    if fano_interval:
        plt.axvline(fano_interval[0])
        plt.axvline(fano_interval[1])
    plt.show()


def process_file(file_name, channel=1, gate=(59000, 70000), n_of_chunks=50,
                 print_output=True, bins_delay=None, bins_fano=None):
    f = HydraHarpFile(file_name, safemode=False)
    df, n_revolutions = get_event_delays(f, channel)
    t_delays = df.delay
    if print_output:
        plot_arrival_time_hist(t_delays, gate, bins=bins_delay)
    events = get_events_array(df, n_revolutions, gate)
    fanos, report = get_fanos(events, n_revolutions, n_of_chunks,
                              print_report=print_output)
    if print_output:
        plot_fanos_hist(fanos, report['fano_interval'], bins=bins_fano)
    return fanos, report
