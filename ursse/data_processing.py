import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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


def save_event_delays(file_path, channel=1):
    f = HydraHarpFile(file_path, safemode=False)
    df, n_revolutions = get_event_delays(f, channel)
    df1 = pd.DataFrame({"revolution": [n_revolutions], "delay": [0]})
    df.append(df1, ignore_index=True)
    dir_name = os.path.dirname(file_path)
    cache_folder = os.path.join(dir_name, "cache")
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    f_name = os.path.basename(file_path)
    pkl_name = f_name.replace("ptu", "pkl")
    pkl_path = os.path.join(cache_folder, pkl_name)
    df.to_pickle(pkl_path)
    return df, n_revolutions


def read_event_delays(file_path):
    dir_name = os.path.dirname(file_path)
    cache_folder = os.path.join(dir_name, "cache")
    f_name = os.path.basename(file_path)
    pkl_name = f_name.replace("ptu", "pkl")
    pkl_path = os.path.join(cache_folder, pkl_name)
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        n_revolutions = df.iloc[-1, 0]
        df = df.iloc[:-1, :]
        return df, n_revolutions
    else:
        return save_event_delays(file_path)


def plot_arrival_time_hist(t_delays, gate=None, bins=None,
                           yscale='log', saveas=None,
                           shift_folder_name=None,
                           time_stamp_file_name=None):
    ax = sns.distplot(t_delays, kde=False, bins=bins)
    ax.set_yscale(yscale)
    ax.set_ylabel('Occurrences of photocounts')
    ax.set_xlabel('Time relaltive to IOTA clock, ps')
    if gate:
        plt.axvline(gate[0])
        plt.axvline(gate[1])
        ax.text(0.95, 0.01, 'gate = ({:.1f}, {:.1f}) ps'.format(gate[0],gate[1]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=15)
    title = "Histogram for photon count arrival time.\n"
    if shift_folder_name:
        title += "Shift folder: {}. ".format(shift_folder_name)
    if time_stamp_file_name:
        title += "File: {}".format(time_stamp_file_name)
    plt.title(title)
    if saveas:
        plt.savefig(saveas)
    plt.show()


def get_events_array(df, n_revolutions, gate):
    counts_revolutions = \
        df.revolution.values[df.delay.between(gate[0], gate[1])]
    if np.any(np.diff(counts_revolutions) == 0):
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
    report['chunk_length'] = chunk_length
    n_events = sum(events)
    report['n_events'] = n_events
    # fanos = np.apply_along_axis(calc_Fano_from_counts_per_time_window,
    #                             1, chunks)
    fanos = np.var(chunks, axis=1)/np.mean(chunks, axis=1)-1
    fanos = np.sort(fanos)
    i1 = int(stat_interval[0]*len(fanos))
    i2 = int(stat_interval[1]*len(fanos))
    f1 = fanos[i1]
    f2 = fanos[i2]
    fano_interval = (f1, f2)
    report['fano_interval'] = fano_interval
    report['fano_interval_percentiles'] = stat_interval
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


def plot_fanos_hist(fanos, report=None, bins=None,
                    shift_folder_name=None,
                    time_stamp_file_name=None):
    ax = sns.distplot(fanos, kde=False, bins=bins)
    ax.set_xlabel("F-1")
    ax.set_ylabel("Occurences")
    ax.set_title("Sampling distribution of Fano factor")
    if report:
        fano_interval = report['fano_interval']
        perc = report['fano_interval_percentiles']
        plt.axvline(fano_interval[0])
        plt.axvline(fano_interval[1])
        ax.text(0.95, 0.01, 'Vertical lines represent the percentiles ({:.3f}, {:.3f})'
                .format(perc[0], perc[1]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=15)
    title = "Histogram for Fano factor.\n"
    if shift_folder_name:
        title += "Shift folder: {}. ".format(shift_folder_name)
    if time_stamp_file_name:
        title += "File: {}".format(time_stamp_file_name)
    ax.set_title(title)
    plt.show()


def process_file(file_name, gate=(60000, 70000), n_of_chunks=50,
                 print_output=True, bins_delay=None, bins_fano=None):
    df, n_revolutions = read_event_delays(file_name)
    t_delays = df.delay
    if print_output:
        plot_arrival_time_hist(t_delays, gate, bins=bins_delay)
    events = get_events_array(df, n_revolutions, gate)
    fanos, report = get_fanos(events, n_revolutions, n_of_chunks,
                              print_report=print_output)
    if print_output:
        plot_fanos_hist(fanos, report, bins=bins_fano)
    return fanos, report
