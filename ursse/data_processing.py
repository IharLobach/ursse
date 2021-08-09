import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io import pickle
import seaborn as sns
import os
from ursse.hydra_harp_file_reader import HydraHarpFile
from ursse.LED_tests.data_analyzis import \
 calc_Fano, get_time_window_hist, calc_Fano_from_counts_per_time_window
from config_ursse import get_from_config
iota_period_sec = get_from_config("IOTA_revolution_period")
from scipy.stats import norm



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
        raise ValueError('File does not exist')
        #return save_event_delays(file_path)


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
                    time_stamp_file_name=None, ax=None):
    if ax is None:
        ax = sns.distplot(fanos, kde=False, bins=bins)
    else:
        sns.distplot(fanos, kde=False, bins=bins, ax=ax)
    ax.set_xlabel("$F-1$")
    ax.set_ylabel("Occurences")
    ax.set_title("Sampling distribution of Fano factor")
    if report:
        fano_interval = report['fano_interval']
        perc = report['fano_interval_percentiles']
        plt.axvline(fano_interval[0], linewidth=3)
        plt.axvline(fano_interval[1], linewidth=3)
        ax.text(0.99, 0.03, 'Vertical lines represent\n' +'  the percentiles ({:.2f}, {:.2f})'
                .format(perc[0], perc[1]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes)
    title = "Histogram for Fano factor.\n"
    if shift_folder_name:
        title += "Shift folder: {}. ".format(shift_folder_name)
    if time_stamp_file_name:
        title += "File: {}".format(time_stamp_file_name)
    ax.set_title(title)
    return ax


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


def load_one_pickle(pickle_path, gate1=(17500, 22500), gate2=(57000, 62000), spad1_channel=2, spad2_channel=3, dt_for_cutoff=2, dt_for_p=0.1):
    df0 = pd.read_pickle(pickle_path)
    total_iota_revolutions0 = df0.iloc[-1]['revolution']
    total_time0 = total_iota_revolutions0 * iota_period_sec

    # finding cutoff (where an electron is lost)
    n_per = dt_for_cutoff/iota_period_sec
    rev = df0['revolution']
    grouped = rev.groupby((rev/n_per).astype(int))
    av_rate = grouped.apply(lambda x: len(x.index)/(x.max()-x.min()))
    av_rev = grouped.apply(np.mean)
    av_rate0 = av_rate[0]


    if (av_rate.iloc[-1] > 0.75 * av_rate0):
        df = df0
    else:
        cutoff = (av_rev[av_rate > 0.75 * av_rate0]).iloc[-2]
        df = df0[df0['revolution'] < cutoff]

    # end finding cutoff




    total_iota_revolutions = df.iloc[-1]['revolution']
    df1_ = df[df['delay'].between(*gate1)].reset_index(drop=True)
    df2_ = df[df['delay'].between(*gate2)].reset_index(drop=True)
    df1 = df1_[df1_['channel'] == spad1_channel].drop(columns=['channel'])
    df2 = df2_[df2_['channel'] == spad2_channel].drop(columns=['channel'])
    total_iota_revolutions = df.iloc[-1]['revolution']
    total_time = total_iota_revolutions * iota_period_sec
    spad1_events = len(df1.index)
    rate1 = spad1_events/total_time
    spad2_events = len(df2.index)
    rate2 = spad2_events/total_time
    num_rev_above_one1 = np.sum(df1['revolution'].value_counts() > 1)
    num_rev_above_one2 = np.sum(df2['revolution'].value_counts() > 1)
    clean_df = df[(df['delay'].between(*gate1) & (df['channel'] == spad1_channel))
                  | (df['delay'].between(*gate2) & (df['channel'] == spad2_channel))]
    coincidence_events = np.sum(clean_df['revolution'].value_counts() == 2)
    coincidence_count_rate = coincidence_events/total_time
    p1 = spad1_events/total_iota_revolutions
    p2 = spad2_events/total_iota_revolutions
    p12_meas = coincidence_events/total_iota_revolutions

    # for p as a function of t
    n_per = dt_for_p/iota_period_sec
    p_dfs = []
    for df0 in [df1, df2]:
        rev = df0['revolution']
        grouped = rev.groupby((rev/n_per).astype(int))
        counts = grouped.apply(lambda x: len(x.index))
        times = dt_for_p * np.arange(len(counts.index))
        p_dfs.append(pd.DataFrame({'time_sec': times, 'counts': counts}))


    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(p_dfs[0]['time_sec'], p_dfs[0]['counts']/dt_for_p,
        label='SPAD1')
    ax.plot(p_dfs[1]['time_sec'], p_dfs[1]['counts']/dt_for_p,
        label='SPAD2')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count rate')
    ax.axvline(total_time, label='Auto data cutoff')
    ax.legend()
    plt.show()

    # hypothesis testing
    mu = np.sum(p_dfs[0]['counts'] * p_dfs[1]['counts']) / n_per
    sigma = np.sqrt(mu - np.sum((p_dfs[0]['counts'] * p_dfs[1]['counts'])**2)/n_per**3)

    Pvalue = 2 * norm.cdf(-np.abs(coincidence_events - mu), 0, sigma)

    return {
        "file_name": os.path.basename(pickle_path),
        "df": [df1, df2],
        "clean_df": clean_df,
        "p_df": p_dfs,
        "total_iota_revolutions0": total_iota_revolutions0,
        "total_time0": total_time0,
        "total_iota_revolutions": total_iota_revolutions,
        "total_time": total_time,
        "total_events": [spad1_events, spad2_events],
        "count_rate": [rate1, rate2],
        "num_rev_above_one": [num_rev_above_one1, num_rev_above_one2],
        "coincidence_count_rate": coincidence_count_rate,
        "p": [p1, p2],
        "p12_meas": p12_meas,
        "p1*p2": p1*p2,
        "mu": mu,
        "sigma": sigma,
        "total_coincidence_events": coincidence_events,
        "P-value": Pvalue
    }
