import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import bisect
from scipy.signal import find_peaks, peak_widths
import numpy as np


def __get_ch1_ch2_float(chunk, chunksize):
    ch1 = np.zeros(chunksize)
    ch2 = np.zeros(chunksize)
    chunkvalues = chunk[0].values
    for i in range(chunksize):
        ch1[i], ch2[i] = [float(x) for x in chunkvalues[i].split(";")]
    return ch1, ch2


def __find_centers(ar):
    return 0.5 * (ar[1:] + ar[:-1])


def find_trig_times(full_path, trigger_level=0.5, chunksize=8000000):
    file_extension = full_path.split(".")[-1]
    if file_extension=="csv":
        return __find_trig_times_csv(full_path, trigger_level, chunksize)
    if file_extension=="bin":
        return __find_trig_times_bin(full_path, trigger_level)



def __get_trig_times_for_one_numpy_ts(ts, trigger_level, chunksize):
    above_tl = np.where(ts > trigger_level, 1, 0)
    dif = np.diff(above_tl)
    pos_edges = np.where(dif > 0, 1, 0)
    ts_dif = np.diff(ts)
    timeline = np.arange(len(ts) - 1)
    interpolated_time = timeline - (ts[1:] - trigger_level)\
                        / np.where(ts_dif != 0,ts_dif,10)
    trig_times = interpolated_time[pos_edges > 0] + 1
    return trig_times

def __get_trig_times_for_one_chunk_bin(ch1, ch2, trigger_level, chunksize):
    return __get_trig_times_for_one_numpy_ts(ch1, trigger_level, chunksize), \
           __get_trig_times_for_one_numpy_ts(ch2, trigger_level, chunksize)


def __get_trig_times_for_one_chunk_list(ch1, ch2, trigger_level, chunksize):
    minch1, maxch1, minch2, maxch2 = [min(ch1), max(ch1), min(ch2), max(ch2)]
    amplitude1 = maxch1 - minch1
    amplitude2 = maxch2 - maxch1
    tl1 = minch1 + trigger_level * amplitude1
    tl2 = minch2 + trigger_level * amplitude2
    trig_times1 = []#np.zeros(800000)#[]
    trig_times2 = []#np.zeros(800000)#[]
    prev1 = ch1[0]
    prev2 = ch2[0]
    t = 0
    i1 = 0
    i2 = 0
    while t < chunksize:
        c1 = ch1[t]
        c2 = ch2[t]
        if prev1 < tl1 <= c1 :
            # trig_times1[i1]=t - (c1 - tl1) / (c1 - prev1)
            # i1+=1
            trig_times1.append(t - (c1 - tl1) / (c1 - prev1))
        prev1 = c1
        if prev2 < tl2 <= c2:
            # trig_times2[i2]=t - (c2 - tl2) / (c2 - prev2)
            # i2+=1
            trig_times2.append(t - (c2 - tl2) / (c2 - prev2))
        prev2 = c2
        t += 1
    return np.asarray(trig_times1),np.asarray(trig_times2),tl1,tl2,c1,c2,prev1,prev2,t


def __trim_times(trig_times1, trig_times2):
    trig_times2_centers = __find_centers(trig_times2)
    first_center_2 = trig_times2_centers[0]
    last_center_2 = trig_times2_centers[-1]
    first_time1_ind = 0
    while trig_times1[first_time1_ind] <= first_center_2:
        first_time1_ind += 1
    last_time1_ind = -1
    while trig_times1[last_time1_ind] > last_center_2:
        last_time1_ind -= 1
    return trig_times1[first_time1_ind:last_time1_ind], trig_times2[1:-1], first_center_2, last_center_2

def __find_trig_times_bin(full_path, trigger_level):
    data = np.fromfile(full_path,dtype=np.float32)
    ch1 = data[::2][1:]
    ch2 = data[1::2][1:]
    chunksize = len(ch1)
    trig_times1, trig_times2 = __get_trig_times_for_one_chunk_bin(ch1, ch2,
                                                                  trigger_level,
                                                                  chunksize)
    return __trim_times(trig_times1, trig_times2)

def __find_trig_times_csv(full_path, trigger_level, chunksize):
    chunks = pd.read_csv(full_path, chunksize=chunksize, header=None)
    chunk0 = next(chunks)
    ch1, ch2 = __get_ch1_ch2_float(chunk0, chunksize)
    trig_times1, trig_times2, tl1, tl2, c1, c2, prev1, prev2, \
    t= __get_trig_times_for_one_chunk_list(ch1, ch2, trigger_level, chunksize)
    for chunk in chunks:
        ch1, ch2 = __get_ch1_ch2_float(chunk, chunksize)
        for c1, c2 in zip(ch1, ch2):
            if prev1 < tl1 <= c1 :
                trig_times1.append(t - (c1 - tl1) / (c1 - prev1))
            prev1 = c1
            if prev2 < tl2 <= c2 :
                trig_times2.append(t - (c2 - tl2) / (c2 - prev2))
            prev2 = c2
            t += 1
    return __trim_times(trig_times1, trig_times2)


def find_trig_times1_mod(trig_times1, trig_times2, gate_tuple):
    trig_times2_centers = np.zeros(len(trig_times2)+1)
    trig_times2_centers[1:-1] = __find_centers(trig_times2)
    trig_times2_centers[0]= float("-inf")
    trig_times2_centers[-1]=float("inf")
    len_trig_times1 = len(trig_times1)
    trig_times1_mod = np.zeros(len_trig_times1)
    start_gate, end_gate = gate_tuple
    events = np.zeros(len(trig_times2),dtype=np.int8)
    j = 0
    c1 = trig_times1[j]
    i = 1
    while i < len(trig_times2_centers):
        c2 = trig_times2_centers[i]
        prev2 = trig_times2_centers[i - 1]
        if prev2 < c1 <= c2:
            t1 = trig_times2[i - 1]
            if t1 + start_gate < c1 < t1 + end_gate:
                events[i - 1] = 1
            trig_times1_mod[j] = c1 - trig_times2[i - 1]
            j += 1
            if j == len_trig_times1:
                break
            c1 = trig_times1[j]
        if c1 > c2: i += 1
    return trig_times1_mod, events

def find_gate(trig_times1_mod, gate_ns,gate_precision_ns):
    times1_max = max(trig_times1_mod)
    times1_min = min(trig_times1_mod)
    bin_size = gate_precision_ns
    my_bins = np.arange(times1_min,times1_max,bin_size)
    occurrences, bin_times = np.histogram(trig_times1_mod, bins=my_bins)
    bin_centers = 0.5 * (bin_times[1:] + bin_times[:-1])
    bins_per_gate=int(gate_ns/bin_size)
    oc_df = pd.DataFrame({'oc':occurrences})
    rolling = oc_df.rolling(window=bins_per_gate,min_periods=1).sum()
    end_idx = rolling['oc'].idxmax()
    start_idx = end_idx-bins_per_gate
    return bin_centers[start_idx],bin_centers[end_idx]


def show_count_time_dist_with_gate(trig_times1_mod, gate_tuple, bins_per_gate=5,save_plot_full_path=None,dpi=300):
    times1_max = max(trig_times1_mod)
    times1_min = min(trig_times1_mod)
    start =gate_tuple[0]
    end =gate_tuple[1]
    gate_ns = end-start
    bin_size = gate_ns/bins_per_gate
    my_bins = np.arange(times1_min,times1_max,bin_size)
    occurrences, bin_times = np.histogram(trig_times1_mod, bins=my_bins)
    bin_centers = 0.5 * (bin_times[1:] + bin_times[:-1])
    sns.distplot(trig_times1_mod, bins=my_bins, kde=False)
    plt.title("Distribution of pulse arrival times with respect to input pulses")
    plt.xlabel("Time, ns")
    plt.ylabel("Occurrences")
    plt.plot(bin_centers, occurrences)
    plt.axvline(start)
    plt.axvline(end)
    if save_plot_full_path:
        plt.savefig(save_plot_full_path, dpi=dpi, bbox_inches="tight")
    plt.show()

import os
from os import path
if __name__=="__main__":
    full_path_with_led = path.join(
        os.getcwd(),"..","11-06","RefCurve_2019-11-06_0_121900.Wfm.bin")
    trig_times1, trig_times2, first_center2, last_center2 = find_trig_times(
        full_path_with_led)
    trig_times1_mod, _ = find_trig_times1_mod(trig_times1, trig_times2)
    start_gate, end_gate = find_gate(trig_times1_mod, gate_ns=17,
                                     time_step_ns=2.5, bins_per_gate=10,
                                     show_plot=True, \
                                     save_plot_full_path=path.join(os.getcwd(),
                                                                   "led_tests_1.png"))
    gate_tuple = (start_gate, end_gate)
    _, events = find_trig_times1_mod(trig_times1, trig_times2, gate_tuple)
    True

