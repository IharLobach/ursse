import ursse_cpp.sync_motion_sim as sm
from config_ursse import get_from_config
import ursse.path_assistant as pa
import os
import sys
from ursse.hydra_harp_file_reader import HydraHarpFile
from ursse.LED_tests.data_analyzis import calc_Fano, get_time_window_hist, calc_Fano_from_counts_per_time_window
import numpy as np
import pandas as pd
import seaborn as sns
from ursse.data_processing import \
    get_event_delays, plot_arrival_time_hist, get_events_array, get_fanos, \
    plot_fanos_hist, process_file, read_event_delays, save_event_delays
import matplotlib.pyplot as plt
import seaborn as sns
iota_period_sec = get_from_config("IOTA_revolution_period")
dt_sec = get_from_config("dt")
iota_period_au = iota_period_sec/dt_sec
from ursse.time_structure import get_bucket_gates, get_rate_in_gate_Hz,\
    reduce_df_to_one_gate, divide_df_into_time_bins, get_properties_in_time_bins
plt.rcParams['figure.figsize'] = [15, 3.75]
plt.rcParams.update({'font.size': 16, 'legend.fontsize': 16})
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy import stats
iota_period_sec = get_from_config("IOTA_revolution_period")


def get_revolution_delay_df_one_gate(
    shift_folder, file_name, gate=(61000, 66000), show_histogram=False):
    """returns df for one gate (one bucket), the average delay is subtracted, delay is in ns"""
    df, n_revolutions = read_event_delays(
        pa.PathAssistant(shift_folder).get_time_stamp_file_path(file_name))
    four_gates = get_bucket_gates(gate)
    if show_histogram:
        n = 500
        i = 200
        plot_arrival_time_hist(
            df[i*n:(i+1)*n,'delay'], gate=gate, bins=1000, yscale='linear')
    df0 = reduce_df_to_one_gate(df, gate).reset_index(drop=True)
    delay_mean = df0['delay'].mean()
    df0['delay'] = (df0['delay']-delay_mean)/1000
    return df0


def f(t, a0, A, B, T):
    return a0 + A*np.cos(2*np.pi/T*t)+B*np.sin(2*np.pi/T*t)


def get_initial_sync_period_estimate(df0, t0=2372,
                                     first_fit_nper=30,
                                     show_plot=False, errorbar=0.35):
    df0_first_fit = df0[df0['revolution']<t0*first_fit_nper]
    x = df0_first_fit['revolution'].values
    y = df0_first_fit['delay'].values
    p0 = (np.mean(y), np.std(y), np.std(y), t0)
    popt, pcov = curve_fit(f, x, y, p0=p0)
    a0, a, b, T0 = popt
    perr = np.sqrt(np.diag(pcov))
    x_fit = np.linspace(x[0],x[-1],10000)
    y_fit = f(x_fit, a0, a, b, T0)
    if show_plot:
        plt.rcParams.update({'font.size': 15,
                     'legend.fontsize':22,
                     'errorbar.capsize': 3,
                     'figure.figsize':(15,3)})
        fig, ax = plt.subplots()
        ax.errorbar(x, y, label='SPAD counts', yerr=errorbar, marker='o', linestyle='None')
        ax.plot(x_fit, y_fit, label='Least squares fit')
        ax.set_ylabel('Detection time relative to\n IOTA revolution marker, ns', fontsize=16)
        ax.set_xlabel('IOTA revolution number')
        ax1 = ax.twiny()
        ax1.set_xlim(1000*iota_period_sec*np.asarray(ax.get_xlim()))
        ax1.set_xlabel('Time, ms')
        #ax.set_xlim(0,0.0025)
        ax.legend(loc=1, fontsize=12)
        fig.show()
    return T0


def get_phase_df_from_revoluton_delay_df(df0, T0, fitper=20,
                                         overlapper=3, nper_step=10):
    fitrange = T0*fitper
    overlaprange = T0*overlapper
    starts = df0.groupby(((df0['revolution'].values)//(fitrange-overlaprange)).astype(int))\
        .apply(lambda v: v.index.values[0]).values
    ends = df0.groupby(((df0['revolution'].values-overlaprange)//(fitrange-overlaprange)).astype(int))\
        .apply(lambda v: v.index.values[-1]).values[1:]
    nintervals = min(len(starts), len(ends))
    starts = starts[:nintervals]
    ends = ends[:nintervals]
    

    alist, blist = [],[]
    for s,e in zip(starts, ends):
        v = df0.loc[s:e,:]
        x = v['revolution'].values
        y = v['delay'].values
        A = np.vstack([np.cos(2*np.pi/T0*x),np.sin(2*np.pi/T0*x)]).T
        a,b = np.linalg.lstsq(A, y, rcond=None)[0]
        alist.append(a)
        blist.append(b)
    

    fits_df = pd.DataFrame({'start_idx': starts, 'end_idx': ends,
                            'A': alist, 'B': blist})
    fits_df_len = len(fits_df.index)
    fits_df['start_revolution'] = df0.loc[starts, 'revolution']\
        .reset_index(drop=True)
    fits_df['end_revolution'] = df0.loc[ends, 'revolution']\
        .reset_index(drop=True)
    fits_df['next_start_revolution'] = fits_df['start_revolution']\
        .shift(-1).values.astype(int)
    fits_df['previous_end_revolution'] = fits_df['end_revolution']\
        .shift(1).values.astype(int)
    fits_df.loc[fits_df_len-1, 'next_start_revolution'] = \
        fits_df.loc[fits_df_len-1, 'end_revolution']
    fits_df.loc[0, 'previous_end_revolution'] = \
        fits_df.loc[0, 'start_revolution']
    fits_df['next_A'] = fits_df['A'].shift(-1)
    fits_df['next_B'] = fits_df['B'].shift(-1)

    def get_one_fit(row):
        revs = np.arange(row['previous_end_revolution'],
                            row['end_revolution'],
                            nper_step)

        def interpAB(val1, val2):
            return np.interp(
                revs,
                [
                    row['previous_end_revolution'],
                    row['next_start_revolution'],
                    row['end_revolution']
                ],
                [val1, val1, val2]
            )
        avals = interpAB(row['A'], row['next_A'])
        bvals = interpAB(row['B'], row['next_B'])
        dels = f(revs, 0, avals, bvals, T0)
        return pd.DataFrame({'revolution': revs.astype(int), 'delay': dels})

    res = fits_df.apply(get_one_fit, axis=1)
    res = pd.concat(res.to_list(), ignore_index=True, sort=False)

    new_revs = np.arange(0, res['revolution'].max(), nper_step)
    new_dels = scipy.interpolate.interp1d(res['revolution'], res['delay'],
                            bounds_error=False,
                            fill_value="extrapolate")(new_revs)
    res = pd.DataFrame({'revolution': new_revs, 'delay': new_dels})

    trigger_level = 0
    ts = res['delay'].values
    above_tl = np.where(ts > trigger_level, 1, 0)
    dif = np.diff(above_tl)
    pos_edges = np.where(dif > 0, 1, 0)
    ts_dif = np.diff(ts)
    timeline = np.arange(len(ts) - 1)
    interpolated_time = timeline - (ts[1:] - trigger_level)\
                        / np.where(ts_dif != 0, ts_dif, 10)
    trig_times = nper_step*interpolated_time[pos_edges > 0] + 1

    rev_numbers = np.arange(len(trig_times))
    Tfit = stats.linregress(rev_numbers, trig_times).slope
    phase_df = pd.DataFrame({"time_sec": trig_times*iota_period_sec,
                             "phase_rad": 2*np.pi*(rev_numbers-trig_times/Tfit)
                             })

    fdf = fits_df.loc[:,
                      ['start_idx',
                       'end_idx',
                       'start_revolution',
                       'end_revolution',
                       'A',
                       'B']]
    fdf['mid_time_sec'] = (fitper-overlapper)*iota_period_sec*T0*np.arange(len(fdf.index))
    fdf['Amplitude_ns'] = np.sqrt(fdf['A']**2+fdf['B']**2)
    fdf['Amp2'] = fdf['Amplitude_ns']**2
    fdf['Kicks'] = fdf['Amp2'].diff().fillna(0)
    return phase_df, fdf


def get_sz_df(df0, spad_tts_ns, dt=0.1):
    revolutions_per_dt = dt/iota_period_sec
    df0['index_of_dt_bin'] = (df0.revolution/revolutions_per_dt).astype(int)
    df0['time_sec'] = df0['index_of_dt_bin']*dt+dt/2
    rms_delay_df = pd.DataFrame({
        'time_sec': df0.groupby('index_of_dt_bin')['index_of_dt_bin']
        .mean()*dt+dt/2,
        'sz_ns': np.sqrt(df0.groupby('index_of_dt_bin').delay.std()**2-spad_tts_ns**2)
    }).reset_index(drop=True)
    return rms_delay_df


def get_polar_df(phase_df, sz_df, npoints=10000):
    time = np.linspace(
        min(phase_df['time_sec'].min(), sz_df['time_sec'].min()),
        max(phase_df['time_sec'].max(), sz_df['time_sec'].max()),
        npoints)
    phase = np.interp(time, phase_df['time_sec'], phase_df['phase_rad'])
    sz = np.interp(time, sz_df['time_sec'], sz_df['sz_ns'])
    polar_df = pd.DataFrame({'time_sec': time, 'phase_rad':phase, 'sz_ns': sz})
    return polar_df


def plot_polar_df_cartesian(polar_df):
    plt.rcParams.update({'font.size': 15,
                     'legend.fontsize':22})
    fig, ax = plt.subplots(2,figsize=(15,6))
    ax[0].plot(polar_df['time_sec'],
            polar_df['sz_ns'],
            linewidth=2)
    #ax[0].set_xlabel('Time (sec)')
    ax[0].set_ylabel('$\sigma_z$ (ns)')
    ax[0].set_ylim(0, ax[0].get_ylim()[1])
    ax[1].plot(polar_df['time_sec'], polar_df['phase_rad'])
    ax[1].set_ylabel(r"Slow phase $\phi$ (rad)")
    ax[1].set_xlabel("Time (sec)")
    fig.show()


def plot_polar_df(polar_df, tmax_sec = 5.0):
    plt.rcParams.update({'font.size': 13,
                     'legend.fontsize':22,
                     'errorbar.capsize': 3,
                     'figure.figsize':(7,7),
                     'figure.dpi':300})
    time = polar_df['time_sec']
    print(f"shown time = {tmax_sec:.1f} sec")
    nmax = int(tmax_sec/max(time)*len(time))
    res_plt = polar_df[:nmax]
    plt.polar(res_plt['phase_rad'],res_plt['sz_ns'],'.-',
            linewidth=1, markersize=1.5)
    plt.show()




