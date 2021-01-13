import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ursse.phase_space_trajectory as pst
import ursse_cpp.sync_motion_sim as sm
from config_ursse import get_from_config
from ursse.path_assistant import PathAssistant
import os
from scipy.stats import chisquare

class Model:
    def __init__(
        self,
        rf_noise_std,
        gamma=None,
        alpha=None,
        f=None,
        h=None,
        k=None,
        theta=None,
        meMeV=None,
        rho=None,
        JE=None,
        files_and_pars=None,
        V=None,
        spad_tts=0.35,
        spad_mean=0.5,
        rms_size_dt_sec=0.1
        ):
        self.rf_noise_std = rf_noise_std
        if gamma is None:
            self.gamma = get_from_config("gamma")
        else:
            self.gamma = gamma
        if alpha is None:
            self.alpha = get_from_config("ring_alpha")
        else:
            self.alpha = alpha
        if f is None:
            self.f = 1/get_from_config("IOTA_revolution_period")
        else:
            self.f = f
        if h is None:
            self.h = get_from_config("RF_q")
        else:
            self.h = h
        if k is None:
            self.k = get_from_config("M")
        else:
            self.k = k
        if theta is None:
            self.theta = get_from_config("Et")/self.k
        else:
            self.theta = theta
        if meMeV is None:
            self.meMeV = get_from_config("me_MeV")
        else:
            self.meMeV = meMeV
        if rho is None:
            self.rho = get_from_config("dipole_rho_m")
        else:
            self.rho = rho
        if JE is None:
            self.JE = get_from_config("damping_partition_JE")
        else:
            self.JE =JE
        if files_and_pars is None:
            self.files_and_pars = [
                    {
                        "shift": 'shift_02_28_2020',
                        "file": '1el_002.ptu',
                        "tau0": None,
                        "delta0": None,
                        "rand_seed_int": 1,
                        "spad_tts_rand_seed": None
                    },
                    {
                        "shift": 'shift_02_28_2020',
                        "file": '1el_000.ptu',
                        "tau0": None,
                        "delta0": None,
                        "rand_seed_int": 2,
                        "spad_tts_rand_seed": None
                    },
                    {
                        "shift": 'shift_02_28_2020',
                        "file": '1el_001.ptu',
                        "tau0": None,
                        "delta0": None,
                        "rand_seed_int": 3,
                        "spad_tts_rand_seed": None
                    },
                    {
                        "shift": 'shift_03_05_2020',
                        "file": '1el_filters_000.ptu',
                        "tau0": None,
                        "delta0": None,
                        "rand_seed_int": 4,
                        "spad_tts_rand_seed": None
                    },
                    {
                        "shift": 'shift_03_05_2020',
                        "file": '1el_filters_008.ptu',
                        "tau0": None,
                        "delta0": None,
                        "rand_seed_int": 5,
                        "spad_tts_rand_seed": None
                    }
                ]
        else:
            self.files_and_pars = files_and_pars
        self.V = get_from_config("Vrf") if V is None else V
        self.spad_tts = spad_tts
        self.spad_mean = spad_mean
        self.rms_size_dt_sec = rms_size_dt_sec

    def simulate(self):
        calc_sim_df_several_files(self.rf_noise_std, self.files_and_pars,
                                  self.V)
        return self.files_and_pars
    
    def add_spad_tts_do_fitting_and_binning(self, verbose=False):
        add_spad_tts_do_fitting_and_binning(self.files_and_pars,
                                    spad_tts=self.spad_tts,
                                    mean_spad=self.spad_mean,
                                    verbose=verbose,
                                    rms_size_dt_sec=self.rms_size_dt_sec)
        return self.files_and_pars
    
    def get_meas_sim_comparison(self, feature, nbins=20,
                                do_chi2=True, chi2_min_count=50,
                                ax=None):
        return get_meas_sim_comparison(self.files_and_pars, feature,
                                nbins, do_chi2, chi2_min_count,
                                rf_noise_std=self.rf_noise_std,
                                spad_tts=self.spad_tts,
                                spad_mean=self.spad_mean,
                                dt=self.rms_size_dt_sec,
                                ax=ax)


def calc_sim_df_one_file(shift, file, rf_noise_std, tau0=None, delta0=None,
                rand_seed_int=1, V=None):
    gamma = get_from_config("gamma")
    alpha = get_from_config("ring_alpha")
    if V is None:
        V = get_from_config("Vrf")
    f = 1/get_from_config("IOTA_revolution_period")
    h = get_from_config("RF_q")
    k = get_from_config("M")
    theta = get_from_config("Et")/k
    meas_df = pst.get_revolution_delay_df_one_gate(shift, file)
    meMeV = get_from_config("me_MeV")
    rho = get_from_config("dipole_rho_m")
    JE = get_from_config("damping_partition_JE")
    E0 = gamma*meMeV*1e6
    eta = alpha - 1/gamma**2
    delta_rms = 0.62e-6*gamma/np.sqrt(JE*rho)
    if tau0 is None:
        tau0 = 1e9*delta_rms / \
            ((f*h)*2*np.pi*np.sqrt(V/(2*np.pi*E0*h*np.abs(eta))))
    if delta0 is None:
        delta0 = delta_rms
    sim_df = sm.get_simulated_revolution_delay_data(
        gamma, alpha, V, f, h, JE, k, theta,
        meas_df['revolution'],
        tau0=tau0, delta0=delta0, rand_seed_int=rand_seed_int,
        rf_noise_std=rf_noise_std)
    return meas_df, sim_df


def add_spad_tts_to_sim_df(sim_df, spad_tts=0.35, mean_spad=0.5,
                           show_spad_tt_dist=False, np_rand_seed=None):
    """Adds random delays to the delays (photon arrival times) in sim_df. The added delays are drawn from a Gamma distribtuion with mean=spad_mean and std=spad_tts

    Args:
        sim_df (pandas df): input sim_df
        spad_tts (float, optional): transit time spread of the spad. Defaults to 0.35.
        mean_spad (float, optional): mean time delay of the spad, for the Gamma distribution model. Defaults to 0.5.
        show_spad_tt_dist (bool, optional): show a histogram for the model Gamma distribution of the spad. Defaults to False.

    Returns:
        [type]: [description]
    """
    np.random.seed(np_rand_seed)
    sim_df = sim_df.copy()
    std_spad = spad_tts
    theta_spad = std_spad**2/mean_spad
    k_spad = mean_spad/theta_spad
    ts_spad = np.random.gamma(k_spad, theta_spad, size=len(sim_df.index))
    if show_spad_tt_dist:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.hist(ts_spad, bins=100, density=True)
        plt.show()
    sim_df['delay'] = sim_df['delay'] + ts_spad
    return sim_df


def calc_sim_df_several_files(rf_noise_std, files_and_pars=None, V=None):
    """[summary]

    Args:
        rf_noise_std (float): [description]
        files_and_pars (list of dic, optional): list of files and parameters for these files. Defaults to None. See source code: it uses five files from 02-28-2020 and 03-05-2020
        V (float, optional): RF voltage. Defaults to None (uses config.json).

    Returns:
        [type]: [description]
    """
    if files_and_pars is None:
        files_and_pars = [
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_002.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 1,
                "spad_tts_rand_seed": None
             },
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_000.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 2,
                "spad_tts_rand_seed": None
             },
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_001.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 3,
                "spad_tts_rand_seed": None
             },
            {
                "shift": 'shift_03_05_2020',
                "file": '1el_filters_000.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 4,
                "spad_tts_rand_seed": None
             },
            {
                "shift": 'shift_03_05_2020',
                "file": '1el_filters_008.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 5,
                "spad_tts_rand_seed": None
             }
        ]
    for i, el in enumerate(files_and_pars):
        print(f"working on {i+1} out of {len(files_and_pars)}")
        el['meas_df'], el['sim_df_before_spad'] = calc_sim_df_one_file(
                                            el['shift'],
                                            el['file'],
                                            rf_noise_std,
                                            el['tau0'],
                                            el['delta0'],
                                            el['rand_seed_int'],
                                            V)
        el['sim_df'] = el['sim_df_before_spad'].copy()
    return files_and_pars


def add_spad_tts_do_fitting_and_binning(files_and_pars,
                           spad_tts=0.35,
                           mean_spad=0.5,
                           verbose=False,
                           rms_size_dt_sec=0.1):
    for el in files_and_pars:
        el['sim_df'] = add_spad_tts_to_sim_df(
            el['sim_df_before_spad'], spad_tts, mean_spad,
            np_rand_seed=el['spad_tts_rand_seed'])
    for i, el in enumerate(files_and_pars):
        if verbose:
            print(f"working on file number {i+1} out of {len(files_and_pars)}")
        el["meas_T0"] = pst.get_initial_sync_period_estimate(el["meas_df"])
        el["meas_phase_df"], el["meas_fits_df"] = \
            pst.get_phase_df_from_revoluton_delay_df(
                el["meas_df"], el["meas_T0"])
        el["meas_sz_df"] = pst.get_sz_df(el["meas_df"], spad_tts_ns=0,
                                         dt=rms_size_dt_sec)
        el["meas_polar_df"] = pst.get_polar_df(
            el["meas_phase_df"], el["meas_sz_df"])
        el["sim_T0"] = pst.get_initial_sync_period_estimate(el["sim_df"])
        el["sim_phase_df"], el["sim_fits_df"] = \
            pst.get_phase_df_from_revoluton_delay_df(
                el["sim_df"], el["sim_T0"])
        el["sim_sz_df"] = pst.get_sz_df(el["sim_df"], spad_tts_ns=0,
                                        dt=rms_size_dt_sec)
        el["sim_polar_df"] = pst.get_polar_df(
            el["sim_phase_df"], el["sim_sz_df"])
    return files_and_pars


def get_meas_sim_hist(input_list, nbins=20,
        do_chi2=True, chi2_min_count=50):
    el = input_list[0]
    l = min(el['meas'].min(), el['sim'].min())
    r = max(el['meas'].max(), el['sim'].max())
    for el in input_list[1:]:
        lc = min(el['meas'].min(), el['sim'].min())
        rc = max(el['meas'].max(), el['sim'].max())
        if lc < l:
            l = lc
        if rc > r:
            r = rc
    bins = np.linspace(l, r, nbins)
    bin_centers = (bins[1:]+bins[:-1])/2
    hist_list = []
    for el in input_list:
        hist_list.append(
            {'meas': np.histogram(el['meas'], bins=bins)[0],
             'sim': np.histogram(el['sim'], bins=bins)[0]}
        )
    hist_meas = np.zeros(bin_centers.shape)
    hist_sim = np.zeros(bin_centers.shape)
    for el in hist_list:
        hist_meas += el['meas']
        hist_sim += el['sim']
    res = {}
    res['individual_hists'] = hist_list
    res['aggregated_hists'] = {'meas': hist_meas, 'sim': hist_sim}    
    if do_chi2:
        above_chi2_min_count = (
            hist_meas > chi2_min_count) * (hist_sim > chi2_min_count)
        meas_chi2 = hist_meas[above_chi2_min_count]
        meas_chi2[-1] += np.sum(hist_meas[~above_chi2_min_count])
        sim_chi2 = hist_sim[above_chi2_min_count]
        sim_chi2[-1] += np.sum(hist_sim[~above_chi2_min_count])
        res['chi-squared'] = chisquare(f_obs=meas_chi2, f_exp=sim_chi2)
    res['bins'] = bins
    res['bin_centers'] = bin_centers
    # calculating mean below
    means = []
    lens = []
    for el in input_list:
        means.append({
            'meas': el['meas'].mean(),
            'sim': el['sim'].mean()
        })
        lens.append({
            'meas': len(el['meas'].index),
            'sim': len(el['sim'].index)
        })
    res['individual_means'] = means
    meas_tot_len = sum([el['meas'] for el in lens])
    sim_tot_len = sum([el['sim'] for el in lens])
    res['tot_means'] = {
        'meas': sum([a['meas']*b['meas'] for a, b in zip(means, lens)])/meas_tot_len,
        'sim': sum([a['sim']*b['sim'] for a, b in zip(means, lens)])/sim_tot_len,
    }
    
    return res


def get_meas_sim_comparison(files_and_pars, feature, nbins=20,
                            do_chi2=True, chi2_min_count=50,
                            rf_noise_std=None,
                            spad_tts=None,
                            spad_mean=None,
                            dt=None,
                            ax=None):
    """files_and_pars must have pre-calculated fits and rms length

    Args:
        files_and_pars ([type]): [description]
        feature ([type]): [description]
        nbins (int, optional): [description]. Defaults to 20.
        do_chi2 (bool, optional): [description]. Defaults to True.
        chi2_min_count (int, optional): [description]. Defaults to 50.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if feature == "amplitude":
        input_list = [{'meas': el['meas_fits_df']['Amplitude_ns'], 'sim': el['sim_fits_df']['Amplitude_ns']}
                      for el in files_and_pars]
        xlabel = "Amplitude (ns)"
    elif feature == "rms_length":
        input_list = [{'meas': el['meas_sz_df']['sz_ns'], 'sim': el['sim_sz_df']['sz_ns']}
                      for el in files_and_pars]
        xlabel = "RMS of electron position in a time window"
        if dt is not None:
            xlabel += f" dt={dt:.2f} sec"
    elif feature == "slow_phase":
        def aux_func(ser):
            return (ser-ser.mean()).abs()
        input_list = [
            {'meas': aux_func(el['meas_phase_df']['phase_rad']),
             'sim': aux_func(el['sim_phase_df']['phase_rad'])}
                      for el in files_and_pars]
        xlabel = r"|Slow phase-$\langle$Slow phase$\rangle$| $(\mathrm{rad})^2$"
    elif feature == "kick_to_amplitude":
        def aux_func(ser):
            return np.sqrt(ser.abs())
        input_list = [
            {'meas': aux_func(el['meas_fits_df']['Kicks']),
             'sim': aux_func(el['sim_fits_df']['Kicks'])}
                    for el in files_and_pars]
        xlabel = "|Kick to Amplitude$^2$|$^{1/2}$ (ns)"
    else:
        raise ValueError("Unknown feature for comparison. Choose from"
        "amplitude, rms_length, slow_phase, kick_to_amplitude")
    hist_dic = get_meas_sim_hist(input_list, nbins, do_chi2, chi2_min_count)
    if ax is not None:
        bins = hist_dic['bins']
        bin_centers = hist_dic['bin_centers']
        ax.hist(bin_centers, weights=hist_dic['aggregated_hists']['meas'], bins=bins, density=False,
                alpha=0.5, label="Measurement")
        ax.hist(bin_centers, weights=hist_dic['aggregated_hists']['sim'], bins=bins, density=False,
                alpha=0.5, label="Simulation")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Occurrences")
        desc = f"p-value = {hist_dic['chi-squared'].pvalue:.2e}" \
                + "\n" + f"meas_mean = {hist_dic['tot_means']['meas']:.2e}" \
                + "\n" + f"sim_mean = {hist_dic['tot_means']['sim']:.2e}"
        if spad_tts is not None:
            desc += "\n" + f"spad_tts = {spad_tts:.3f}"
        if spad_mean is not None:
            desc += "\n" + f"spad_mean = {spad_mean:.3f}"
        if rf_noise_std is not None:
            desc += "\n" + f"rf_noise_std = {rf_noise_std:.2e}"

        ax.annotate(desc, (0.5, 0.5), xycoords='axes fraction')
        ax.legend()
    return hist_dic
