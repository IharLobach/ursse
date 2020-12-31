import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ursse.phase_space_trajectory as pst
import ursse_cpp.sync_motion_sim as sm
from config_ursse import get_from_config
from ursse.path_assistant import PathAssistant
import os


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
    return sim_df


def add_spad_tts_to_sim_df(sim_df, spad_tts=0.35, mean_spad=0.5,
                           show_spad_tt_dist=False):
    """Adds random delays to the delays (photon arrival times) in sim_df. The added delays are drawn from a Gamma distribtuion with mean=spad_mean and std=spad_tts

    Args:
        sim_df (pandas df): input sim_df
        spad_tts (float, optional): transit time spread of the spad. Defaults to 0.35.
        mean_spad (float, optional): mean time delay of the spad, for the Gamma distribution model. Defaults to 0.5.
        show_spad_tt_dist (bool, optional): show a histogram for the model Gamma distribution of the spad. Defaults to False.

    Returns:
        [type]: [description]
    """
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
    if files_and_pars is None:
        files_and_pars = [
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_002.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 1
             },
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_000.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 2
             },
            {
                "shift": 'shift_02_28_2020',
                "file": '1el_001.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 3
             },
            {
                "shift": 'shift_03_05_2020',
                "file": '1el_filters_000.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 4
             },
            {
                "shift": 'shift_03_05_2020',
                "file": '1el_filters_008.ptu',
                "tau0": None,
                "delta0": None,
                "rand_seed_int": 5
             }
        ]
    for i, el in enumerate(files_and_pars):
        print(f"working on {i+1} out of {len(files_and_pars)}")
        el['sim_df'] = calc_sim_df_one_file(el['shift'],
                                            el['file'],
                                            rf_noise_std,
                                            el['tau0'],
                                            el['delta0'],
                                            el['rand_seed_int'],
                                            V)
    return files_and_pars
