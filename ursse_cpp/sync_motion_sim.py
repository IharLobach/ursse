import numpy as np
import pandas as pd
import ursse_cpp.sync_motion_sim_cpp as sm_cpp




def get_simulated_revolution_delay_data(gamma, alpha, Vrf, f, h, JE, k, theta,
                                        revolution_numbers,
                                        tau0, delta0, rand_seed_int=1,
                                        rf_noise_std=0):
    revs = np.array(revolution_numbers, dtype=np.int64)
    phi0 = tau0*1e-9*h*f*2*np.pi
    res_dict = sm_cpp.get_simulated_revolution_delay_data(
        np.array([gamma, alpha, Vrf, f, h, JE, k, theta,
                  phi0, delta0, rand_seed_int, rf_noise_std],
                 dtype=np.float64),
        revs)
    taus = 1e9/2/np.pi/(h*f)*res_dict["phi"]
    deltas = res_dict["delta"]
    return pd.DataFrame({"revolution": revs, "delay": taus, "delta": deltas})


def RandomEnergyGammaDistribution(k, theta, size, seed):
    return sm_cpp.RandomEnergyGammaDistribution(
        np.array([k, theta, size, seed], dtype=np.float64))


InvSynchFractInt = sm_cpp.InvSynchFractInt
