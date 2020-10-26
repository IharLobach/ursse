import numpy as np
import pandas as pd
import ursse_cpp.sync_motion_sim_cpp as sm_cpp




def get_simulated_revolution_delay_data(gamma, alpha, Vrf, f, h, JE, k, theta,
                                        revolution_numbers,
                                        tau0, delta0, rand_seed_int=1):
    # double gamma = prms[0];
    # double alpha = prms[1];
    # double V = prms[2];
    # double f = prms[3];
    # int h = (int)(prms[4]);
    # double JE = prms[5];
    # double k = prms[6];
    # double theta = prms[7];
    # double phi0 = prms[8];
    # double delta0 = prms[9];
    # int seed = (int)(prms[10]);
    revs = np.array(revolution_numbers, dtype=np.int64)
    phi0 = tau0*1e-9*h*f*2*np.pi
    res_dict = sm_cpp.get_simulated_revolution_delay_data(
        np.array([gamma, alpha, Vrf, f, h, JE, k, theta,
                  phi0, delta0, rand_seed_int],
                 dtype=np.float64),
        revs)
    taus = 1e9/2/np.pi/(h*f)*res_dict["phi"]
    deltas = res_dict["delta"]
    return pd.DataFrame({"revolution": revs, "delay": taus, "delta": deltas})


def RandomEnergyGammaDistribution(k, theta, size, seed):
    return sm_cpp.RandomEnergyGammaDistribution(
        np.array([k,theta,size,seed], dtype=np.float64))


InvSynchFractInt = sm_cpp.InvSynchFractInt
