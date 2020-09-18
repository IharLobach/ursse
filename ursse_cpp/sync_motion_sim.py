import numpy as np
import pandas as pd
import ursse_cpp.sync_motion_sim_cpp as sm_cpp


def get_trajectory(gamma, alpha, Vrf, f, n_IOTA_per,
                   tau0, delta0, rand_seed_int=1):
    phi0 = tau0*1e-9*f*2*np.pi
    res_dict = sm_cpp.get_trajectory(
        np.array([gamma, alpha, Vrf, f, n_IOTA_per,
                  phi0, delta0, rand_seed_int],
        dtype=np.float64)
    )
    taus = 1e9/2/np.pi/f*res_dict["phi"]
    deltas = res_dict["delta"]
    return pd.DataFrame({"tau_ns": taus, "delta": deltas})
