from datetime import datetime
import numpy as np


def get_time_and_counts_arr(year, spad_rate_path):
    with open(spad_rate_path) as f:
        lines = f.readlines()
    only_data = [l[:-1].split("\t") for l in lines[12:-2]]
    dates_str = [year+" "+l[0] for l in only_data]
    dates = \
        [datetime.strptime(t, "%Y %a %b %d %H:%M:%S.%f") for t in dates_str]
    counts = [int(l[1]) for l in only_data]
    return np.array(dates), np.array(counts)
