import ursse.transverse_scan.spad_counts as spad_counts
import ursse.transverse_scan.picomotor_pos as picomotor_pos
from datetime import datetime
import numpy as np


def datetime_to_sec(t):
    return (t-datetime(1970, 1, 1)).total_seconds()


def get_positions_and_counts_arrays(scan_db_path, spad_rate_path):
    picomotor_dates, picomotor_position, year = \
        picomotor_pos.get_time_and_picomotor_pos_arr(scan_db_path)
    spad_dates, spad_cnts = \
        spad_counts.get_time_and_counts_arr(year, spad_rate_path)
    to_sec = np.vectorize(datetime_to_sec)
    spad_cnts_interp = np.interp(to_sec(picomotor_dates),
                                 to_sec(spad_dates), spad_cnts)
    return picomotor_position.transpose(), spad_cnts_interp
