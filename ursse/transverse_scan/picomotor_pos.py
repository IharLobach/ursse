import sqlite3
from datetime import datetime
import numpy as np


def get_time_and_picomotor_pos_arr(scan_db_path):
    conn = sqlite3.connect(scan_db_path)
    c = conn.cursor()
    sql_command = 'SELECT * FROM positions'
    c.execute(sql_command)
    res = c.fetchall()
    conn.close()
    dates_str = [l[0] for l in res]
    dates = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in dates_str]
    positions = [[int(pos) for pos in l[1].split(";")] for l in res]
    year = str(dates[0].year)
    return np.array(dates), np.array(positions), year