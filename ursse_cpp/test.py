import numpy as np
import pandas as pd
import ursse_cpp.sync_motion_sim as sm

res = sm.get_trajectory(
         100/0.511,
         0.07088,
         380,
         30e6,
         100000,
         0,
         0,
         1)

print(res)
