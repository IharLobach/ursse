import numpy as np
import pandas as pd
import ursse_cpp.sync_motion_sim_cpp as sm

# res = sm.get_trajectory(
#     np.array(
#         [100/0.511,
#         0.07088,
#         380,
#         30e6,
#         100000,
#         0,
#         0,
#         1])
# )

# print(pd.DataFrame(res))

res = sm.get_simulated_revolution_delay_data(
    np.array(
        [100/0.511,
         0.07088,
         380,
         30e6,
         0,
         0,
         1],
         ),
    np.array([1, 3, 5, 8, 20], dtype=np.int64)
)

print(pd.DataFrame(res))
