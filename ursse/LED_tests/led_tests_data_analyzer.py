#%%

import matplotlib.pyplot as plt
import os
from os import path
plt.rcParams['figure.figsize'] = [15, 7.5]
plt.rcParams.update({'font.size': 20,'legend.fontsize':22})
#from loading_data import load_data
from LED_tests.trig_times import find_trig_times,find_trig_times1_mod,find_gate
#from finding_period import find_period
from LED_tests.data_analyzis import get_time_window_hist


#%%

full_path_no_led = path.join(os.getcwd(),"no-led","RefCurve_2019-10-31_2_141058.Wfm.csv")
full_path_with_led = path.join(os.getcwd(),"with-led","RefCurve_2019-10-31_1_135001.Wfm.csv")

#%%

trig_times1,trig_times2,first_center2,last_center2 = find_trig_times(full_path_with_led)

#%%

# ch1,ch2 = load_data(full_path_with_led)

#%%

# plt.plot(ch2[:2000])
# plt.show()

#%%

trig_times1_mod,_ = find_trig_times1_mod(trig_times1,trig_times2)

#%%

start_gate,end_gate = find_gate(trig_times1_mod,gate_ns=17,time_step_ns=2.5,bins_per_gate=10,show_plot=True,\
                                save_plot_full_path=path.join(os.getcwd(),"led_tests_1.png"))

#%%

gate_tuple = (start_gate,end_gate)
_,events = find_trig_times1_mod(trig_times1,trig_times2,gate_tuple)


#%%

n_periods = 5*int(len(trig_times2)/sum(events))
occurrences,counts_min_max = get_time_window_hist(events,n_periods,show_plot=True,\
                                                  save_plot_full_path=path.join(os.getcwd(),"led_tests_2.png"))
