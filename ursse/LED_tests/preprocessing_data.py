from .trig_times import find_trig_times
import pandas as pd
def generate_AllCounts12(wfs,wfs_names,time_step_ns,all_counts_1_pickle_name,all_counts_2_pickle_name,verbose=True,verbose_update_step = 10):
    all_counts_1_list = []
    all_counts_2_list = []
    for i in range(len(wfs)):
        if verbose and i%verbose_update_step==0:
            print('working on waveform #{}'.format(i))
        trig_times1,trig_times2,_,_ = find_trig_times(wfs[i])
        all_counts_1_list.append(pd.Series(trig_times1*time_step_ns))#conversion to ns
        all_counts_2_list.append(pd.Series(trig_times2*time_step_ns))#conversion to ns
    all_counts_1 = pd.concat(all_counts_1_list,keys = wfs_names,names = ['waveform_name','count_time_ns'])
    all_counts_2 = pd.concat(all_counts_2_list,keys = wfs_names,names = ['waveform_name','count_time_ns'])
    all_counts_1.to_pickle(all_counts_1_pickle_name)
    all_counts_2.to_pickle(all_counts_2_pickle_name)
    if verbose:
        print('Done.')