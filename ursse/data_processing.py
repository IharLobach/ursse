

def get_event_delays(f):
    """caclulates delays of detection events with respect to iota clock events
    f : instance of HydraHarpFile
    returns a pandas series"""
    # find first index where it's an iota clock event (most likely first)
    idx0 = 0
    while True:
        if f.TimeTags.iloc[idx0,0]==0:
            break
        else:
            idx0+=1
    df = f.TimeTags.iloc[idx0::] 
    t = df.TimeTag
    ch01 = df.Channel
    t_with_nan = t.where(ch01==0)
    t_iota_clock = t_with_nan.fillna(method='ffill')
    df_counts_only = df[ch01==4]
    t_delays = df_counts_only.TimeTag-t_iota_clock[ch01==4]
    return t_delays