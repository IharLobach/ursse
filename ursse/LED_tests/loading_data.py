import pandas as pd
def load_data(full_path):
    data =  pd.read_csv(full_path,header=None)
    len_data = len(data.index)
    ch1 = [0]*len_data
    ch2 = [0]*len_data
    for i,line in enumerate(data.values):
        ch1[i],ch2[i] = [float(x) for x in line[0].split(";")]
    return [ch1,ch2]