import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_Fano_from_counts_per_time_window(ar):
    return np.var(ar)/np.mean(ar)-1

def calc_Fano(events,n_periods):
    tot_per = len(events)
    counts_per_bin, bs = np.histogram(np.linspace(0, tot_per - 1, tot_per), bins=np.arange(0, tot_per, n_periods), weights=events)
    return calc_Fano_from_counts_per_time_window(counts_per_bin)

def divide_into_chunks(l, n):
    for i in range(0, len(l), n):
        yield sum(l[i:i + n])


def get_time_window_hist(events,n_periods,show_plot = False,save_plot_full_path=None,dpi=300):
    counts_per_time_bin = list(divide_into_chunks(events, n_periods))
    counts_min = min(counts_per_time_bin)
    counts_max = max(counts_per_time_bin)
    occurences, _ = np.histogram(counts_per_time_bin, bins=np.arange(counts_min - 0.5, counts_max + 0.5, 1))

    sns.barplot(list(range(counts_min, counts_max)), occurences, color='g', alpha=0.5)
    plt.title("Distribution of number of counts within a time window = {} periods".format(n_periods))
    plt.ylabel("Occurrences")
    plt.xlabel("Count")
    if save_plot_full_path:
        plt.savefig(save_plot_full_path,dpi=dpi,bbox_inches = "tight")
    if show_plot:
        plt.show()


    return occurences,(counts_min,counts_max)

if __name__=="__main__":
    get_time_window_hist(np.asarray([0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,
                                     0,0,0,0,0,1,0,0,0,0,0]),5)