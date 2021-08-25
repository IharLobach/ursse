import os
import sys
from config_ursse import get_from_config, get_path_from_config
import pandas as pd

data_folder = get_path_from_config("data_folder")

shifts_folder = os.path.join(data_folder, 'shifts')
shift_folders = os.listdir(shifts_folder)
repo_dir = get_path_from_config("repo_dir")
plot_style_sheets_path = os.path.join(
    repo_dir, "plot_style_sheets")


def show_shift_folders():
    global shift_folders
    for i, f in enumerate(shift_folders):
        print(i, f)


class WorkingDirectory():
    def __init__(self, wd):
        self.wd = wd

    def fi(self, file_name):
        return os.path.join(self.wd, file_name)


class PathAssistant():
    def __init__(self, shift_fodler_name, ignore_files=None):
        self.shift_folder_name = shift_fodler_name
        self.time_stamp_files_folder_path = \
            os.path.join(shifts_folder,
                         self.shift_folder_name, "time_stamp_files")
        self.shift_dir = os.path.join(shifts_folder,
                                      self.shift_folder_name)
        self.shift_results_dir = os.path.join(self.shift_dir,
                                              "results")
        self.waveforms_folder_path = \
            os.path.join(self.shift_dir, "waveforms")
        self.acnet_data_dir = os.path.join(self.shift_dir, "acnet_data")
        self.ignore_files = ['desktop.ini']
        if ignore_files:
            self.ignore_files += ignore_files

    def get_time_stamp_files_folder_path(self):
        return os.path.join(shifts_folder, self.shift_folder_name,
                            "time_stamp_files")

    def get_shift_cache_folder_path(self):
        return os.path.join(self.time_stamp_files_folder_path, "cache")

    def get_shift_files(self):
        shift = self.time_stamp_files_folder_path
        files = [f for f in os.listdir(shift)
                 if os.path.isfile(os.path.join(shift, f))]
        files1 = [f for f in files if (f not in self.ignore_files)]
        return [f for f in files1 if ('.ptu' in f)]

    def show_time_stamp_file_names(self):
        files = self.get_shift_files()
        for i, f in enumerate(files):
            print(i, f)

    def get_time_stamp_file_paths(self):
        shift = self.time_stamp_files_folder_path
        files = self.get_shift_files()
        return [os.path.join(shift, f) for f in files]

    def get_time_stamp_file_path(self, time_stamp_file_name):
        shift = self.time_stamp_files_folder_path
        return os.path.join(shift, time_stamp_file_name)

    def get_time_stamp_files_dir(self):
        return WorkingDirectory(self.time_stamp_files_folder_path)

    def get_waveforms_dir(self):
        return WorkingDirectory(self.waveforms_folder_path)

    def get_results_dir(self):
        return WorkingDirectory(self.shift_results_dir)

    def get_acnet_data_dir(self):
        return WorkingDirectory(self.acnet_data_dir)

    def get_pickle_path(self, time_stamp_file_name):
        return os.path.join(self.time_stamp_files_folder_path, "cache", time_stamp_file_name[:-3]+"pkl")

    def generate_csv_cache(self, time_stamp_file_name, filetype='T2'):
        repo_dir = get_path_from_config("repo_dir")
        ptu_file_path = self.get_time_stamp_file_path(time_stamp_file_name)
        output_file_path = \
        os.path.join(self.time_stamp_files_folder_path, "cache",    time_stamp_file_name[:-3]+"csv")
        if filetype=='T2':
            prog_path = os.path.join(repo_dir, "ursse_cpp",      "hydra_harp", "hydra_harp_reader")
            process = os.popen(f'{prog_path} "{ptu_file_path}" "{output_file_path}"')
            preprocessed = process.read()
            process.close()
            return preprocessed
        elif filetype=='T3':
            prog_path = os.path.join(repo_dir, "ursse", "hydra_harp_T3.py")
            process = os.popen(f'python {prog_path} "{ptu_file_path}" "{output_file_path}"')
            preprocessed = process.read()
            process.close()
            return preprocessed
        else:
            raise ValueError(f'Unknown filetype {filetype}')
    
    def generate_single_channel_cache_pickle(self, time_stamp_file_name, channel=2):
        self.generate_csv_cache(time_stamp_file_name)
        output_file_path = \
        os.path.join(self.time_stamp_files_folder_path, "cache", time_stamp_file_name[:-3]+"csv")
        df = pd.read_csv(output_file_path)
        df = df[df['channel']==channel].loc[:,['revolution', 'delay']].reset_index(drop=True)
        df.to_pickle(output_file_path[:-3]+"pkl")
        os.remove(output_file_path)



def get_plot_style_sheet(name):
    return os.path.join(plot_style_sheets_path, name+".mplstyle")
