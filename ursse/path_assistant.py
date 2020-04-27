import os
import sys
from config_ursse import get_from_config
data_folder = get_from_config("data_folder")
# '/mnt/c/Users/lobac_000/OneDrive - Fermi National Accelerator Laboratory/URSSE'
shifts_folder = os.path.join(data_folder, 'shifts')
shift_folders = os.listdir(shifts_folder)


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
        return [f for f in files if (f not in self.ignore_files)]

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
