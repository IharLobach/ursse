import os
import sys
data_folder = 'G:\\My Drive\\UHCICAGO\\Thesis\\URSSE'
shifts_folder = os.path.join(data_folder, 'shifts')
shift_folders = os.listdir(shifts_folder)


def show_shift_folders():
    global shift_folders
    for i, f in enumerate(shift_folders):
        print(i, f)


def get_shift_path(shift_folder_name):
    return os.path.join(shifts_folder, shift_folder_name,
                        "time_stamp_files")


def get_shift_cache_folder_path(shift_folder_name):
    return os.path.join(get_shift_path(shift_folder_name), "cache")
    

def get_shift_files(shift_folder_name):
    shift = get_shift_path(shift_folder_name)
    files = [f for f in os.listdir(shift) \
        if os.path.isfile(os.path.join(shift, f))]
    return files


def show_time_stamp_file_names(shift_folder_name):
    files = get_shift_files(shift_folder_name)
    for i, f in enumerate(files):
        print(i, f)


def get_time_stamp_file_paths(shift_folder_name):
    shift = get_shift_path(shift_folder_name)
    files = get_shift_files(shift_folder_name)
    return [os.path.join(shift, f) for f in files]


def get_time_stamp_file_path(shift_folder_name, time_stamp_file_name):
    shift = get_shift_path(shift_folder_name)
    return os.path.join(shift, time_stamp_file_name)


class WorkingDirectory():
    def __init__(self, wd):
        self.wd = wd


    def fi(self, file_name):
        return os.path.join(self.wd, file_name)