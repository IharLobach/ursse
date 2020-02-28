import os
import sys
data_folder = 'G:\\My Drive\\UHCICAGO\\Thesis\\URSSE'
shifts_folder = os.path.join(data_folder, 'shifts')
shift_folders = os.listdir(shifts_folder)


def show_shift_folders():
    global shift_folders
    for i, f in enumerate(shift_folders):
        print(i, f)


def get_shift_path(shift_index):
    return os.path.join(shifts_folder, shift_folders[shift_index],
                        "time_stamp_files")


def get_shift_files(shift_index):
    shift = get_shift_path(shift_index)
    files = os.listdir(shift)
    return files


def show_time_stamp_file_names(shift_index):
    files = get_shift_files(shift_index)
    for i, f in enumerate(files):
        print(i, f)


def get_time_stamp_file_paths(shift_index):
    shift = get_shift_path(shift_index)
    files = get_shift_files(shift_index)
    return [os.path.join(shift, f) for f in files]


class WorkingDirectory():
    def __init__(self, wd):
        self.wd = wd


    def fi(self, file_name):
        return os.path.join(self.wd, file_name)