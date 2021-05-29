import json
import os
import copy

config_path = os.path.join(os.path.dirname(__file__), "config.json")
config_path_backup = os.path.join(os.path.dirname(__file__), "config_backup.json")


def get_from_config(name):
    with open(config_path) as f:
        val = json.load(f)[name]
    return val


def get_path_from_config(name):
    paths = get_from_config(name)
    for p in paths:
        if os.path.exists(p):
            return p


def get_all_config():
    with open(config_path) as f:
        return json.load(f)


def save_to_config(name, val):
    all_config = get_all_config()
    all_config_new = copy.deepcopy(all_config)
    try:
        all_config_new[name] = val
        with open(config_path, 'w') as f:
            json.dump(all_config_new, f, indent=4)
    except Exception as e:
        with open(config_path_backup, 'w') as f:
            json.dump(all_config, f, indent=4)
        print("Exception happened while adding {} = {}"
              " to config file.".format(name, val))
        print("Error message: ", e)
        print("Backup copy of initial config file save to {}",
              config_path_backup)