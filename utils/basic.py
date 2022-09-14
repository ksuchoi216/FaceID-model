import json
import os
import sys


class Config_Manager:
    def __init__(self):
        self.cfg_path = None

    def set_cfg_path(self, cfg_path):
        self.cfg_path = cfg_path

    def get_cfg(self):
        with open(self.cfg_path) as f:
            self.cfg = json.load(f)
        return self.cfg

    def print(self):
        print(self.cfg)


def checkLocalServer(isLocalServer, sys_path_to_be_added=None):
    print(f'isLocalServer is {isLocalServer}')
    if isLocalServer is True:
        if sys_path_to_be_added not in sys.path:
            sys.path.insert(0, sys_path_to_be_added)
            os.chdir(sys_path_to_be_added)

        print("System path as follows:")
        for path in sys.path:
            print(f"{path}")