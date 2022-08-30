import json, os, sys

def load_config(filename):
  path_for_config = './configs/'+filename
  with open(path_for_config) as f:
    cfg = json.load(f)
  
  return cfg


def path_add_to_sys_path(path_to_be_add):
  # path_to_be_add = '/notebook/personal/ksuchoi216/face-id-model/'
  if not path_to_be_add in sys.path:
    sys.path.insert(0, path_to_be_add)
    os.chdir(path_to_be_add)

  print(f'sys.path: {sys.path}')