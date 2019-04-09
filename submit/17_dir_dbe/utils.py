
import pickle
from os.path import isfile, splitext
from datetime import datetime


class Timer:
    
    def __init__(self, msg):
        self.msg = msg
    
    def __enter__(self):
        self.start = datetime.now()
        
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start).total_seconds()
        print('{} on last batch took {:.3f}'.format(self.msg, elapsed),
              flush=True)


def pickle_load(path):
  """function to load pickle object"""
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')

def pickle_dump(file, path, force=False):

  def _pickle_dump(file, path):
    """function to dump picke object"""
    with open(path, 'wb') as f:
      pickle.dump(file, f, -1)

  def _get_new_name(path):
    """rename file if file already exist"""
    i = 0
    new_path = path
    while isfile(new_path):
      ext = splitext(path)[1]
      new_path = path.replace(ext, '_{}{}'.format(i, ext))
      i += 1
    return new_path

  """dump a file without deleting an existing one"""
  if force:
    _pickle_dump(file, path)
  elif not force:
    new_path = _get_new_name(path)
    _pickle_dump(file, new_path)
