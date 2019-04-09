
import os
import sys
import glob
import pickle
import multiprocessing
import hashlib
from datetime import datetime
from os.path import join, splitext, basename, exists, isfile

import numpy as np
import cv2

STATUS_CHECK_ITERATIONS = 10000
N_PROCESS = 12

def pickle_load(path):
  """function to load pickle object"""
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')

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

def pickle_dump(file, path, force=False):
  """dump a file without deleting an existing one"""
  if force:
    _pickle_dump(file, path)
  elif not force:
    new_path = _get_new_name(path)
    _pickle_dump(file, new_path)


def extract_sift(list_images_path):
  output_dir = sys.argv[2]
  sift = cv2.xfeatures2d.SIFT_create(1000)
  num_images = len(list_images_path)
  start = datetime.now()
  obj = []
  pkl_filename = ''
  for i, img_path in enumerate(list_images_path):
    filename = splitext(basename(img_path))[0]

    if i % STATUS_CHECK_ITERATIONS == 0 and i != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('Loading image {} out of {}, last {} images took {:.5f}'
                ' seconds'.format(i, num_images, STATUS_CHECK_ITERATIONS, 
                    elapsed), flush=True)
        start = datetime.now()
    
    try:
      pkl_filename += filename
      img = cv2.imread(img_path)
      kp, des = sift.detectAndCompute(img, None)
      loc = np.array([k.pt for k in kp])
      obj.append((filename, loc, des))
      if len(obj) == 100:
        hash_name = hashlib.sha512(bytes(pkl_filename.encode('utf8')))
        pkl_filename = hash_name.hexdigest()[:20]
        pickle_dump(obj, join(output_dir, '{}.sift'.format(pkl_filename)))
        obj = []
    except Exception as e:
      print('warning: Failed to extact on image {}. error : {}'.format(
        filename, e))
      continue

  # last batch
  hash_name = hashlib.sha512(bytes(pkl_filename.encode('utf8')))
  pkl_filename = hash_name.hexdigest()[:20]
  pickle_dump(obj, join(output_dir, '{}.sift'.format(pkl_filename)))



def main():
  if len(sys.argv) != 3:
    print('Syntax: %s <input_dir> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (input_dir, output_dir) = sys.argv[1:]

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  list_images_path = glob.glob(join(input_dir, '*'))
  list_images_path_split = np.array_split(list_images_path, N_PROCESS)

  pool = multiprocessing.Pool(processes=N_PROCESS)
  pool.map(extract_sift, list_images_path_split)


if __name__ == '__main__':
  main()