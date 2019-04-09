
import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import time
import gc
import copy
import argparse
import pickle
import tarfile
import multiprocessing
from multiprocessing import Queue, JoinableQueue
from threading import Thread
from os.path import splitext, basename, join, isfile, dirname, realpath, exists
from datetime import datetime

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io
import cv2

from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform

_DISTANCE_THRESHOLD = 0.8


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


def sanitize(data):
  """ convert array to a c-contiguous float array """
  if data.dtype != 'float32':
    data = data.astype('float32')
  if not data.flags['C_CONTIGUOUS']:
    data = np.ascontiguousarray(data)
  return data

results = {}

def worker(values):

  global results
  start = datetime.now()
  for i, (query, index) in enumerate(values):

    # debug
    if i == 3:
      break

    path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_test_rescale_delf_pkl/{}.pkl'.format(query)
    if not exists(path):
      continue

    loc1, desc1 = pickle_load(path)
    if desc1 is None: continue
    num_features_1 = desc1.shape[0]

    index_array = index.split(' ')
    index_first = index_array[:5]
    index_last = index_array[5:]

    # print(path)
    if i % 5 == 0 and i != 0:
      elapsed = (datetime.now() - start).total_seconds()
      print('last 5 images took {}'.format(elapsed))
      start = datetime.now()


    res = []
    for name in index_first:

      path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_rescale_delf_pkl/{}.pkl'.format(name)
      loc2, desc2 = pickle_load(path)
      if desc2 is None: continue
      num_features_2 = desc2.shape[0]

      # Find nearest-neighbor matches using a KD tree.
      # d1_tree = cKDTree(desc1)
      # _, indices = d1_tree.query(
      #   desc2, distance_upper_bound=_DISTANCE_THRESHOLD)

      index = faiss.IndexFlatL2(40)
      index.add(sanitize(desc2))
      D, I = index.search(sanitize(desc1), 10)

      ix = np.arange(0, desc1.shape[0]).repeat(10).reshape(desc1.shape[0], 10)

      locations_2_to_use = loc2[I[D < 0.8]]
      locations_1_to_use = loc1[ix[D < 0.8]]

      # # Select feature locations for putative matches.
      # locations_2_to_use = loc2[indices[indices != -1]]
      # locations_1_to_use = loc1[indices[indices != -1]]

      try:
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
          (locations_1_to_use, locations_2_to_use),
          AffineTransform,
          min_samples=3,
          residual_threshold=20,
          max_trials=2)
        nb_inliers = 0 if inliers is None else np.sum(inliers)
      except:
        nb_inliers = 0
      res.append((name, nb_inliers))

    sorted_res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)]
    results[query] = ' '.join(sorted_res) + ' '.join(index_last)


if __name__ == '__main__':
  submit = pd.read_csv('./ssr_k256/submit_2018-04-06_18.43.46_vlad_full_index.csv.gz')

  values = np.array_split(submit.values, 12)
  pool = multiprocessing.Pool(processes=12)
  pool.map(worker, values)

  sample_submission = join('/home/alexandrearaujo/kaggle/landmark-retrieval-challenge', 'data', 
      'sample_submission.csv')

  submit = pd.read_csv(sample_submission)
  submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))
  submit_filename = 'submit_{}_rerank.csv.gz'.format(
              datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
  submit.to_csv(submit_filename, index=False, compression='gzip')
