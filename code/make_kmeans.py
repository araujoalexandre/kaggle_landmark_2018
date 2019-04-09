
import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import argparse
import pickle
import tarfile
from os.path import splitext, basename, join, isfile, dirname, realpath, exists
from datetime import datetime

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io

# custom class 
from kmeans import Kmeans


def pickle_load(path):
  """function to load pickle object"""
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')


# def images_iterator(images_path):
#   for iteration, path in enumerate(images_path):
#     filename = splitext(basename(path))[0]
#     try:
#       loc, _, desc, _, _ = feature_io.ReadFromFile(path)
#     except Exception as error:
#       print('problem with file {}. error : {}'.format(
#               filename, error), flush=True)
#       desc = np.zeros((1, 40))
#     yield iteration, filename, desc

def images_iterator(images_path):
  for iteration, path in enumerate(images_path):
    filename = splitext(basename(path))[0]
    loc, desc = pickle_load(path)
    if not all([isinstance(filename, str), isinstance(desc, np.ndarray)]):
      print('file {} is corrupted'.format(filename))
      filename = ''
      desc = np.zeros((1, 128))
    yield iteration, filename, desc


def sanitize(data):
  """ convert array to a c-contiguous float array """
  if data.dtype != 'float32':
    data = data.astype('float32')
  if not data.flags['C_CONTIGUOUS']:
    data = np.ascontiguousarray(data)
  return data

def make_kmeans(k):

  cur_dir = dirname(realpath(__file__))
  files = glob.glob('/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_rescale_delf_pkl/*')

  # init kmeans
  kmeans = Kmeans(40, k, niter=100, verbose=True, max_points_per_centroid=10000000, gpu=False)

  database = []
  total = 0
  print('started', flush=True)
  for iteration, filename, desc in images_iterator(files):
    database.append(desc)
    if iteration == 30000:
      break
  database = sanitize(np.concatenate(database))
  kmeans.train(database)
  kmeans.save(cur_dir)


if __name__ == '__main__':
  make_kmeans(128)
