
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

def make_vlad(x, predicted, kmeans, ncentroids):
  x = x - kmeans.centroids[predicted]
  vlad = np.zeros((ncentroids, 40))
  for cluster in range(ncentroids):
    mask = predicted == cluster
    if np.sum(mask):
      vlad[cluster, :] = x[mask, :].sum(axis=0)
  # reshape
  vlad = vlad.reshape(ncentroids * 40)
  # power normalization, also called square-rooting normalization
  vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
  # L2 normalization
  vlad = vlad / np.linalg.norm(vlad)
  return vlad

def make_dataset(ncentroids):

  cur_dir = dirname(realpath(__file__))
  index = glob.glob('/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_rescale_delf_pkl/*')
  test = glob.glob('/media/hdd1/kaggle/landmark-retrieval-challenge/feature_test_rescale_delf_pkl/*')

  nimg_train = len(index)
  nimg_test = len(test)

  # init kmeans
  kmeans = Kmeans(40, ncentroids, niter=100, verbose=True, max_points_per_centroid=10000000, gpu=False)
  kmeans.load('kmeans.dump')

  filenames = [splitext(basename(path))[0] for path in index]

  with open('xindex.csv', 'a') as xindex:
    start = datetime.now()
    for iteration, filename, desc in images_iterator(index):
      if iteration % 10000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('last 10000 images loaded in {:.3f}'.format(elapsed), flush=True)
        start = datetime.now()
      desc = sanitize(desc)
      predicted = kmeans.assign(desc)[1].flatten()
      vlad = make_vlad(desc, predicted, kmeans, ncentroids)
      xindex.write('{},{}\n'.format(filename, ','.join(map(str, vlad))))

  with open('xtest.csv', 'a') as xtest:
    for iteration, filename, desc in images_iterator(test):
      if iteration % 10000 == 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('last 10000 images loaded in {:.3f}'.format(elapsed), flush=True)
        start = datetime.now()
      desc = sanitize(desc)
      predicted = kmeans.assign(desc)[1].flatten()
      vlad = make_vlad(desc, predicted, kmeans, ncentroids)
      xtest.write('{},{}\n'.format(filename, ','.join(map(str, vlad))))


if __name__ == '__main__':
  make_dataset(128)
