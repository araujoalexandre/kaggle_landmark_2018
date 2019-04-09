
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import time
import gc
import copy
import argparse
import pickle
import tarfile
from functools import partial
from collections import defaultdict
from multiprocessing import Queue, JoinableQueue, Pool
from threading import Thread
from os.path import splitext, basename, join, isfile, dirname, realpath, exists
from datetime import datetime

from utils import pickle_dump, pickle_load, Timer

from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '3'

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
sys.path.append('/home/alexandrearaujo/caffe/python/')
import pandas as pd
import numpy as np
import faiss
import caffe
import cv2
from delf import feature_io

# custom class 
from kmeans import Kmeans


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'


def sanitize(data):
  """ convert array to a c-contiguous float array """
  if data.dtype != 'float32':
    data = data.astype('float32')
  if not data.flags['C_CONTIGUOUS']:
    data = np.ascontiguousarray(data)
  return data

def make_vlad(x, predicted, kmeans_centroids):
  ncentroids = 128
  dim = 40
  predicted = predicted.flatten()
  x = x - kmeans_centroids[predicted]
  x /= np.linalg.norm(x, ord=1, axis=1)[..., np.newaxis]
  vlad = np.zeros((ncentroids, dim))
  for cluster in range(ncentroids):
    mask = predicted == cluster
    if np.sum(mask):
      vlad[cluster, :] = x[mask, :].sum(axis=0)
  # reshape
  vlad = vlad.reshape(ncentroids * dim)
  # power normalization, also called square-rooting normalization
  vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
  # L2 normalization
  vlad = vlad / np.linalg.norm(vlad)
  vlad = vlad[np.newaxis]
  return vlad




queue = Queue(1000000)
index_dict = {}
total_processed = 0

class LoadThread(Thread):
    
  def __init__(self, files, ssd):
    super(LoadThread, self).__init__()
    self.files = files
    self.ssd = ssd

    if self.ssd:
      self.index_path = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge/tmp/feature_index_rescale_delf'
    else:
      self.index_path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_rescale_delf_pkl'

  def load(self, path):
    filename = splitext(basename(path))[0]
    if self.ssd:
      try:
        loc, _, desc, _, _ = feature_io.ReadFromFile(path)
      except Exception as error:
        print('problem with file {}. error : {}'.format(
                filename, error), flush=True)
        desc = np.zeros((1, 40))
      return desc
    else:
      try:
        loc, desc = pickle_load(path)
      except Exception as error:
        print('problem with file {}. error : {}'.format(
                filename, error), flush=True)
        desc = np.zeros((1, 40))
      return desc

  def run(self):
    global queue
    start = datetime.now()
    for iteration, img_id in enumerate(self.files):

      if iteration % 1000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('index images {}/{}, last 1000 images loaded in {:.3f}'.format(
          queue.qsize(), len(self.files), elapsed), flush=True)
        start = datetime.now()

      ext = 'delf' if self.ssd else 'pkl'
      path = '{}/{}.{}'.format(self.index_path, img_id, ext)
      desc = self.load(path)
      desc = sanitize(desc)
      queue.put((img_id, desc))

    queue.put((None, None))
    return

class ProcessThread(Thread):

  def __init__(self, index_files):
    super(ProcessThread, self).__init__()

    self.index_files = index_files

    self.ncentroids = 128
    self.dim = 40
    self.gpu = False
    self.cur_dir = dirname(realpath(__file__))

    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=100, verbose=True, 
            max_points_per_centroid=10000000, 
            gpu=self.gpu)
    self._try_load_kmeans()

  def _try_load_kmeans(self):
    # check if kmeans centroid already trained
    kmeans_checkpoint = join(self.cur_dir, 'kmeans.dump')
    if exists(kmeans_checkpoint):
      print('kmeans centroids found, kmeans training not necessary')
      self.kmeans.load(kmeans_checkpoint)

  def run(self):

    global index_dict
    global total_processed
    start = datetime.now()
    iteration = 0
    while True:

      filename, desc = queue.get()
      if filename is None:
        break
      predicted = self.kmeans.assign(desc)[1].flatten()
      index_dict[filename] = make_vlad(desc, predicted, self.kmeans.centroids)
      iteration += 1
      total_processed += 1

      if iteration % 1000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('index processed images {}/{}, last 1000 images loaded in {:.3f}'.format(
          total_processed, len(self.index_files), elapsed), flush=True)
        start = datetime.now()



queue_queries = Queue(200000)
queries_dict = {}
total_load = 0

class QueryLoadThread(Thread):

  def __init__(self, total, id_queries, images, ssd):
    super(QueryLoadThread, self).__init__()
    self.total = total
    self.id_queries = id_queries
    self.images = images
    self.ssd = ssd

    if self.ssd:
      self.test_path = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge/tmp/feature_test_rescale_delf'
    else:
      self.test_path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_test_rescale_delf_pkl'

  def query_to_desc(self, query_id):

    ext = 'delf' if self.ssd else 'pkl'
    path = '{}/{}.{}'.format(self.test_path, query_id, ext)

    if self.ssd:
      try:
        loc, _, desc, _, _ = feature_io.ReadFromFile(path)
      except Exception as error:
        print('problem with file {}. error : {}'.format(
                query_id, error), flush=True)
        desc = np.zeros((1, 40))
      return desc
    else:
      try:
        loc, desc = pickle_load(path)
      except Exception as error:
        print('problem with file {}. error : {}'.format(
                query_id, error), flush=True)
        desc = np.zeros((1, 40))
      return desc

  def run(self):
    global queue_queries
    global total_load
    start = datetime.now()
    for iteration, (img_id, preds) in enumerate(zip(self.id_queries, self.images)):

      if not exists('{}/{}.pkl'.format(self.test_path, img_id)):
        continue

      if iteration % 1000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('load images {}/{}, last 1000 images loaded in {:.3f}'.format(
          total_load, self.total, elapsed), flush=True)
        start = datetime.now()

      query_desc = sanitize(self.query_to_desc(img_id))
      queue_queries.put((img_id, query_desc, preds))
      total_load += 1
    queue_queries.put((None, None, None))


class QueryProcessThread(Thread):

  def __init__(self, total):
    super(QueryProcessThread, self).__init__()
    
    self.total = total

    self.ncentroids = 128
    self.dim = 40
    self.gpu = False
    self.cur_dir = dirname(realpath(__file__))

    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=100, verbose=True, 
            max_points_per_centroid=10000000, 
            gpu=self.gpu)
    self._try_load_kmeans()

  def _try_load_kmeans(self):
    # check if kmeans centroid already trained
    kmeans_checkpoint = join(self.cur_dir, 'kmeans.dump')
    if exists(kmeans_checkpoint):
      print('kmeans centroids found, kmeans training not necessary')
      self.kmeans.load(kmeans_checkpoint)

  def run(self):
    global queue_queries
    global queries_dict
    iteration = 0
    start = datetime.now()
    while True:

      filename, query_desc, preds = queue_queries.get()
      if filename is None:
        break

      predicted = self.kmeans.assign(query_desc)[1].flatten()
      query_vlad = make_vlad(query_desc, predicted, self.kmeans.centroids)
      query_vlad = sanitize(query_vlad)
      iteration += 1

      queries_dict[filename] = (query_vlad, preds)
      
      if iteration % 1000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('index processed images {}/{}, last 1000 images loaded in {:.3f}'.format(
          len(queries_dict.keys()), self.total, elapsed), flush=True)
        start = datetime.now()






class Reranking:

  def __init__(self, debug=False, gpu=False, nimg=5, ssd=False):

    self.ncentroids = 128
    self.dim = 40
    self.gpu = gpu
    self.debug = debug
    self.nimg = nimg
    self.ssd = ssd
    print('nimg to rerank : {}'.format(self.nimg))
    print('load on ssd : {}'.format(self.ssd))

    if self.ssd:
      self.index_path = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge/tmp/feature_index_rescale_delf'
      self.test_path = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge/tmp/feature_test_rescale_delf'
    else:
      self.index_path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_rescale_delf_pkl'
      self.test_path = '/media/hdd1/kaggle/landmark-retrieval-challenge/feature_test_rescale_delf_pkl'

    self.submit = pd.read_csv('submit_2018-04-17_15.13.32_dir_full_index_qe_2_0.424.csv.gz')

    def select_left(x):
        if isinstance(x, str):
            array = x.split(' ')
            return array[:self.nimg]
        return x

    def select_right(x):
        if isinstance(x, str):
            array = x.split(' ')
            return ' '.join(array[self.nimg:])
        return x
        
    self.submit['left'] = self.submit['images'].apply(select_left)
    self.submit['right'] = self.submit['images'].apply(select_right)
    self.right_dict = self.submit[['id', 'right']].dropna().set_index('id')['right'].to_dict()

    self.cur_dir = dirname(realpath(__file__))
    self.sample_submission = join(kaggle_folder, 'data', 'sample_submission.csv')

  def load_index_vlad(self):

    if exists('index_dict.pkl'):
      self.index_dict = pickle_load('index_dict.pkl')
      print('index dict already saved, lenght : {}'.format(
        len(self.index_dict.keys())), flush=True)
      return 

    index_files = set()
    for preds in self.submit['left'].values:
      if isinstance(preds, list):
        index_files.update(set(preds))
    print('number index files {}'.format(len(index_files)), flush=True)

    if self.debug:
      index_files = list(index_files)[:300]

    index_files = list(index_files)

    # I/O workload
    process = 3
    loaders = []
    for subset in np.array_split(index_files, process):
      loaders.append(LoadThread(subset, self.ssd))
    for loader in loaders:
      loader.start()

    process_treads = []
    for _ in range(process):
      process_treads.append(ProcessThread(index_files))
    for pthreads in process_treads:
      pthreads.start()
    for pthreads in process_treads:
      pthreads.join()

    self.index_dict = index_dict

    # save result
    pickle_dump(self.index_dict, 'index_dict.pkl')

  def run(self):

    queries = self.submit['id'].values
    predictions = self.submit['left'].values
    total = len(queries)

    if self.debug:
      queries = queries[:300]
      predictions = predictions[:300]

    results = {}
    default = np.zeros((1, self.ncentroids * self.dim))

    process = 3

    queries_split = np.array_split(queries, process)
    predictions_split = np.array_split(predictions, process)

    loaders = []
    for queries_subset, predictions_subset in zip(queries_split, predictions_split):
      loaders.append(QueryLoadThread(total, queries_subset, predictions_subset, self.ssd))
    for loader in loaders:
      loader.start()

    process_queries = []
    for _ in range(process):
      process_queries.append(QueryProcessThread(total))
    for p in process_queries:
      p.start()
    for p in process_queries:
      p.join()

    self.queries_dict = queries_dict
    total = len(self.queries_dict.keys())

    start = datetime.now()
    for iteration, (query_id, (query_vlad, preds)) in enumerate(self.queries_dict.items()):

      if iteration % 1000 == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('processed images {}/{}, last 1000 images loaded in {:.3f}'.format(
          iteration, total, elapsed), flush=True)
        start = datetime.now()

      # init new index
      index = faiss.IndexFlatL2(self.ncentroids * self.dim)
      filenames = []

      for pred_img in preds:
        vlad = self.index_dict.get(pred_img, default)
        index.add(sanitize(vlad))
        filenames.append(pred_img)

      if self.gpu:
        ## Search on GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        new_predicted = gpu_index.search(query_vlad, self.nimg)[1]
      else:
        # Search on CPU
        new_predicted = index.search(query_vlad, self.nimg)[1]
      new_predicted = np.squeeze(new_predicted)

      results[query_id] = ' '.join(
        [filenames[ix] for ix in new_predicted if ix != -1])

    submit = pd.read_csv(self.sample_submission)

    def merge(ix):
      return '{} {}'.format(results.get(ix, ''), self.right_dict.get(ix, ''))

    submit['images'] = submit['id'].apply(merge)
    submit_filename = 'submit_{}_dir_full_index_qe2_reranking.csv.gz'.format(
                datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
    submit.to_csv(submit_filename, index=False, compression='gzip')


def main():
  
  parser = argparse.ArgumentParser(description="KaggleRetrieval")
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--gpu",  type=bool, default=False)
  parser.add_argument("--nimg",  type=int, default=5)
  parser.add_argument("--ssd",  type=bool, default=False)
  args = parser.parse_args()

  ranking = Reranking(debug=args.debug, gpu=args.gpu, 
    nimg=args.nimg, ssd=args.ssd)
  ranking.load_index_vlad()
  ranking.run()
  print('done', flush=True)


if __name__ == '__main__':
  main()


