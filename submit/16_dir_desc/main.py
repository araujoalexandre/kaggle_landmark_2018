
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

kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
valid_folder = join(kaggle_folder, 'validation')
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'


class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means

    def prepare_image_and_grid_regions_for_network(self, im, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(im, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def get_rmac_features_batch(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, im, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


class KaggleRetrievalTools:

  def _print_status(self, iteration, num_images):
    # print status
    if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
      elapsed = (datetime.now() - self.start).total_seconds()
      print('Loading image {} out of {}, last {} images took {:.5f}'
          ' seconds'.format(iteration, num_images, 
            self._STATUS_CHECK_ITERATIONS, elapsed), flush=True)
      self.start = datetime.now()

  def _sanitize(self, data):
    """ convert array to a c-contiguous float array """
    if data.dtype != 'float32':
      data = data.astype('float32')
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguousarray(data)
    return data

  def _make_vlad_by_batch(self, x, n_desc, predicted):
    desc_cumsum = np.cumsum(n_desc)
    img_indexes = list(zip([0] + list(desc_cumsum[:-1]), desc_cumsum))
    database = np.zeros((len(n_desc), self.ncentroids * self.dim))
    for ix, (start_ix, end_ix) in enumerate(img_indexes):
      x_subset = x[start_ix:end_ix]
      predicted_susbset = predicted[start_ix:end_ix]
      database[ix, :] = self._make_vlad(x_subset, predicted_susbset)
    database = self._sanitize(database)
    return database

  def _make_vlad(self, x, predicted):
    predicted = predicted.flatten()
    x = x - self.kmeans.centroids[predicted]
    x /= np.linalg.norm(x, ord=1, axis=1)[..., np.newaxis]
    vlad = np.zeros((self.ncentroids, self.dim))
    for cluster in range(self.ncentroids):
      mask = predicted == cluster
      if np.sum(mask):
        vlad[cluster, :] = x[mask, :].sum(axis=0)
    # reshape
    vlad = vlad.reshape(self.ncentroids * self.dim)
    # power normalization, also called square-rooting normalization
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    # L2 normalization
    vlad = vlad / np.linalg.norm(vlad)
    vlad = vlad[np.newaxis]
    return vlad


class KaggleRetrievalDebug:

  def _init_debug_mode(self):

    self.train_files = self.train_files[:500]
    self.test_files = self.test_files[:10]
    self.nimg_train = len(self.train_files)
    self.nimg_test = len(self.test_files)
    
    self._STATUS_CHECK_ITERATIONS = 1000
    self._make_results = self._make_results_debug
    
    # init index
    self._try_load_index()
    
  def _make_results_debug(self, predicted):
    results = {}
    for query_ix, query_result in enumerate(predicted):
      res = []
      for img_ix in query_result:
        if img_ix != -1:
          try:
            filename = self.train_filenames[img_ix]
            res.append(filename)
          except IndexError:
            print('img ix', img_ix)
      try:
        query_filename = self.test_filenames[query_ix]
      except IndexError:
        print('query_ix', query_ix)
      results[query_filename] = ' '.join(res)
    return results



class KaggleRetrievalValidation:

  def _init_validation(self):
    self.target_dict = pickle_load(join(valid_folder, 'target.pkl'))
    self.target_train = pickle_load(join(valid_folder, 'target_train.pkl'))

    self.train_files = glob.glob(join(valid_folder, 'train', '*'))
    self.test_files = glob.glob(join(valid_folder, 'test', '*'))

    self._STATUS_CHECK_ITERATIONS = 1000

    self.nimg_train = len(glob.glob(join(valid_folder, 'train', '*')))
    self.nimg_test = len(glob.glob(join(valid_folder, 'test', '*')))

    # init index
    self._try_load_index()

  def map100metric(self, predicted, actual):

    actual = actual.reshape(-1, 1)
    metric = 0.
    for i in range(100):
        metric += np.sum(actual == predicted[:, i]) / (i + 1)
    metric /= actual.shape[0]
    return metric

  def predict_validation(self, query_database):
    print('compute score', flush=True)
    k = 100
    predicted = self.index.search(query_database, k)[1]

    qe = 3
    if qe > 0:
      print('query extension : {}'.format(qe))
      norm = 1
      for i, pred in enumerate(predicted):
        for index in pred[:qe]:
          query_database[i] += self._get_feat_qe(index)
          norm += 1
        query_database[i] /= norm
      predicted = self.index.search(query_database, k)[1]

    
    target = []
    for i, query in enumerate(query_database):
      target.append(self.target_dict[self.test_filenames[i]])
    target = np.array(target)

    predicted_map = np.zeros_like(predicted)
    for i, j in np.ndindex(predicted.shape):
      if predicted[i, j] == -1:
        predicted_map[i, j] = -1
        continue
      name = self.train_filenames[predicted[i, j]]
      predicted_map[i, j] = self.target_train[name]

    score = self.map100metric(predicted_map, target)
    print('map100 = {}'.format(score))


class KaggleRetrieval(KaggleRetrievalTools, KaggleRetrievalDebug, 
                        KaggleRetrievalValidation):

  def __init__(self, debug=False, gpu=False, valid=True, qe=0):

    self.cur_dir = dirname(realpath(__file__))
    self.gpu = gpu
    self.qe = qe

    self.images_folder = join(kaggle_folder, 'img')
    self.sample_submission = join(kaggle_folder, 'data', 
      'sample_submission.csv')
    
    self.train_files = glob.glob(join(storage_folder, 'index_rescale', '*'))
    self.test_files = glob.glob(join(storage_folder, 'test_rescale', '*'))

    means = np.array([103.93900299,  116.77899933,  123.68000031], 
              dtype=np.float32)[None, :, None, None]
    self.img_helper = ImageHelper(512, 2, means)

    self.train_filenames = []
    self.test_filenames = []

    # caffe
    proto = 'deploy_resnet101_normpython.prototxt'
    weights  = 'model.caffemodel'

    # Configure caffe and load the network
    caffe.set_device(0)
    caffe.set_mode_gpu()
    self.net = caffe.Net(proto, weights, caffe.TEST)

    if debug:
      self._init_debug_mode()
    elif valid:
      self._init_validation()
      self.predict = self.predict_validation
    else:
      self._init()

  def _init(self):

    self._STATUS_CHECK_ITERATIONS = 10000

    self.nimg_train = len(self.train_files)
    self.nimg_test = len(self.test_files)

    # init index
    self._try_load_index()

  def _try_load_index(self):
    # check if index.dump exists
    index_checkpoint = join(self.cur_dir, 'index_full.dump')
    
    # try to load full index first
    if exists(index_checkpoint):
      print('index checkpoint found, loading database not necessary')
      self.index = faiss.read_index(index_checkpoint)
      self.index_full_trained = True
      # if index_full exists filename should also exists
      self.train_filenames = pickle_load(join(self.cur_dir, 
                  'filenames_index.dump'))
    
    # init new index
    else:
      self.index_trained = False
      self.index_full_trained = False
      self.index = faiss.IndexFlatL2(2048)

  def load_train(self):

    if self.index_full_trained:
      print('index already trained', flush=True)
      return

    self.start = datetime.now()
    self.train_filenames = [None] * self.nimg_train
    xindexes = np.zeros((self.nimg_train, 2048))

    for i, path in tqdm(list(enumerate(self.train_files)), 
              file=sys.stdout, leave=False, dynamic_ncols=True):
      filename = splitext(basename(path))[0]
      im = cv2.imread(path)
      I, R = self.img_helper.prepare_image_and_grid_regions_for_network(im, roi=None)
      xindexes[i, :] += self.img_helper.get_rmac_features(I, R, self.net)
      self.train_filenames[i] = filename
    xindexes /= np.sqrt((xindexes * xindexes).sum(axis=1))[:, None]


    print('computing database done, saving index', flush=True)
    self.index.add(self._sanitize(xindexes))
    faiss.write_index(self.index, join(self.cur_dir, 'index_full.dump'))
    self.index_full_trained = True

    print('saving filenames index', flush=True)
    pickle_dump(self.train_filenames, 'filenames_index.dump')

  def load_test(self):
    print('query expension : {}'.format(self.qe))

    if exists('xqueries.pkl') and exists('test_filenames.pkl'):
      print('xqueries already saved', flush=True)
      self.test_filenames = pickle_load('test_filenames.pkl')
      return pickle_load('xqueries.pkl')

    self.start = datetime.now()
    self.test_filenames = [None] * self.nimg_test
    xqueries = np.zeros((self.nimg_test, 2048))

    for i, img_path in tqdm(list(enumerate(self.test_files)), file=sys.stdout, leave=False, dynamic_ncols=True):
      self.test_filenames[i] = splitext(basename(img_path))[0]
      im = cv2.imread(img_path)
      I, R = self.img_helper.prepare_image_and_grid_regions_for_network(im, roi=None)
      xqueries[i, :] += self.img_helper.get_rmac_features(I, R, self.net)
    xqueries /= np.sqrt((xqueries * xqueries).sum(axis=1))[:, None]
    # xqueries = self._sanitize(xqueries)
    pickle_dump(xqueries, 'xqueries.pkl')
    pickle_dump(self.test_filenames, 'test_filenames.pkl')
    return xqueries

  def _get_feat_qe(self, index):
    path = self.train_files[index]
    im = cv2.imread(path)
    I, R = self.img_helper.prepare_image_and_grid_regions_for_network(im, roi=None)
    return self.img_helper.get_rmac_features(I, R, self.net)

  def _make_results(self, predicted):
    results = {}
    for query_ix, query_result in enumerate(predicted):
      results[self.test_filenames[query_ix]] = ' '.join(
        [self.train_filenames[ix] for ix in query_result if ix != -1])
    return results

  def predict(self, query_database):
    print('make prediction and write submission file', flush=True)
    k = 100
    query_database = self._sanitize(query_database)

    if exists('predicted.pkl'):
      print('predicted already saved', flush=True)
      predicted = pickle_load('predicted.pkl')
    else:
      predicted = self.index.search(query_database, k)[1]
      pickle_dump(predicted, 'predicted.pkl')
    
    qe = self.qe
    if qe > 0:
      print('query extension : {}'.format(qe))
      norm = 1
      for i, pred in tqdm(list(enumerate(predicted)), file=sys.stdout, leave=False, dynamic_ncols=True):
        for index in pred[:qe]:
          query_database[i] += self._get_feat_qe(index)
          norm += 1
        query_database[i] /= norm
      predicted = self.index.search(query_database, k)[1]

    results = self._make_results(predicted)

    submit = pd.read_csv(self.sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))
    submit_filename = 'submit_{}_dir_full_index_qe_{}.csv.gz'.format(
                datetime.now().strftime('%Y-%m-%d_%H.%M.%S'), qe)
    submit.to_csv(submit_filename, index=False, compression='gzip')



def main():
  
  parser = argparse.ArgumentParser(description="KaggleRetrieval")
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--gpu",  type=bool, default=False)
  parser.add_argument("--valid",  type=bool, default=False)
  parser.add_argument("--qe",  type=int)
  args = parser.parse_args()

  kaggle = KaggleRetrieval(debug=args.debug, gpu=args.gpu, valid=args.valid, qe=args.qe)
  kaggle.load_train()
  query_database = kaggle.load_test()
  # predict and make submit
  kaggle.predict(query_database)
  print('done', flush=True)


if __name__ == '__main__':
  main()
