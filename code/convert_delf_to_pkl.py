
import os
import sys
import glob
import pickle
import multiprocessing
from functools import partial
from os.path import isfile, splitext, basename, join, exists

from delf import feature_io
import numpy as np

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


def worker(list_path, path_out=None):
    """
        path : path to the image to resize
        out : path to the folder to save the resized image
    """
    for iteration, path in enumerate(list_path):
        filename = splitext(basename(path))[0]
        if exists(join(path_out, filename + '.pkl')):
            continue
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
            continue

        obj = (loc, desc)
        pickle_dump(obj, join(path_out, filename + '.pkl'))


def main():
    if len(sys.argv) != 3:
        print('Syntax: {} <input_dir/> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (input_dir, out_dir) = sys.argv[1:]

    if not exists(out_dir):
        os.mkdir(out_dir)

    list_images = glob.glob(join(input_dir, '*'))
    worker(list_images, path_out=out_dir)

    # list_images_split = np.array_split(list_images, 2)
    # pool = multiprocessing.Pool(processes=2)
    # pool.map(partial(worker, path_out=out_dir), list_images_split)

if __name__ == '__main__':
  main()
