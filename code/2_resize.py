
import sys
from os import mkdir
from os.path import exists, join
from functools import partial
import multiprocessing
import glob

import PIL
from PIL import Image
import pandas as pd
import numpy as np

# _BATCH_SIZE = 20000
N_PROCESS = 12

def resize_image(image, target_size=512):
  def calc_by_ratio(a, b):
    return int(a * target_size / float(b))
  size = image.size
  if size[0] < size[1]:
    w = calc_by_ratio(size[0], size[1])
    h = target_size
  else:
    w = target_size
    h = calc_by_ratio(size[1], size[0])
  image = image.resize((w, h), Image.ANTIALIAS)
  return image

def worker(list_images, out):
  """
      path : path to the image to resize
      out : path to the folder to save the resized image
  """
  for path in list_images:
    filename = path.split('/')[-1]
    output = join(out, filename)
    if exists(output):
      continue
    try:
      img = resize_image(Image.open(path))
      img.save(output)
    except Exception as e:
      print('{} not working'.format(filename), e)
    continue

def main():
  if len(sys.argv) != 3:
    print('Syntax: {} <input_dir/> <output_dir/>'.format(sys.argv[0]))
    sys.exit(0)
  (input_dir, out_dir) = sys.argv[1:]

  if not exists(out_dir):
    mkdir(out_dir)

  list_images = glob.glob(join(input_dir, '*'))
  list_images_split = np.array_split(list_images, N_PROCESS)
  pool = multiprocessing.Pool(processes=N_PROCESS)
  pool.map(partial(worker, out=out_dir), list_images_split)

if __name__ == '__main__':
  main()
