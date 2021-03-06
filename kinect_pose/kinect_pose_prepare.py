import logging
import os
import numpy as np
import cv2
import sqlite3
import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig()
logger = logging.getLogger('kinect_pose_prepare')
logger.setLevel(logging.INFO)


def load_label(path):
  data = []
  if os.path.isfile(path):
    logger.info('loading %s' % (path))
    with open(path, 'r') as f:
      while True:
        line = f.readline()
        if not line: break
        if line.startswith('#'): continue
        data.append([float(d) for d in line.split(' ')])
  else:
    logger.info('%s does not exists.' % (path))
  return np.array(data)


def load_all_images(folder, flag):
  images = []
  if os.path.isdir(folder):
    logger.info('loading images from %s' % (folder))
    images_path = sorted(os.listdir(folder))
    counter = 0
    for image_name in images_path:
      counter += 1
      if counter % 5 == 0:
        print('for test: %s' % image_name)
      else:
        print(image_name)
        image_path = os.path.join(folder, image_name)
        img = cv2.imread(image_path, flag)
        img = cv2.resize(img, (320, 240))
        images.append([float(image_name[:-4]), img])
  return images

def load_test_images(folder, flag):
  images = []
  if os.path.isdir(folder):
    logger.info('loading images from %s' % (folder))
    images_path = sorted(os.listdir(folder))
    counter = 0
    for image_name in images_path:
      counter += 1
      if counter % 5 == 0:
        print('for test: %s' % image_name)
        image_path = os.path.join(folder, image_name)
        img = cv2.imread(image_path, flag)
        img = cv2.resize(img, (320, 240))
        images.append([float(image_name[:-4]), img])
      #  else:
      #    print(image_name)
  return images


def find_label(timestamp, label):
  minval = 30
  l = None
  for i in range(len(label)):
    val = np.abs(timestamp - label[i, 0])
    if val < minval:
      minval = val
      l = label[i, :]
  return l


def associate_labels(images, label):
  logger.info('associating labels...')
  data = []
  asso_label = []
  for i in range(len(images)):
    timestamp = images[i][0]
    l = find_label(timestamp, label)
    #  print(timestamp, l[0])
    data.append(images[i][1])
    asso_label.append(l[1:])
  return np.array(data), np.array(asso_label)


def save(images, label, table_name, dbname):
  connection = sqlite3.connect(dbname)
  cursor = connection.cursor()
  cursor.execute("""CREATE TABLE IF NOT EXISTS %s(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    channel INTEGER NOT NULL,
    tx FLOAT NOT NULL,
    ty FLOAT NOT NULL,
    tz FLOAT NOT NULL,
    qx FLOAT NOT NULL,
    qy FLOAT NOT NULL,
    qz FLOAT NOT NULL,
    qw FLOAT NOT NULL);""" % (table_name))

  for i in range(len(images)):
    img = images[i, :]
    img = img.reshape(img.shape[0], img.shape[1], -1)
    l = label[i, :]
    cursor.execute("""INSERT INTO %s(image, width, height, channel,
      tx, ty, tz, qx, qy, qz, qw)
      VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" % (table_name),
      (buffer(img), img.shape[1], img.shape[0], img.shape[2],
      l[0], l[1], l[2], l[3], l[4], l[5], l[6],))
  connection.commit()
  connection.close()


class FreiburgData(object):
  def __init__(self, dbname):
    if os.path.isfile(dbname):
      logger.info('loading from %s...' % (dbname))
      connection = sqlite3.connect(dbname)
      cursor = connection.cursor()
      cursor.execute("""SELECT * FROM depth;""")
      depth_raw_data = cursor.fetchall()
      cursor.execute("""SELECT * FROM rgb;""")
      rgb_raw_data = cursor.fetchall()
      connection.close()

      logger.info('loading depth images...')
      self.depth_images, self.depth_labels = self._parse_depth(depth_raw_data,
          np.uint16)
      logger.info('loading rgb images...')
      self.rgb_images, self.rgb_labels = self._parse_rgb(rgb_raw_data, np.uint8)
      logger.info('depth images data shape: %s' %
        (str(self.depth_images.shape)))
      logger.info('depth labels data shape: %s' %
        (str(self.depth_labels.shape)))
      logger.info('rgb images data shape: %s' %
        (str(self.rgb_images.shape)))
      logger.info('rgb labels data shape: %s' %
        (str(self.rgb_labels.shape)))

  def _parse_depth(self, raw_data, dtype):
    images = []
    labels = []
    for i in range(len(raw_data)):
      d = raw_data[i]
      img = np.frombuffer(d[1], dtype=dtype).reshape(d[3], d[2], d[4])
      img = img / 5000.
      l = d[5:8]
      images.append(img)
      labels.append(l)
    logger.info('max depth value: %f' % np.amax(images))
    return np.array(images) /np.amax(images), np.array(labels) *100

  def _parse_rgb(self, raw_data, dtype):
    images = []
    labels = []
    for i in range(len(raw_data)):
      d = raw_data[i]
      img = np.frombuffer(d[1], dtype=dtype).reshape(d[3], d[2], d[4])
      img = img / 255.
      l = d[5:8]
      images.append(img)
      labels.append(l)
    return np.array(images), np.array(labels) *100

  def diff_depth_batch(self, batch_size, diff=10):
    data_size = len(self.depth_images)
    index1 = np.random.permutation(np.arange(data_size))
    index2 = np.round(diff * np.random.randn(data_size)).astype(np.int32)
    index2 = index1 + index2
    index2[index2 >= data_size] = data_size - 1
    index2[index2 < 0] = 0

    index1 = index1[:batch_size]
    index2 = index2[:batch_size]
    prev_images = self.depth_images[index1, :]
    next_images = self.depth_images[index2, :]
    prev_labels = self.depth_labels[index1, :]
    next_labels = self.depth_labels[index2, :]
    #  print(prev_labels, next_labels)
    labels = next_labels - prev_labels
    return prev_images, next_images, labels

  def diff_rgb_batch(self, batch_size, diff=10):
    base = 20
    data_size = len(self.rgb_images)-10
    index1 = np.random.permutation(np.arange(data_size))
    #  index1 = np.random.permutation(np.arange(data_size-(diff-base)/2))
    #  index2 = np.round(diff * np.random.randn(data_size)).astype(np.int32)
    index2 = diff
    #  index2 = np.round((diff-base) * np.random.randn(data_size-(diff-base)/2)+base).astype(np.int32)

    index2 = index1 + index2
    index2[index2 >= data_size] = data_size - 1
    index2[index2 < 0] = 0

    index1 = index1[:batch_size]
    index2 = index2[:batch_size]
    prev_images = self.rgb_images[index1, :]
    next_images = self.rgb_images[index2, :]
    prev_labels = self.rgb_labels[index1, :]
    next_labels = self.rgb_labels[index2, :]
    #  print(prev_labels, next_labels)
    labels = next_labels - prev_labels
    #  print(next_images)
    #  print(labels)
    return prev_images, next_images, labels

  def diff_rgbd_batch(self, batch_size, diff=50):
    base = 20
    data_size = len(self.rgb_images)
    #  index1 = np.random.permutation(np.arange(data_size))
    index1 = np.random.permutation(np.arange(data_size-(diff-base)/2))
    #  index2 = np.round(diff * np.random.randn(data_size)).astype(np.int32)
    index2 = np.round((diff-base) * np.random.randn(data_size-(diff-base)/2)+base).astype(np.int32)

    index2 = index1 + index2
    index2[index2 >= data_size] = data_size - 1
    index2[index2 < 0] = 0

    index1 = index1[:batch_size]
    index2 = index2[:batch_size]
    prev_rgb_images = self.rgb_images[index1, :]
    next_rgb_images = self.rgb_images[index2, :]
    prev_depth_images = self.depth_images[index1, :]
    next_depth_images = self.depth_images[index2, :]
    prev_labels = self.rgb_labels[index1, :]
    next_labels = self.rgb_labels[index2, :]
    prev_images = np.concatenate((prev_rgb_images, prev_depth_images), axis=3)
    next_images = np.concatenate((next_rgb_images, next_depth_images), axis=3)
    #  print(prev_labels, next_labels)
    labels = next_labels - prev_labels
    #  print(next_images)
    #  print(labels)
    return prev_images, next_images, labels

def main():
  parser = ArgumentParser()
  parser.add_argument('--depth-images', dest='depth',
    default='rgbd_dataset_freiburg1_xyz/depth',
    help='depth image folder')
  parser.add_argument('--rgb-images', dest='rgb',
    default='rgbd_dataset_freiburg1_xyz/rgb',
    help='rgb image folder')
  parser.add_argument('--label', dest='label',
    default='rgbd_dataset_freiburg1_xyz/groundtruth.txt',
    help='ground truth label')
  parser.add_argument('--output-db', dest='dbname',
    default='freiburg1_xyz.sqlite3',
    help='output db name')
  parser.add_argument('--output-test-db', dest='test_dbname',
    default='freiburg1_xyz_test.sqlite3',
    help='output test db name')
  parser.add_argument('--mode', dest='mode',
    default='create',
    help='mode (test/create)')
  args = parser.parse_args()

  if args.mode == 'test':
    data_loader = FreiburgData('freiburg1_xyz.sqlite3')
    p, n, l = data_loader.diff_depth_batch(256)
    #  l *= 1e3
    print(p.shape, n.shape, l.shape)
    print(l[:10, :])
  else:
    label = load_label(args.label)
    depth_data = load_all_images(args.depth, cv2.IMREAD_ANYDEPTH)
    rgb_data = load_all_images(args.rgb, cv2.IMREAD_COLOR)
    depth_test_data = load_test_images(args.depth, cv2.IMREAD_ANYDEPTH)
    rgb_test_data = load_test_images(args.rgb, cv2.IMREAD_COLOR)

    logger.info('depth_data[0]:')
    logger.info(depth_data[0])
    logger.info('rgb_data[0]:')
    logger.info(rgb_data[0])
    logger.info('rgb image count: %d' % len(rgb_data))
    logger.info('rgb test image count: %d' % len(rgb_test_data))

    depth_data, depth_label = associate_labels(depth_data, label)
    rgb_data, rgb_label = associate_labels(rgb_data, label)
    depth_test_data, depth_test_label = associate_labels(depth_test_data, label)
    rgb_test_data, rgb_test_label = associate_labels(rgb_test_data, label)

    logger.info('saving...')
    save(depth_data, depth_label, 'depth', args.dbname)
    save(rgb_data, rgb_label, 'rgb', args.dbname)
    save(depth_test_data, depth_test_label, 'depth', args.test_dbname)
    save(rgb_test_data, rgb_test_label, 'rgb', args.test_dbname)


if __name__ == '__main__':
  main()
