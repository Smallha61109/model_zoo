import tensorflow as tf
import os
import logging
from argparse import ArgumentParser

from kinect_pose import KinectPoseModel
from kinect_pose_prepare import FreiburgData


def main():
  logging.basicConfig()
  logger = logging.getLogger('train')
  logger.setLevel(logging.INFO)
  parser = ArgumentParser()
  parser.add_argument('--db', dest='db', default='freiburg1_xyz.sqlite3',
    help='input database')
  parser.add_argument('--batch-size', type=int, dest='batch_size',
    default=256, help='batch size for training')
  parser.add_argument('--max-epoch', type=int, dest='max_epoch',
    default=10000, help='max epoch for training')
  parser.add_argument('--output-epoch', type=int, dest='output_epoch',
    default=10, help='epoch for display and saving')
  parser.add_argument('--keep-prob', type=float, dest='keep_prob',
    default=0.8, help='keep probability for drop out')
  parser.add_argument('--decay-epoch', type=int, dest='decay_epoch',
    default=1000, help='decay learning rate epoch')
  parser.add_argument('--input-type', dest='input_type',default='rgbd',
    help='type of input(channels)')

  args = parser.parse_args()
  data_loader = FreiburgData(args.db)

  input_width = 320
  input_height = 240
  if args.input_type == 'depth':
    input_channel = 1
    logger.info('input type: depth')
  elif args.input_type == 'rgb':
    input_channel = 3
    logger.info('input type: rgb')
  elif args.input_type == 'rgbd':
    input_channel = 4
    logger.info('input type: rgbd')
  output_size = 3

  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, kernel_size=3, model_name='KinectPose', saving=True)

  with tf.Session() as sess:
    model.train_with_loader(sess, data_loader,
      batch_size=args.batch_size,
      output_period=args.output_epoch,
      decay_epoch=args.decay_epoch,
      keep_prob=args.keep_prob,
      max_epoch=args.max_epoch,
      input_type=args.input_type)


if __name__ == '__main__':
  main()
