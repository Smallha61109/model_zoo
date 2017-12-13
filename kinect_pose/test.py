import os
import logging
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

from kinect_pose import KinectPoseModel
from kinect_pose_prepare import FreiburgData


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_seq(seq_images, checkpoint, labels, diff):
  input_width, input_height, input_channel = 320, 240, 3
  output_size = 3
  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, kernel_size=3, model_name='KinectPose', saving=False)

  poses = [labels[0, :]] #change if don't have init pose
  if os.path.isfile(checkpoint + '.index') and \
      os.path.isfile(checkpoint + '.meta'):
    with tf.Session() as sess:
      model.load(sess, checkpoint)

      counter = 0
      acc_all = 0
      for i in range(len(seq_images) - diff):
        counter += 1
        p = np.expand_dims(seq_images[i, :], axis=0)
        n = np.expand_dims(seq_images[i + diff, :], axis=0)
        prediction = np.squeeze(model.predict(sess, p, n), axis=0)
        if counter == 1:
          print('poses 0:')
          print(poses[0])
          predict_diff = prediction / diff
          for j in range(diff-1):
            poses.append(poses[j] + predict_diff)
            print('poses %d:' % (j+1))
            print(poses[j+1])
        poses.append(poses[-diff] + prediction)
        if counter <= 10:
          print('==============================')
          print('image %d & %d label:' % (i, i+diff))
          print(labels[i+diff]-labels[i])
          print('image %d & %d prediction:' % (i, i+diff))
          print(prediction)
          print('------------------------------')
          print('image %d pose:' % (i+diff))
          print(labels[i+diff])
          print('predicted image %d pose:' % (i+diff))
          print(poses[-1])
        labels0 = labels[i+diff][0]-labels[i][0]
        labels1 = labels[i+diff][1]-labels[i][1]
        labels2 = labels[i+diff][2]-labels[i][2]
        acc = abs(labels0-prediction[0])+\
              abs(labels1-prediction[1])+\
              abs(labels2-prediction[2])
        acc_all = acc_all + acc
  return np.array(poses), acc_all/(counter*3)


def main():
  parser = ArgumentParser()
  parser.add_argument('--input-db', dest='input_db',
    default='freiburg1_360_test.sqlite3',
    help='input dataset')
  parser.add_argument('--checkpoint', dest='checkpoint',
    default='saved_KP_320_240_3_diff10_c11/KinectPose-3400',
    help='checkpoint path')
  parser.add_argument('--diff', dest='diff', type=int,
    default='2',
    help='prediction frame diff')
  args = parser.parse_args()

  data = FreiburgData(args.input_db)
  logger.info(data.depth_images.shape)

  predicted_poses, acc = predict_seq(data.rgb_images, args.checkpoint,
    data.rgb_labels, diff=args.diff)
  print(acc)
  #  print(data.rgb_labels[0:10])
  #  print(predicted_poses[0:10])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(data.depth_labels[:, 0], data.depth_labels[:, 1],
    data.depth_labels[:, 2], 'g') #depth?????
  ax.plot(predicted_poses[:, 0], predicted_poses[:, 1],
    predicted_poses[:, 2], 'r')
  ax.set_xlabel('$X$')
  ax.set_ylabel('$Y$')
  plt.show()


if __name__ == '__main__':
  main()
