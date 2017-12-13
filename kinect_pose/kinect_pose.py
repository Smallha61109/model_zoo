import tensorflow as tf
import numpy as np
import os
import time
import logging


class KinectPoseModel(object):
  def __init__(self, input_width, input_height, input_channel, output_size,
      model_name='KinectPoseModel', learning_rate=5e-5, decay=0.9,
      kernel_size=5, saving=True):
      #  model_name='KinectPoseModel', learning_rate=1e-3, decay=0.9, saving=True):
    # logger
    logging.basicConfig()
    self.logger = logging.getLogger(model_name)
    self.logger.setLevel(logging.INFO)
    self.logger.info('setting up model...')
    # model
    with tf.variable_scope(model_name):
      self.prev_images, self.next_images, self.labels, self.keep_prob, \
        self.outputs, self.loss, self.error, self.train_op = \
        self._build_model(input_width, input_height, input_channel,
        output_size, kernel_size, learning_rate, decay)
    # checkpoint
    self.start_epoch = 0
    self.saving = saving
    self.saver = tf.train.Saver()
    if saving:
      self.checkpoint_path, summary_path = self._prepare_save_dir(model_name)
      # saver
      self.logger.info('setting up saver...')
      # summary writer
      self.logger.info('setting up summary writer...')
      self.summary_writer = tf.summary.FileWriter(summary_path,
        tf.get_default_graph())
    self.merged_summary = tf.summary.merge_all()

  def _prepare_save_dir(self, model_name):
    index = 0
    while os.path.isdir(model_name + str(index)):
      index += 1
    model_path = model_name + str(index)
    self.logger.info('creating model directory %s...' % (model_path))
    os.mkdir(model_path)
    summary_path = os.path.join(model_path, 'summary')
    os.mkdir(summary_path)
    checkpoint_path = os.path.join(model_path, model_name)
    return checkpoint_path, summary_path

  def _build_model(self, input_width, input_height, input_channel,
    output_size, kernel_size, learning_rate, decay):
    # inputs
    prev_images = tf.placeholder(dtype=tf.float32,
      shape=[None, input_height, input_width, input_channel])
    tf.summary.image(name='prev_images', tensor=prev_images)
    next_images = tf.placeholder(dtype=tf.float32,
      shape=[None, input_height, input_width, input_channel])
    tf.summary.image(name='next_images', tensor=next_images)
    images = tf.concat([prev_images, next_images], axis=3)
    tf.summary.histogram(name='input',values=images)
    # labels
    labels = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
    tf.summary.histogram(name='labels0', values=labels[:,0])
    tf.summary.histogram(name='labels1', values=labels[:,1])
    tf.summary.histogram(name='labels2', values=labels[:,2])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=())
    # learning rate
    self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
      trainable=False)
    self.decay_lr = tf.assign(self.learning_rate, self.learning_rate * decay)
    tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
    # model
    #  regularization_term = []
    l2_loss = []

    devanish = 3.
    with tf.variable_scope('conv1'):
      h1_size = 32
      w = tf.get_variable(name='conv_w1',
        shape=[kernel_size, kernel_size, input_channel * 2, h1_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=1./6.*devanish))
        #  initializer=tf.random_normal_initializer(stddev=1./255./6.))
      tf.summary.histogram(name='w_conv1', values=w)
      b = tf.get_variable(name='conv_b1', shape=[h1_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(images, w, strides=[1, 2, 2, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv1', values=h)
      h = tf.nn.dropout(h, keep_prob)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w1),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv2'):
      h2_size = 64
      w = tf.get_variable(name='conv_w2',
        shape=[kernel_size, kernel_size, h1_size, h2_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h1_size*devanish))
      b = tf.get_variable(name='conv_b2', shape=[h2_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv2', values=h)
      h = tf.nn.dropout(h, keep_prob)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w2),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv3'):
      h3_size = 64
      w = tf.get_variable(name='conv_w3',
        shape=[kernel_size, kernel_size, h2_size, h3_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h2_size*devanish))
      b = tf.get_variable(name='conv_b3', shape=[h3_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 2, 2, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv3', values=h)
      h = tf.nn.dropout(h, keep_prob)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w3),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv4'):
      h4_size = 128
      w = tf.get_variable(name='conv_w4',
        shape=[kernel_size, kernel_size, h3_size, h4_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h3_size*devanish))
      b = tf.get_variable(name='conv_b4', shape=[h4_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv4', values=h)
      h = tf.nn.dropout(h, keep_prob)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w4),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv5'):
      h5_size = 128
      w = tf.get_variable(name='conv_w5',
        shape=[kernel_size, kernel_size, h4_size, h5_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h4_size*devanish))
      b = tf.get_variable(name='conv_b5', shape=[h5_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 2, 2, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv5', values=h)
      h = tf.nn.dropout(h, keep_prob)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w5),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv6'):
      h6_size = 256
      w = tf.get_variable(name='conv_w6',
        shape=[kernel_size, kernel_size, h5_size, h6_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h5_size*devanish))
      b = tf.get_variable(name='conv_b6', shape=[h6_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv6', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv7'):
      h7_size = 256
      w = tf.get_variable(name='conv_w7',
        shape=[kernel_size, kernel_size, h6_size, h7_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h6_size*devanish))
      b = tf.get_variable(name='conv_b7', shape=[h7_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 2, 2, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv7', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv8'):
      h8_size = 512
      w = tf.get_variable(name='conv_w8',
        shape=[kernel_size, kernel_size, h7_size, h8_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h7_size*devanish))
      b = tf.get_variable(name='conv_b8', shape=[h8_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv8', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv9'):
      h9_size = 512
      w = tf.get_variable(name='conv_w9',
        shape=[kernel_size, kernel_size, h8_size, h9_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h8_size*devanish))
      b = tf.get_variable(name='conv_b9', shape=[h9_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 2, 2, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv9', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv10'):
      h10_size = 1024
      w = tf.get_variable(name='conv_w10',
        shape=[kernel_size, kernel_size, h9_size, h10_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h9_size*devanish))
      b = tf.get_variable(name='conv_b10', shape=[h10_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv10', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('conv11'):
      h11_size = 1024
      w = tf.get_variable(name='conv_w11',
        shape=[kernel_size, kernel_size, h10_size, h11_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=2./h10_size*devanish))
      b = tf.get_variable(name='conv_b11', shape=[h11_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.nn.conv2d(h, w, strides=[1, 1, 1, 1],
        padding='SAME') + b)
      #  h = tf.nn.max_pool(h, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1],
      #    padding='SAME')
      tf.summary.histogram(name='h_conv11', values=h)
      h = tf.nn.dropout(h, keep_prob)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('fc12'):
      h_size = h.get_shape().as_list()
      self.logger.info('connect size: %s' % (str(h_size)))
      connect_size = h_size[1] * h_size[2] * h_size[3]
      h12_size = 512
      w = tf.get_variable(name='w10',
        shape=[connect_size, h12_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / connect_size*devanish)))
      b = tf.get_variable(name='b12', shape=[h12_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.matmul(tf.reshape(h, [-1, connect_size]), w) + b)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w),name='regu'))
      tf.summary.histogram(name='h_fc12', values=h)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('fc13'):
      h13_size = 512
      w = tf.get_variable(name='w13',
        shape=[h12_size, h13_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / h12_size*devanish)))
      b = tf.get_variable(name='b13', shape=[h13_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-1))
      h = tf.nn.relu(tf.matmul(h, w) + b)
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w),name='regu'))
      tf.summary.histogram(name='h_fc13', values=h)
      l2_loss.append(tf.nn.l2_loss(w))

    with tf.variable_scope('output'):
      w = tf.get_variable(name='ow', shape=[h13_size, output_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
          stddev=np.sqrt(1. / h13_size*devanish)))
      b = tf.get_variable(name='ob', shape=[output_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value=1e-2))
      logits = tf.matmul(h, w) + b
      self.outputs = logits
      #  regularization_term.append(tf.reduce_mean(input_tensor=tf.square(w),name='regu'))
      l2_loss.append(tf.nn.l2_loss(w))

      tf.summary.histogram(name='output0', values=self.outputs[:,0])
      tf.summary.histogram(name='output1', values=self.outputs[:,1])
      tf.summary.histogram(name='output2', values=self.outputs[:,2])

    # loss and optimizer
    with tf.variable_scope('loss'):
      # scale up to cm as unit
      #  loss = tf.reduce_mean(tf.square(logits - labels)) #* output_size * (10 **2)
      #  rt = tf.reduce_sum(regularization_term) * 2.
      #  tf.summary.histogram(name='regularization_term',values=rt)
      #  loss = tf.reduce_mean(tf.squared_difference(logits,labels)) + rt
      #  self.l2_regu = tf.reduce_sum(l2_loss) * 0.01
      #  tf.summary.scalar(name='l2_loss', tensor=l2_regu)
      loss = tf.reduce_mean(tf.squared_difference(logits, labels)) #+ self.l2_regu

      tf.summary.scalar(name='loss', tensor=loss)
      error = tf.reduce_mean(tf.abs(logits - labels))
      tf.summary.scalar(name='error', tensor=error)

    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      train_op = optimizer.minimize(loss)

    return prev_images, next_images, labels, keep_prob, self.outputs, \
        loss, error, train_op

  def load(self, sess, checkpoint_path):
    if os.path.isfile(checkpoint_path + '.meta') and \
        os.path.isfile(checkpoint_path + '.index'):
      self.logger.info('loading last %s checkpoint...' % (checkpoint_path))
      self.saver.restore(sess, checkpoint_path)
      self.start_epoch = int(checkpoint_path.split('-')[-1].strip())
    else:
      self.logger.warning('%s does not exists' % (checkpoint_path))

  def train(self, sess, prev_images, next_images, label,
      batch_size=1024, output_period=10,
      keep_prob=0.8, max_epoch=100000):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      # prepare batch data
      offset = (epoch * batch_size) % (prev_images.shape[0] - batch_size + 1)
      batch_prev = prev_images[offset:offset+batch_size, :]
      batch_next = next_images[offset:offset+batch_size, :]
      batch_label = label[offset:offset+batch_size, :]
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.prev_images: batch_prev,
        self.next_images: batch_next,
        self.labels: batch_label,
        self.keep_prob: keep_prob
      })
      # output
      if epoch % output_period == 0:
        feed_dict={
          self.prev_images: batch_prev,
          self.next_images: batch_next,
          self.labels: batch_label,
          self.keep_prob: 1.0
        }
        ms, loss, error, out = sess.run(
          [self.merged_summary, self.loss, self.error, self.outputs], feed_dict)
        self.logger.info('ZZ %d. loss: %f | error: %f | time used: %f' %
          (epoch, loss, error, (time.time() - last)))
        last = time.time()
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)

  def train_with_loader(self, sess, data_loader,
      batch_size=1024, output_period=10, decay_epoch=1000,
      keep_prob=0.8, max_epoch=100000, input_type='rgbd'):
    # initialize
    if self.start_epoch == 0:
      self.logger.info('initializing variables...')
      sess.run(tf.global_variables_initializer())
    # training
    self.logger.info('start training...')
    last = time.time()
    for epoch in range(self.start_epoch, self.start_epoch + max_epoch + 1):
      # prepare batch data
      if input_type == 'depth':
        p, n, l = data_loader.diff_depth_batch(batch_size)
      elif input_type == 'rgb':
        p, n, l = data_loader.diff_rgb_batch(batch_size)
      elif input_type == 'rgbd':
        p, n, l = data_loader.diff_rgbd_batch(batch_size)
      else:
        logger.info('wrong input type.')
      #  l_avg = sum(l)/float(len(l))
      #  print(l_avg)
      #  l_avg = sum(l_avg)/float(len(l_avg))
      #  print('label avg: %f' % l_avg)
      _, loss = sess.run([self.train_op, self.loss], feed_dict={
        self.prev_images: p,
        self.next_images: n,
        self.labels: l,
        self.keep_prob: keep_prob
      })
      # output
      if epoch % output_period == 0:
        feed_dict={
          self.prev_images: p,
          self.next_images: n,
          self.labels: l,
          self.keep_prob: 1.0
        }
        ms, loss, error, out = sess.run(
          [self.merged_summary, self.loss, self.error, self.outputs], feed_dict)
        self.logger.info('%d. loss: %f | error: %f | time used: %f' %
          (epoch, loss, error, (time.time() - last)))
        self.logger.info(l[:10])
        self.logger.info(out[:10])
        #  self.logger.info('--------------------')
        #  self.logger.info(l2)
        #  self.logger.info('--------------------')
        last = time.time()
        if self.saving:
          self.saver.save(sess, self.checkpoint_path, global_step=epoch)
          self.summary_writer.add_summary(ms, global_step=epoch)
      if epoch % decay_epoch == 0 and epoch != 0:
        self.logger.info('decay learning rate...')
        sess.run(self.decay_lr)

  def predict(self, sess, prev_images, next_images):
    return sess.run(self.outputs, feed_dict={
      self.prev_images: prev_images,
      self.next_images: next_images,
      self.keep_prob: 1.0
    })


def test():
  input_width = 640
  input_height = 480
  input_channel = 3
  output_size = 7

  # test data
  test_batch_size = 64
  data1 = np.random.randn(test_batch_size,
    input_height, input_width, input_channel)
  data2 = np.random.randn(test_batch_size,
    input_height, input_width, input_channel)
  label = np.zeros(shape=[test_batch_size, output_size])
  label[:, 0] = 1.0

  model = KinectPoseModel(input_width, input_height, input_channel,
    output_size, model_name='test', saving=False)

  with tf.Session() as sess:
    model.train(sess, data1, data2, label,
      batch_size=32,
      output_period=10,
      keep_prob=0.8,
      max_epoch=100)


if __name__ == '__main__':
  test()
