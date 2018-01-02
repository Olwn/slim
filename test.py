import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from datasets import cifar10, dataset_factory
from tensorflow.contrib import slim

from nets import nets_factory
from preprocessing import preprocessing_factory

from keras.datasets import cifar10, mnist

"""
with tf.device('/cpu:0'):
  vector = tf.Variable([7., 7.], 'vector')

  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))

  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
  init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  optimizer.minimize(session)
  print session.run(loss)
"""
# data
tf.app.flags.DEFINE_string("dataset_name", "mnist", "dataset")
tf.app.flags.DEFINE_string("dataset_dir", "/tmp/mnist", "location of dataset")
tf.app.flags.DEFINE_string("dataset_split_name", "train", "split name")
tf.app.flags.DEFINE_string("num_preprocessing_threads", 1, "threads")
tf.app.flags.DEFINE_string("preprocessing_name", "lenet", "preprocess")
tf.app.flags.DEFINE_string("train_image_size", None, "")
# model
tf.app.flags.DEFINE_string("model_name", "lenet", "network name")
# train
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_integer("num_readers", 1, "reders")
tf.app.flags.DEFINE_float("weight_decay", 0.001, "weight decay")

FLAGS = tf.app.flags.FLAGS


def main(_):
  dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])

  network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=dataset.num_classes,
    weight_decay=FLAGS.weight_decay,
    is_training=True)

  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    FLAGS.preprocessing_name or FLAGS.model_name, is_training=True)
  train_image_size = FLAGS.train_image_size or network_fn.default_image_size
  image = image_preprocessing_fn(image, train_image_size, train_image_size)
  images, labels = tf.train.batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_preprocessing_threads,
      capacity=5 * FLAGS.batch_size)
  labels = slim.one_hot_encoding(labels, dataset.num_classes)
  batch_queue = slim.prefetch_queue.prefetch_queue([images, labels])

  images, labels = batch_queue.dequeue()
  logits, end_points = network_fn(images)
  loss = slim.losses.softmax_cross_entropy(logits, labels)
  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sci_optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # threads = tf.train.start_queue_runners()
    for i in xrange(200):
      sess.run(train_op)
      if i % 100 == 0:
        print sess.run(accuracy)
    print sess.run(loss)
    sci_optimizer.minimize(sess)
    print sess.run(loss)


if __name__ == '__main__':
  tf.app.run()