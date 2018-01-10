import os
from operator import itemgetter

import tensorflow as tf
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

TRAIN_LOSS = 'cross_entropy_train'
TEST_LOSS = 'cross_entropy_test'
ACCURACY = 'accuracy'

def read_event_data(event_file, keys):
  data = {}
  for key in keys: data[key] = []
  for event in tf.train.summary_iterator(event_file):
    for v in event.summary.value:
      if v.tag in keys:
        data[v.tag].append((event.step, v.simple_value))

  for key in data:
    data[key] = sorted(data[key], key=itemgetter(0))
  return data


def plot(data, title, x_name, y_name, legend_prefix, save_to, x_limit=None, y_limit=None):
  fig, ax = plt.subplots()
  colors = ['b', 'g', 'r', 'y', 'c']
  # colors = ['tab]
  markers = ['.', 'o', 'v', '^', 'd', '*', '+']
  line_count = 0
  fig, axarr = plt.subplots(ncols=1, sharey=True, figsize=plt.figaspect(3./4.))
  for idx, key in enumerate(sorted(data.keys())):
    # if idx % 2 == 0: continue
    axis = axarr
    xs = [x for x, y in data[key]]
    ys = [y for x, y in data[key]]
    sampling_interval = max(1, len(xs) / 100)
    xs = xs[::sampling_interval]
    ys = ys[::sampling_interval]
    window = 5
    xs = np.convolve(xs, np.ones(window) / window, mode='valid')
    ys = np.convolve(ys, np.ones(window) / window, mode='valid')
    axis.plot(xs, ys, label=legend_prefix + str(key), ms=2,
              marker=markers[idx], linestyle='-', linewidth=1, color=colors[idx])
    line_count += 1

  ax1 = axarr
  params = {'legend.fontsize': 14, 'legend.handlelength': 1}
  # .rcParams.update(params)
  legends = axarr.legend(loc='best', markerscale=2.0)
  #leg = plt.legend(keys, loc='upper right')
  for legobj in legends.legendHandles:
    legobj.set_linewidth(1.0)
  if y_limit: ax1.set_ylim(y_limit)
  ax1.set_xlabel(x_name)
  ax1.set_ylabel(y_name)
  plt.title(title, y=0.99)
  fig.tight_layout()
  fig.savefig(save_to, dpi=300)


if __name__ == '__main__':
  keys_train = {
    'global_step/sec': {'title': 'Training Speed', 'y_name': 'step/sec'},
    'losses/clone_0/softmax_cross_entropy_loss/value': {
      'title': 'Cross Entropy Loss of Training', 'y_limit': (0, 3)},
    # 'total_loss_1': {'title': 'Total Training Loss', 'y_limit': (0.5, 2)}
    }
  keys_test = {
    'eval/Accuracy': {'title': 'Evaluation Accuracy', 'y_name': 'Accuracy', 'y_limit': (0.3, 1.0)},
    'eval/ValLoss': {'title': 'Evaluation Cross Entropy Loss', 'y_name': 'Loss', 'y_limit': (0, 3)}
  }
  exp_dir = '/hdd/x/exp/slim'
  exp_filter = '4041'
  val_for_plot = [0.5, 1.0, 1.5, 2.0]
  data_train = {}
  data_test = {}
  for d in os.listdir(exp_dir):
    if not exp_filter in d: continue
    stddev = re.search(r"(?<=var)[0-9\\.]*(?=-)", d).group()
    dp = re.search(r"(?<=dp)[0-9\\.]*(?=-)", d).group()

    stddev = float(stddev)
    dp = float(dp)
    if stddev not in val_for_plot: continue
    train_dir = os.path.join(exp_dir, d)
    events_files_train = [x for x in os.listdir(train_dir) if x.startswith('events')]
    data_train[stddev] = read_event_data(os.path.join(train_dir, events_files_train[0]), keys_train)

    test_dir = os.path.join(exp_dir, d, 'eval')
    events_files_test = [x for x in os.listdir(test_dir) if x.startswith('events')]
    data_test[stddev] = read_event_data(os.path.join(test_dir, events_files_test[0]), keys_test)

  for (keys, data) in [(keys_train, data_train), (keys_test, data_test)]:
    for tag in keys:
      data_for_key = {dp_rate: data[dp_rate][tag] for dp_rate in data}
      key = keys[tag]
      # for i in [0, 1]:
      var_for_plot = sorted(data_for_key.keys())[0::1]
      plot({k: v for k, v in data_for_key.iteritems() if k in var_for_plot}, key['title'],
           x_name=key.get('x_name') or 'step',
           y_name=key.get('y_name') or 'value',
           legend_prefix='drop_rate=',
           x_limit=key.get('x_limit'),
           y_limit=key.get('y_limit'),
           save_to='./figures/var-dp%.2f-%s-%s.png' % (dp, tag.replace('/', ''), exp_filter))
