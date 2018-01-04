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

def read_event_data(event_file, keys, read_histo=False):
  data = {}
  for key in keys: data[key] = []
  for event in tf.train.summary_iterator(event_file):
    for v in event.summary.value:
      if v.tag in keys:
        data[v.tag].append((event.step, v.simple_value if not read_histo else v.histo))

  for key in data:
    data[key] = sorted(data[key], key=itemgetter(0))
  return data


def plot(data, title, x_name, y_name, legend_prefix, save_to, x_limit=None, y_limit=None):
  fig, ax = plt.subplots()
  colors = ['b', 'g', 'r', 'y', 'm']
  # colors = ['tab]
  markers = ['.', 'o', 'v', '^', 'd', '*', '+']
  line_count = 0
  # fig, axarr = plt.subplots(ncols=2, sharey=True, figsize=plt.figaspect(3./4.))
  for idx, key in enumerate(sorted(data.keys())):
    # if idx % 2 == 0: continue
    xs = [x for x, y in data[key]]
    ys = [y for x, y in data[key]]
    sampling_interval = max(1, len(xs) / 200)
    xs = xs[::sampling_interval]
    ys = ys[::sampling_interval]
    window = 5
    xs = np.convolve(xs, np.ones(window) / window, mode='valid')
    ys = np.convolve(ys, np.ones(window) / window, mode='valid')
    plt.plot(xs, ys, label=legend_prefix + "%.1E" % key, ms=1,
              marker=markers[idx], linestyle='-', linewidth=1, color=colors[idx])
    line_count += 1

  params = {'legend.fontsize': 14, 'legend.handlelength': 1}
  plt.rcParams.update(params)
  legends = plt.legend(loc='best')
  for legobj in legends.legendHandles:
    legobj.set_linewidth(3.0)
  if y_limit: plt.ylim(y_limit)
  plt.xlabel(x_name)
  plt.ylabel(y_name)
  plt.xlabel(x_name)
  plt.title(title, y=0.99)
  plt.savefig(save_to, dpi=300)


if __name__ == '__main__':
  keys_train = {
    'losses/clone_0/softmax_cross_entropy_loss/value': {
      'title': 'Cross Entropy Loss of Training', 'y_limit': (0.5, 1.6)},
    # 'total_loss_1': {'title': 'Total Training Loss', 'y_limit': (0.5, 2)}
    }
  keys_test = {
    'eval/Accuracy': {'title': 'Evaluation Accuracy', 'y_name': 'Accuracy', 'y_limit': (0.6, 0.80)},
    'eval/ValLoss': {'title': 'Evaluation Cross Entropy Loss', 'y_name': 'Loss', 'y_limit': (0.6, 1.2)}
  }

  keys_hist = {
    'all': {'title': 'Weights Distribution'}
  }
  # exp_dir = '/hdd/x/exp/'
  exp_dir = '/home/x/exp/slim/'
  exp_filter = '118'
  data_train = {}
  data_test = {}
  data_hist = {}
  for d in os.listdir(exp_dir):
    if not exp_filter in d: continue
    wd = re.search(r"(?<=wd-)[0-9\\.]*(?=-)", d).group()
    wd = float(wd)
    if wd in [1.0, 0.1]: continue
    train_dir = os.path.join(exp_dir, d)
    events_files_train = [x for x in os.listdir(train_dir) if x.startswith('events')]
    data_train[wd] = read_event_data(os.path.join(train_dir, events_files_train[0]), keys_train)

    data_hist[wd] = read_event_data(os.path.join(train_dir, events_files_train[0]), keys_hist, read_histo=True)

    test_dir = os.path.join(exp_dir, d, 'eval')
    events_files_test = [x for x in os.listdir(test_dir) if x.startswith('events')]
    data_test[wd] = read_event_data(os.path.join(test_dir, events_files_test[0]), keys_test)


  for (keys, data) in [(keys_train, data_train), (keys_test, data_test)]:
    for tag in keys:
      data_for_key = {wd: data[wd][tag] for wd in data}
      key = keys[tag]
      # for i in [0, 1]:
      plot(data_for_key, key['title'],
         x_name=key.get('x_name') or 'step',
         y_name=key.get('y_name') or 'value',
         legend_prefix='wd=',
         x_limit=key.get('x_limit'),
         y_limit=key.get('y_limit'),
         save_to='./figures/wd-%s%d.png' % (tag.replace('/', ''), 2))

  axis_n = len(data_hist)

  fig, axarr = plt.subplots(nrows=axis_n, ncols=1, sharex=True, figsize=plt.figaspect(3./4.))
  colors = ['b', 'g', 'r', 'y', 'm']
  x_range = (-0.2, 0.2)
  for idx, wd in enumerate(sorted(data_hist)):
    # print type(data_hist[wd]['all'][-1][1])
    # break
    print data_hist[wd]['all'][-1][0]
    xs = data_hist[wd]['all'][-1][1].bucket_limit
    weights = data_hist[wd]['all'][-1][1].bucket
    ax = axarr[idx]
    ax.set_xlim(x_range)
    ax.hist(xs, weights=weights, bins=xs[:-1], range=x_range, label="wd=%.1E" % wd, color=colors[idx],
            histtype='bar', density=False)
    ax.legend(loc='upper left')
  fig.tight_layout()
  fig.savefig('./figures/wd-hist.png', dpi=300)