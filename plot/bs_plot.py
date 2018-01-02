import os
from operator import itemgetter

import tensorflow as tf
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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
    print data[key]
  return data


def plot(data, title, x_name, y_name, legend_prefix, save_to):
  fig, ax = plt.subplots()

  for key in sorted(data.keys()):
    xs = [x for x, y in data[key]]
    ys = [y for x, y in data[key]]
    plt.plot(xs, ys, label=legend_prefix + str(key), ms=2, linestyle='--', linewidth=1)

  params = {'legend.fontsize': 14, 'legend.handlelength': 1}
  plt.rcParams.update(params)
  plt.legend(loc='upper right')
  #leg = plt.legend(keys, loc='upper right')
  # for legobj in leg.legendHandles:
  #  legobj.set_linewidth(3.0)
  plt.xlabel(x_name)
  plt.ylabel(y_name)
  plt.title(title)
  plt.savefig(save_to)

if __name__ == '__main__':
  keys = ['global_step/sec', 'losses/clone_0/softmax_cross_entropy_loss/value', 'total_loss_1']
  exp_dir = '/home/x/exp/slim'
  exp_filter = '318'
  data = {}
  for d in os.listdir(exp_dir):
    if not exp_filter in d: continue
    batch = re.search(r"(?<=bs)\S*(?=-cifar)", d).group()
    batch = int(batch)
    train_dir = os.path.join(exp_dir, d)
    events_files = [x for x in os.listdir(train_dir) if x.startswith('events')]
    data[batch] = read_event_data(os.path.join(train_dir, events_files[0]), keys)
  print(data.keys())

  for key in keys:
    data_for_key = {bs: data[bs][key] for bs in data}
    plot(data_for_key, key, 'steps', 'value', 'bs=', './figures/bs-%s.eps' % key.replace('/', ''))
