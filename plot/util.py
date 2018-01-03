from matplotlib import pyplot as plt
import numpy as np


def plot(data, title, x_name, y_name, legend_prefix, save_to, x_limit=None, y_limit=None):
  fig, ax = plt.subplots()
  # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
  markers = ['.', 'o', 'v', '^', 'd', '*', '+']
  line_count = 0
  for idx, key in enumerate(sorted(data.keys())):
    # if idx % 2 == 0: continue
    xs = [x for x, y in data[key]]
    ys = [y for x, y in data[key]]
    sampling_interval = max(1, len(xs) / 100)
    xs = xs[::sampling_interval]
    ys = ys[::sampling_interval]
    window = 5
    xs = np.convolve(xs, np.ones(window) / window, mode='valid')
    ys = np.convolve(ys, np.ones(window) / window, mode='valid')
    plt.plot(xs, ys, label=legend_prefix + str(key), ms=1, marker=markers[line_count], linestyle='-', linewidth=1, color=colors[idx])
    line_count += 1

  params = {'legend.fontsize': 14, 'legend.handlelength': 1}
  plt.rcParams.update(params)
  legends = plt.legend(loc='best')
  #leg = plt.legend(keys, loc='upper right')
  for legobj in legends.legendHandles:
    legobj.set_linewidth(3.0)
  if x_limit: plt.xlim(x_limit)
  if y_limit: plt.ylim(y_limit)
  plt.xlabel(x_name)
  plt.ylabel(y_name)
  plt.title(title)
  plt.savefig(save_to, dpi=300)