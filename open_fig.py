import argparse
import sys
import os

import matplotlib.pyplot as plt
from utilities.plots import save_fig_pickle, load_fig_pickle, save_fig


parser = argparse.ArgumentParser()
parser.prog = 'Spectrum Anomaly Detection'
parser.description = 'Use this command parser for training or testing the anomaly detector'
parser.add_argument('-f', '--file', help='fig path')


namespace = parser.parse_args(sys.argv[1:])

file_path = namespace.file
f = load_fig_pickle(file_path)
plt.show()




