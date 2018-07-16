from utilities.config_handler import get_config

conf=get_config()

mode = conf['mode']

assert mode in ['development','production']
if mode == 'production':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectogram(fft,freqs,time):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(fft,aspect='auto', origin='upper', extent=[freqs[0], freqs[-1], time[0], time[-1]])
