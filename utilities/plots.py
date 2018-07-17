import numpy as np
import pickle
import matplotlib.pyplot as plt

def save_fig(fig, file_path):
    save_fig_pickle(fig, file_path + '.pklfig')
    plt.savefig(file_path + '.png')
    command = 'CALL activate tf-gpu\ncd C:\\Users\\USER\\PycharmProjects\\spectrum_analysis\npython open_fig.py -f '+ file_path + '.pklfig'

    with open(file_path+'.cmd' , 'w') as file:
        file.write(command)

def save_fig_pickle(fig, file_path):
    axes = fig.axes
    share_x_mat = np.zeros((len(axes), len(axes),), dtype=bool)
    share_y_mat = np.zeros((len(axes), len(axes),), dtype=bool)

    for i, ax1 in enumerate(axes):
        for j, ax2 in enumerate(axes):
            if ax2 in ax1.get_shared_x_axes().get_siblings(ax1):
                share_x_mat[i,j] = True
            if ax2 in ax1.get_shared_y_axes().get_siblings(ax1):
                share_y_mat[i,j] = True

    with open(file_path, 'wb') as file:
        pickle.dump([fig, share_x_mat, share_y_mat], file)

def load_fig_pickle(file_path):
    with open(file_path, 'rb') as file:
        fig, share_x_mat, share_y_mat = pickle.load(file)

    axes = fig.axes

    for i, ax1 in enumerate(axes):
        for j, ax2 in enumerate(axes):
            if share_x_mat[i,j] and not (ax2 in ax1.get_shared_x_axes().get_siblings(ax1)):
                ax1.get_shared_x_axes().join(ax1, ax2)
            if share_y_mat[i,j] and not (ax2 in ax1.get_shared_y_axes().get_siblings(ax1)):
                ax1.get_shared_y_axes().join(ax1, ax2)

    return fig
