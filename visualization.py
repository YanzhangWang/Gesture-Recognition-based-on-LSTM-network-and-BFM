import numpy as np
import matplotlib.pyplot as plt

def update_plot(V_k, ax):
    """更新V矩阵可视化图表"""
    V_k_squeezed = np.squeeze(V_k)

    V_k_0 = V_k_squeezed[:, 0, :]
    V_k_1 = V_k_squeezed[:, 1, :]
    V_k_2 = V_k_squeezed[:, 2, :]
    V_k_3 = V_k_squeezed[:, 3, :]

    x = range(V_k_0.shape[0])

    ax[0, 0].cla()
    ax[0, 1].cla()
    ax[1, 0].cla()
    ax[1, 1].cla()

    ax[0, 0].plot(x, np.abs(V_k_0[:, 0]))
    ax[0, 1].plot(x, np.abs(V_k_0[:, 1]))
    ax[1, 0].plot(x, np.abs(V_k_1[:, 0]))
    ax[1, 1].plot(x, np.abs(V_k_1[:, 1]))

    ax[0, 0].set_xlabel('Sub-channel Index', fontsize=13)
    ax[0, 1].set_xlabel('Sub-channel Index', fontsize=13)
    ax[1, 0].set_xlabel('Sub-channel Index', fontsize=13)
    ax[1, 1].set_xlabel('Sub-channel Index', fontsize=13)

    ax[0, 0].set_ylabel(r'$\tilde{V}_{k}$ Magnitude', fontsize=12)
    ax[0, 1].set_ylabel(r'$\tilde{V}_{k}$ Magnitude', fontsize=12)
    ax[1, 0].set_ylabel(r'$\tilde{V}_{k}$ Magnitude', fontsize=12)
    ax[1, 1].set_ylabel(r'$\tilde{V}_{k}$ Magnitude', fontsize=12)

    ax[0, 0].set_title(r'$m=1$ & $n_{ss}=1$', fontsize=12)
    ax[0, 1].set_title(r'$m=2$ & $n_{ss}=1$', fontsize=12)
    ax[1, 0].set_title(r'$m=1$ & $n_{ss}=2$', fontsize=12)
    ax[1, 1].set_title(r'$m=2$ & $n_{ss}=2$', fontsize=12)

    ax[0, 0].tick_params(axis='y', labelsize=12)
    ax[0, 1].tick_params(axis='y', labelsize=12)
    ax[1, 0].tick_params(axis='y', labelsize=12)
    ax[1, 1].tick_params(axis='y', labelsize=12)

    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()

    xtick_locs = [0, 250, 500]
    xtick_labels = ['0', '250', '500']

    ytick_locs = [0.0, 0.50, 1]
    ytick_labels = ['0.00', '0.50', '1.00']

    ax[0, 0].set_xticks(xtick_locs)
    ax[0, 0].set_xticklabels(xtick_labels, fontsize=12)
    ax[0, 1].set_xticks(xtick_locs)
    ax[0, 1].set_xticklabels(xtick_labels, fontsize=12)
    ax[1, 0].set_xticks(xtick_locs)
    ax[1, 0].set_xticklabels(xtick_labels, fontsize=12)
    ax[1, 1].set_xticks(xtick_locs)
    ax[1, 1].set_xticklabels(xtick_labels, fontsize=12)

    ax[0, 0].set_yticks(ytick_locs)
    ax[0, 0].set_yticklabels(ytick_labels, fontsize=12)
    ax[0, 1].set_yticks(ytick_locs)
    ax[0, 1].set_yticklabels(ytick_labels, fontsize=12)
    ax[1, 0].set_yticks(ytick_locs)
    ax[1, 0].set_yticklabels(ytick_labels, fontsize=12)
    ax[1, 1].set_yticks(ytick_locs)
    ax[1, 1].set_yticklabels(ytick_labels, fontsize=12)

    plt.tight_layout()