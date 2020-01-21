# -*- coding: utf-8 -*-
"""
Created on Nov 13 2019

Last Update Nov 13 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

Plots resukt of single run

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_data(dataStruc, FieldName, type='lin'):
    # linear plot
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    # log plot
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)

    # set x-label
    plt.xlabel("time")
    #maxTData = np.nanmax(dataStruc['time'])
    #plt.xlim((0, maxTData))

    return None


def plot_heatmap(fig, axs, data, yName, type='lin'):
    # linear plot
    if type == 'lin':
        currData = data.transpose()
        labelName = "density"
        cRange = [0, 0.1]
    # log plot
    elif type == 'log':
        currData = np.log10(
            data.transpose() + np.finfo(float).eps)
        labelName = "log10 density"
        cRange = [-2, -1]

    im = axs.imshow(currData, cmap="viridis",
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = cRange[0],
                    vmax = cRange[1],
                    aspect='auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel(yName)
    axs.set_xlabel('time')
    fig.colorbar(im, ax=axs, orientation='vertical',
                fraction=.1, label=labelName)
    axs.set_yticklabels([0, 1])

    return None

# run model, plot dynamics


def plot_single_run(model_par, output, distFCoop, distGrSize):
    # setup figure formatting
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 6}
    matplotlib.rc('font', **font)

    # open figure
    fig = plt.figure()
    nR = 3
    nC = 2

    # plot number of groups
    plt.subplot(nR, nC, 1)
    plot_data(output, "NGroup")
    plot_data(output, "NGroup_mav")

    plt.ylabel("# group")
    plt.legend()

    # plot number of cells
    plt.subplot(nR, nC, 2)
    for tt in range(int(model_par['indv_NType'])):
        plot_data(output, 'N%i' % tt)
        plot_data(output, 'N%imut' % tt)
    plt.ylabel("# cell")
    plt.legend()

    # plot fraction of coop
    plt.subplot(nR, nC, 3)
    plot_data(output, "NCoop")
    plot_data(output, "NCoop_mav")
    plt.ylabel("density cooperator")
    plt.legend()

    # plot rms error
    plt.subplot(nR, nC, 4)
    plot_data(output, "rms_err_NCoop",type='log')
    plot_data(output, "rms_err_NGroup",type='log')
    plt.legend()
    plt.ylabel("rms error")

    #plot distribution group size
    axs = plt.subplot(nR, nC, 5)
    plot_heatmap(fig, axs, distGrSize, 'group size', type='lin')

    #plot distribution fraction coop
    axs = plt.subplot(nR, nC, 6)
    plot_heatmap(fig, axs, distFCoop, 'coop. freq.', type='lin')

    # set figure size
    fig.set_size_inches(4, 6)
    plt.tight_layout()  # cleans up figure and aligns things nicely

    return None
