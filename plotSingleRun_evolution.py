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


def plot_heatmap(fig, axs, data):
    
    maxValue = np.max(data)
    im = axs.imshow(data, cmap="viridis",
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = 0,
                    vmax = maxValue,
                    aspect='auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel('fraction to offspring')
    axs.set_xlabel('offspring size')
    fig.colorbar(im, ax=axs, orientation='vertical',
                fraction=.1, label='prop of cells')
    axs.set_yticklabels([0, 1])

    return None

# run model, plot dynamics


def plot_single_run(model_par, output, traitDistr):
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
    plot_data(output, "NGroup_mav")
    plt.ylabel("# group")
    plt.legend()
    
    # plot fraction of coop
    plt.subplot(nR, nC, 2)
    plot_data(output, "NCoop_mav")
    plot_data(output, "NTot_mav")
    plt.ylabel("cell number")
    plt.legend()
    
    # plot fraction of coop
    plt.subplot(nR, nC, 3)
    plot_data(output, "offspr_size")
    plot_data(output, "offspr_frac")
    plot_data(output, "offspr_size_mav")
    plot_data(output, "offspr_frac_mav")
    plt.ylabel("traits")
    plt.legend()
    plt.ylim(0, 1)
    
    
    # plot fraction of coop
    plt.subplot(nR, nC, 4)
    plot_data(output, "groupSizeAv_mav")
    plot_data(output, "groupSizeMed_mav")
    plt.ylabel("GroupsSize")
    plt.legend()
    
    

    #plot distribution group size
    axs = plt.subplot(nR, nC, 5)
    plot_heatmap(fig, axs, traitDistr[0,:,:])

    #plot distribution fraction coop
    axs = plt.subplot(nR, nC, 6)
    plot_heatmap(fig, axs, traitDistr[-1,:,:])

    # set figure size
    fig.set_size_inches(4, 6)
    plt.tight_layout()  # cleans up figure and aligns things nicely

    return fig
