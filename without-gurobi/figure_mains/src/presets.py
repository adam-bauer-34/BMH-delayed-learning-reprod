"""Plotting preset function.

Here I write a simple function which returns a set of matplotlib parameters and
two base file names to save figures.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
6.19.2023
"""

import os
import datetime

import matplotlib.pyplot as plt


def get_presets(markers=False):
    """Get plotting presets.

    Parameters
    ----------
    markers: bool (default=False)
        do you want a marker on each data point?

    Returns
    -------
    plotting_presents: dict
        plotting presets to be passed to plt.rcParams.update

    basefile: string
        filename to save figure to 'figs' folder
    """

    linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted'] * 2
    color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                  '#0072B2', '#D55E00', '#CC79A7']
    marker_list = ['o', 's', 'P', '+', 'D', 'v', '3', ',']

    if markers:
        plotting_presets = {'axes.linewidth': 3,
                    'axes.axisbelow': False,
                    'axes.edgecolor': 'black',
                    'axes.facecolor': 'None',
                    'axes.grid': False,
                    'axes.labelcolor': 'black',
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'axes.titlelocation': 'left',
                    'figure.facecolor': 'white',
                    'figure.figsize': (18,10),
                    'lines.solid_capstyle': 'round',
                    'lines.linewidth': 2.5,
                    'patch.edgecolor': 'w',
                    'patch.force_edgecolor': True,
                    'text.color': 'black',
                    'legend.frameon': False,
                    'xtick.bottom': True,
                    'xtick.major.width': 3,
                    'xtick.major.size': 6,
                    'xtick.color': 'black',
                    'xtick.top': False,
                    'ytick.color': 'black',
                    'ytick.direction': 'out',
                    'ytick.left': True,
                    'ytick.right': False,
                    'ytick.color': 'black',
                    'ytick.major.width': 3,
                    'ytick.major.size': 6,
                    'axes.prop_cycle': plt.cycler(linestyle=linestyle_list,
                                                  color=color_list,
                                                  marker=marker_list),
                    'font.size': 16,
                    'font.family': 'sans'}

    else:
        plotting_presets = {'axes.linewidth': 3,
                    'axes.axisbelow': False,
                    'axes.edgecolor': 'black',
                    'axes.facecolor': 'None',
                    'axes.grid': False,
                    'axes.labelcolor': 'black',
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'axes.titlelocation': 'left',
                    'figure.facecolor': 'white',
                    'figure.figsize': (18,10),
                    'lines.solid_capstyle': 'round',
                    'lines.linewidth': 2.5,
                    'patch.edgecolor': 'w',
                    'patch.force_edgecolor': True,
                    'text.color': 'black',
                    'legend.frameon': False,
                    'xtick.bottom': True,
                    'xtick.major.width': 3,
                    'xtick.major.size': 6,
                    'xtick.color': 'black',
                    'xtick.top': False,
                    'ytick.color': 'black',
                    'ytick.direction': 'out',
                    'ytick.left': True,
                    'ytick.right': False,
                    'ytick.color': 'black',
                    'ytick.major.width': 3,
                    'ytick.major.size': 6,
                    'axes.prop_cycle': plt.cycler(linestyle=linestyle_list,
                                                  color=color_list),
                    'font.size': 16,
                    'font.family': 'sans'}


    today = datetime.datetime.now()
    year = str(today.year)
    day = str(today.day)
    month = str(today.month)

    cwd = os.getcwd()
    basefile = ''.join([cwd, '/figs/', year, '-', month, '-', day, '-'])

    return plotting_presets, basefile
