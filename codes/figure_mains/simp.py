"""Sectoral response of uncertainty.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
2.22.2024
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx

from src.presets import get_presets
from datatree import DataTree, open_datatree

# get presets from source function
presets, basefile = get_presets()
plt.rcParams.update(presets)

# data base path
cwd = os.getcwd()
data_head_path = ''.join([cwd, '/data/output/'])

# initial year for plotting
ti = 2020

# import data
data = open_datatree(data_head_path + 'simp_N1_T30_B3_GHQ_TRUNC_invrec_output.nc')
base = xr.open_dataset(data_head_path + 'simp_inv_output.nc')

mdata = open_datatree(data_head_path + 'simp_N1_T30_B3_GHQ_TRUNC_macrec_output.nc')
mbase = xr.open_dataset(data_head_path + 'simp_mac_output.nc')

tf = data['0'].ds.T
time = np.arange(0, tf, 1) + ti

inv_paths = np.zeros((len(data['1'].ds.state), len(time)), dtype=float)
abate_paths = np.zeros_like(inv_paths, dtype=float)
cum_paths = np.zeros_like(inv_paths, dtype=float)

minv_paths = np.zeros_like(inv_paths, dtype=float)
mabate_paths = np.zeros_like(inv_paths, dtype=float)
mcum_paths = np.zeros_like(inv_paths, dtype=float)

for state in data['1'].ds.state:
    learning_time_ind = np.where(time <= data['1'].ds.time.values[0] + ti)[0][-1]  # last t < t*
    inv_paths[state, :learning_time_ind] = data['0'].ds.investment.sel({'state': 0, 'sector': 'Widgets'}) 
    abate_paths[state, :learning_time_ind] = data['0'].ds.abatement.sel({'state': 0, 'sector': 'Widgets', 'time_state': np.arange(0, 30, 1)})
    cum_paths[state, :learning_time_ind] = data['0'].ds.cumulative_emissions.sel({'state': 0, 'time_state': np.arange(0, 30, 1)})

    inv_paths[state, learning_time_ind:] = data['1'].ds.investment.sel(sector='Widgets', state=state)
    abate_paths[state, learning_time_ind:] = data['1'].ds.abatement.sel(sector='Widgets', state=state, time_state=np.arange(30, tf, 1))
    cum_paths[state, learning_time_ind:] = data['1'].ds.cumulative_emissions.sel(state=state, time_state=np.arange(30, tf, 1))

    minv_paths[state, :learning_time_ind] = 0.5 * mbase.gbars.values[0] * mdata['0'].ds.abatement.sel({'state': 0, 'sector': 'Widgets'})**2
    mabate_paths[state, :learning_time_ind] = mdata['0'].ds.abatement.sel({'state': 0, 'sector': 'Widgets'})
    mcum_paths[state, :learning_time_ind] = mdata['0'].ds.cumulative_emissions.sel({'state': 0, 'time_state': np.arange(0, 30, 1)})

    minv_paths[state, learning_time_ind:] = 0.5 * mbase.gbars.values[0] * mdata['1'].ds.abatement.sel({'state': state, 'sector': 'Widgets'})**2
    mabate_paths[state, learning_time_ind:] = mdata['1'].ds.abatement.sel({'state': state, 'sector': 'Widgets'})
    mcum_paths[state, learning_time_ind:] = mdata['1'].ds.cumulative_emissions.sel(state=state, time_state=np.arange(30, tf, 1))

import matplotlib.transforms as mtransforms

fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'f']], figsize=(14, 18), sharex=True)

labels = ['Stochastic model,\nLow carbon budget path', 'Stochastic model,\nAverage carbon budget path', 'Stochastic model,\nHigh carbon budget path']
colors = ['#E69F00', '#56B4E9', '#009E73']

print(data['0'].ds.status, mdata['0'].ds.total_cost)

# add asymptotes
ax['a'].hlines(y=0.5 * base.cbars.values[0] * (10 * 0.05)**2 / 1000, xmin=2020, xmax=2150, color='k', linestyle='dotted', linewidth=2)
ax['b'].hlines(y=0.5 * mbase.gbars.values[0] * (10)**2 / 1000, xmin=2020, xmax=2150, color='k', linestyle='dotted', linewidth=2)

ax['c'].hlines(y=10, xmin=2020, xmax=2150, color='k', linestyle='dotted', linewidth=2)
ax['d'].hlines(y=10, xmin=2020, xmax=2150, color='k', linestyle='dotted', linewidth=2)

ax['e'].hlines(y=200, xmin=2020, xmax=2150, color=colors[0], linestyle='dotted', linewidth=2)
ax['e'].hlines(y=300, xmin=2020, xmax=2150, color=colors[1], linestyle='dotted', linewidth=2)
ax['e'].hlines(y=400, xmin=2020, xmax=2150, color=colors[2], linestyle='dotted', linewidth=2)
ax['f'].hlines(y=200, xmin=2020, xmax=2150, color=colors[0], linestyle='dotted', linewidth=2)
ax['f'].hlines(y=300, xmin=2020, xmax=2150, color=colors[1], linestyle='dotted', linewidth=2)
ax['f'].hlines(y=400, xmin=2020, xmax=2150, color=colors[2], linestyle='dotted', linewidth=2)

# add labels for asymtotes
ax['a'].text(x=2025, y=1.07 * base.cbars.values[0] * (10.* 0.05)**2 * 0.5 / 1000, s=r'$c(\delta \bar{a}$)', color='k', fontsize=20)
ax['b'].text(x=2025, y=1.05 * mbase.gbars.values[0] * (10.)**2 * 0.5 / 1000, s=r'$\gamma(\bar{a}$)', color='k', fontsize=20)

ax['c'].text(x=2025, y=10.2, s=r'$\bar{a}$', color='k', fontsize=20)
ax['d'].text(x=2025, y=10.2, s=r'$\bar{a}$', color='k', fontsize=20)

ax['e'].text(x=2025, y=410, s=r'$B_{high}$', color=colors[2], fontsize=20)
ax['e'].text(x=2025, y=310, s=r'$B_{avg}$', color=colors[1], fontsize=20)
ax['e'].text(x=2025, y=210, s=r'$B_{low}$', color=colors[0], fontsize=20)

ax['f'].text(x=2025, y=410, s=r'$B_{high}$', color=colors[2], fontsize=20)
ax['f'].text(x=2025, y=310, s=r'$B_{avg}$', color=colors[1], fontsize=20)
ax['f'].text(x=2025, y=210, s=r'$B_{low}$', color=colors[0], fontsize=20)

# add data
for state in data['1'].ds.state:
    ax['a'].plot(time, 0.5 * base.cbars.values[0] * inv_paths[state]**2  / 1000, linestyle='solid', color=colors[state.values])
    ax['c'].plot(time, abate_paths[state], linestyle='solid', label=labels[state.values], color=colors[state.values])
    ax['e'].plot(time, cum_paths[state], linestyle='solid', label=labels[state.values], color=colors[state.values])

    ax['b'].plot(time, minv_paths[state] / 1000, linestyle='solid', color=colors[state.values], label=labels[state.values])
    ax['d'].plot(time, mabate_paths[state], linestyle='solid', label=labels[state.values], color=colors[state.values])
    ax['f'].plot(time, mcum_paths[state], linestyle='solid', label=labels[state.values], color=colors[state.values])

# plot bases
ax['a'].plot(time, 0.5 * base.cbars.values[0] * base.investment.values[0]**2 / 1000, linestyle='dashed', color='grey',
            linewidth=1.5, label='Deterministic model,\nAverage carbon budget path')
ax['c'].plot(time, base.abatement.values[0, :-1], linestyle='dashed', color='grey', linewidth=1.5, label='Deterministic with average\ncarbon budget')
ax['e'].plot(time, base.cumulative_emissions.values[:-1], linestyle='dashed', color='grey', linewidth=1.5, label='Deterministic model,\nAverage carbon budget path')

ax['b'].plot(time, 0.5 * mbase.gbars.values[0] * mbase.abatement.values[0]**2 / 1000, linestyle='dashed', color='grey', linewidth=1.5,
        label='Deterministic model,\nAverage carbon budget path')
ax['d'].plot(time, mbase.abatement.values[0], linestyle='dashed', color='grey', linewidth=1.5, label='Deterministic with average\ncarbon budget')
ax['f'].plot(time, mbase.cumulative_emissions.values, linestyle='dashed', color='grey', linewidth=1.5, label='Deterministic with average\ncarbon budget')

# aesthetics
ax['a'].set_xlim((2020, 2150))
ax['a'].set_ylim((0, 5.6))
ax['b'].set_ylim((0, 5.6))
ax['c'].set_ylim((0, 10.1))
ax['d'].set_ylim((0, 10.1))
ax['e'].set_ylim((0, 405))
ax['f'].set_ylim((0, 405))
ax['b'].legend(fontsize=12, loc='upper center')

# x labales
ax['e'].set_xlabel("Year")
ax['f'].set_xlabel("Year")

# titles
ax['a'].set_title("Abatement investment model")
ax['b'].set_title('"Strawman" model')

# y labels
ax['a'].set_ylabel("Investment effort\n(Trillions of $ per year)", labelpad=10)
ax['c'].set_ylabel("Abatement (GtCO$_2$ per year)", labelpad=10)
ax['e'].set_ylabel("Cumulative emissions (GtCO$_2$)", labelpad=10)

for label in ['b', 'd', 'f']:
    ax[label].set_yticklabels([])

for label in ax.keys():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[label].text(0.9, 0.96, label, transform=ax[label].transAxes + trans, fontsize=16, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

print(data['0'].ds.total_cost / mdata['0'].ds.total_cost)

fig.savefig(basefile + 'simp.png', dpi=300, bbox_inches='tight')

plt.show()