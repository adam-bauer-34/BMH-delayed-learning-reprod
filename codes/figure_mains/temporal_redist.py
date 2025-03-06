"""How additional uncertainty spending is distributed in time. 

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
2.22.2024
"""

import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
import matplotlib.cm as cmx

from src.presets import get_presets
from datatree import DataTree, open_datatree

def get_rec_t_dep_obj(dt_rec, pers):
    """Get optimal discounted cost at every point in time; the sum
    of the resulting array is the total value of the objective.

    Parameters
    ----------
    dt_rec: `datatree` object
        datatree for model with uncertainty data

    pers: array
        all the learning times in the RiskPrem simulation

    Returns
    -------
    objs: (len(pers), N_sectors, len(times))
        discounted objective in each sector as a function of time and for
        each learning time.
    """

    N_secs = len(dt_rec['0.0']['0'].ds.sector)
    disc = dt_rec['0.0']['0'].ds.beta
    N_time = len(dt_rec['0.0']['0'].ds.time.values)
    objs = np.zeros((len(pers), 7, N_time))
    for i in range(len(pers)):
        tmp_dt = dt_rec[str(pers[i])]
        if i == 0:
            tmp_ds = tmp_dt['0'].ds
            discount = disc**(tmp_ds.time.values)
            avg_inv = (tmp_ds.investment * tmp_ds.B_probs).sum('state')
            obj = discount * np.array([(tmp_ds.power.values[i]+1)**(-1) * tmp_ds.cbars.values[i] * avg_inv[i].values**(tmp_ds.power.values[i]+1) for i in range(N_secs)])
            
        else:
            # loop through pre- and post- periods
            tmp_objs = []
            for j in range(2):
                tmp_ds = tmp_dt[str(j)].ds
                discount = disc**(tmp_ds.time.values)
                avg_inv = (tmp_ds.investment * tmp_ds.B_probs).sum('state')
                obj = discount * np.array([(tmp_ds.power.values[i]+1)**(-1) * tmp_ds.cbars.values[i] * avg_inv[i].values**(tmp_ds.power.values[i]+1) for i in range(N_secs)])
                tmp_objs.append(obj)
            obj = np.hstack([tmp_objs[0], tmp_objs[1]])
        
        objs[i] = obj

    return objs

def get_base_t_dep_obj(ds_base):
    """Get the objective function (minimized discounted costs) split
    up at each point in time from the base model run.

    Parameters
    ----------
    ds_base: `xarray.Dataset`
        dataset from base model simulation

    Returns
    -------
    disc_inv: (len(time),)
        discounted cost of investment at each point in time from optimal
        solution
    """

    disc_factor = ds_base.beta**(ds_base.time.values)
    power = ds_base.power.values + 1
    investment = np.array([(power[i]**(-1) * ds_base.cbars.values[i] * ds_base.investment[i].values**power[i]) for i in range(len(ds_base.sector.values))])
    disc_inv = disc_factor * investment
    return disc_inv

# parse run-specific information
run_type = sys.argv[1]

if run_type == 'low-linear':
    cal_prefix = 'ar6'

elif run_type == 'high-linear':
    cal_prefix = 'ar6hi'

elif run_type == 'pow':
    cal_prefix = 'ar6pow'

elif run_type == 'emis':
    cal_prefix = 'ar6emis'

elif run_type == 't15':
    cal_prefix = 'ar6'

else:
    raise ValueError("Invalid calibration type.")

# get presets from source function
presets, basefile = get_presets()
plt.rcParams.update(presets)

# data base path
cwd = os.getcwd()
data_head_path = ''.join([cwd, '/data/output/'])

# initial year for plotting
ti = 2020
tf = 2100 # generally...?
# get objective for base and reecourse models

pers = np.arange(0.0, 35.0, 5.0) # learning times

time = np.arange(0, 80.0, 1)
cutoff = 2030

# import data
if run_type == 't15':
    t15_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_15_inv_output.nc')
    t15_inv_rec = open_datatree(data_head_path + cal_prefix + '_15_15N1_T30_B8_method3_inv_rp_data.nc')

    # base model and uncertainty model
    cstars_base_15 = get_base_t_dep_obj(t15_inv_base)
    cstars_15 = get_rec_t_dep_obj(t15_inv_rec, pers)

    jet = cm = plt.get_cmap('magma') 
    cNorm  = colors.Normalize(vmin=ti, vmax=max(pers)+ti)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    fig, ax = plt.subplots(1, figsize=(7, 6))

    for i in range(7):
        colorVal = scalarMap.to_rgba(pers[i]+ti)
        ax.plot(time + ti, np.sum(cstars_15, axis=1)[i]/1000, linestyle='solid', color=colorVal)

    ax.plot([-100, -100], [0.1, 0.1], linestyle='solid', color='grey', label='Stochastic')

    ax.plot(time + ti, np.sum(cstars_base_15, axis=0)/1000, color='grey', linestyle='dashed', label="Deterministic")

    ax.legend()

    ax.set_ylabel("Total investment effort\n(Trillions of $ per year)")
    ax.set_xlabel("Year")

    ax.set_title(r"$T^* = 1.5 \ ^\circ$C")

    ax.set_xlim((2015, 2105))

    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax.text(0.95, 1.0, 'a', transform=ax.transAxes + trans, 
                                fontsize=16, fontweight='bold',
                                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

    cbar = plt.colorbar(scalarMap, ax=ax)
    cbar.set_label("Year information is revealed", rotation=270, labelpad=25)

else:
    t17_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_17_inv_output.nc')
    t17_inv_rec = open_datatree(data_head_path + cal_prefix + '_17_N1_T30_B8_method3_inv_rp_data.nc')

    t2_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_2_inv_output.nc')
    t2_inv_rec = open_datatree(data_head_path + cal_prefix + '_2_N1_T30_B8_method3_inv_rp_data.nc')

    # base models
    cstars_base_17 = get_base_t_dep_obj(t17_inv_base)
    cstars_base_2 = get_base_t_dep_obj(t2_inv_base)

    # unceratinty models
    cstars_17 = get_rec_t_dep_obj(t17_inv_rec, pers)
    cstars_2 = get_rec_t_dep_obj(t2_inv_rec, pers)

    jet = cm = plt.get_cmap('magma') 
    cNorm  = colors.Normalize(vmin=ti, vmax=max(pers)+ti)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(7):
        colorVal = scalarMap.to_rgba(pers[i]+ti)
        ax[0].plot(time + ti, np.sum(cstars_17, axis=1)[i]/1000, linestyle='solid', color=colorVal)
        ax[1].plot(time + ti, np.sum(cstars_2, axis=1)[i]/1000, linestyle='solid', color=colorVal)

    ax[1].plot([-100, -100], [0.1, 0.1], linestyle='solid', color='grey', label='Stochastic')

    ax[0].plot(time + ti, np.sum(cstars_base_17, axis=0)/1000, color='grey', linestyle='dashed', label="Deterministic")
    ax[1].plot(time + ti, np.sum(cstars_base_2, axis=0)/1000, color='grey', linestyle='dashed', label='Deterministic')

    ax[1].legend(loc='center right')

    ax[0].set_ylabel("Total investment effort\n(Trillions of $ per year)")
    ax[0].set_xlabel("Year")
    ax[1].set_xlabel("Year")

    ax[0].set_title(r"$T^* = 1.7 \ ^\circ$C")
    ax[1].set_title(r"$T^* = 2 \ ^\circ$C")

    ax[1].set_xlim((2015, 2105))

    labels = ['a', 'b']
    for label in [0, 1]:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
        ax[label].text(0.95, 1.0, labels[label], transform=ax[label].transAxes + trans, 
                                    fontsize=16, fontweight='bold',
                                    verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

    cbar = plt.colorbar(scalarMap, ax=ax)
    cbar.set_label("Year information is revealed", rotation=270, labelpad=25)

if run_type == 't15':
    fig.savefig(basefile + cal_prefix + '-temp-redist-t15.png'.format(int(cutoff)), dpi=400, bbox_inches='tight')

else:
    fig.savefig(basefile + cal_prefix + '-temp-redist.png'.format(int(cutoff)), dpi=400, bbox_inches='tight')

plt.show()
