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
    disc = (1 + 0.02)**(-1)
    objs = np.zeros((len(pers), 7, 80))
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

pers = np.arange(0.0, 80.0, 5.0) # learning times

time = np.arange(0, 80, 1) + ti
cutoff = 2030

# import data
if run_type == 't15':
    t15_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_15_inv_output.nc')
    t15_inv_rec = open_datatree(data_head_path + cal_prefix + '_15_15N1_T30_B8_method3_inv_rp_data.nc')

    # base model and uncertainty model
    cstars_base_15 = get_base_t_dep_obj(t15_inv_base)
    cstars_15 = get_rec_t_dep_obj(t15_inv_rec, pers)

    #base model, pre- and post-cutoff
    cstars_base_15_pre = np.sum(cstars_base_15[:, time<=cutoff], axis=1)
    cstars_base_15_post = np.sum(cstars_base_15[:, time>cutoff], axis=1)

    # unceratinty model, pre- and post-cutoff
    cstars_15_pre = np.sum(cstars_15[:,:, time<=cutoff], axis=2)
    cstars_15_post = np.sum(cstars_15[:,:, time>cutoff], axis=2)

    # base model aggregated across sectors
    cstars_base_15_pre_agg = np.sum(cstars_base_15_pre, axis=0)
    cstars_base_15_post_agg = np.sum(cstars_base_15_post, axis=0)

    # uncertainty model, aggregated across sectors
    cstars_15_pre_agg = np.sum(cstars_15_pre, axis=1)
    cstars_15_post_agg = np.sum(cstars_15_post, axis=1)

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

    # base model, pre- and post-cutoff
    cstars_base_17_pre = np.sum(cstars_base_17[:, time<=cutoff], axis=1)
    cstars_base_17_post = np.sum(cstars_base_17[:, time>cutoff], axis=1)

    cstars_base_2_pre = np.sum(cstars_base_2[:, time<=cutoff], axis=1)
    cstars_base_2_post = np.sum(cstars_base_2[:, time>cutoff], axis=1)

    # unceratinty model, pre- and post-cutoff
    cstars_17_pre = np.sum(cstars_17[:,:, time<=cutoff], axis=2)
    cstars_17_post = np.sum(cstars_17[:,:, time>cutoff], axis=2)

    cstars_2_pre = np.sum(cstars_2[:,:, time<=cutoff], axis=2)
    cstars_2_post = np.sum(cstars_2[:,:, time>cutoff], axis=2)

    # base model, aggregated across sectors
    cstars_base_17_pre_agg = np.sum(cstars_base_17_pre, axis=0)
    cstars_base_17_post_agg = np.sum(cstars_base_17_post, axis=0)

    cstars_base_2_pre_agg = np.sum(cstars_base_2_pre, axis=0)
    cstars_base_2_post_agg = np.sum(cstars_base_2_post, axis=0)

    # uncertainty model, aggregated across sectors
    cstars_17_pre_agg = np.sum(cstars_17_pre, axis=1)
    cstars_17_post_agg = np.sum(cstars_17_post, axis=1)

    cstars_2_pre_agg = np.sum(cstars_2_pre, axis=1)
    cstars_2_post_agg = np.sum(cstars_2_post, axis=1)

# make figure~
import matplotlib.transforms as mtransforms

if run_type == 't15':
    # make figure~
    fig, ax = plt.subplot_mosaic([['a']], sharex=True,
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(7.5,7.5), sharey=True)

    pre, = ax['a'].plot(time[::5], cstars_15_pre_agg/cstars_base_15_pre_agg, label="Pre-{} Spending".format(cutoff))
    post, = ax['a'].plot(time[::5], cstars_15_post_agg/cstars_base_15_post_agg, label="Post-{} Spending".format(cutoff), linestyle='solid')
    ax['a'].set_title(r"$T^* = 1.5 \ ^\circ$C")

    for i in ['a']:
        ax[i].set_xlim((2020,2050))
        ax[i].set_xlabel("Year information is revealed")
        
    ax['a'].set_ylabel("Additional cost of uncertainty index")

    right = ['a']
    for label in right:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
        ax[label].text(0.9, 1.0, label, transform=ax[label].transAxes + trans, fontsize=22, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

        
    fig.legend([pre, post], ["Pre-{} Spending".format(cutoff), "Post-{} Spending".format(cutoff)], 
            bbox_to_anchor=(0.5, -0.11), 
            loc='lower center', ncol=3, fancybox=True, shadow=True, fontsize=22)

else:
    fig, ax = plt.subplot_mosaic([['a', 'b']], sharex=True,
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1, 1]},
                                figsize=(16,7.5), sharey=True)

    pre, = ax['a'].plot(time[::5], cstars_17_pre_agg/cstars_base_17_pre_agg, label="Pre-{} Spending".format(cutoff))
    post, = ax['a'].plot(time[::5], cstars_17_post_agg/cstars_base_17_post_agg, label="Post-{} Spending".format(cutoff), linestyle='solid')
    ax['a'].set_title(r"$T^* = 1.7 \ ^\circ$C")

    ax['b'].plot(time[::5], cstars_2_pre_agg/cstars_base_2_pre_agg, label="Pre-{} Spending".format(cutoff))
    ax['b'].plot(time[::5], cstars_2_post_agg/cstars_base_2_post_agg, label="Post-{} Spending".format(cutoff), linestyle='solid')
    ax['b'].set_title(r"$T^* = 2 \ ^\circ$C")

    for i in ['a', 'b']:
        ax[i].set_xlim((2020,2050))
        ax[i].set_xlabel("Year information is revealed")
        
    ax['a'].set_ylabel("Additional cost of uncertainty index")

    right = ['a', 'b']
    for label in right:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
        ax[label].text(0.9, 1.0, label, transform=ax[label].transAxes + trans, fontsize=22, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
        
    fig.legend([pre, post], ["Pre-{} Spending".format(cutoff), "Post-{} Spending".format(cutoff)], 
            bbox_to_anchor=(0.5, -0.11), 
            loc='lower center', ncol=3, fancybox=True, shadow=True, fontsize=22)

if run_type == 'low-linear':
    ax['a'].set_yticks([0.5, 1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    ax['b'].set_yticks([0.5, 1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    ax['a'].set_ylim((0.4, 5.1))

if run_type == 'pow':
    ax['a'].set_ylim((0.4, 10.1))

sns.despine(trim=True, offset=10)

if run_type == 't15':
    fig.savefig(basefile + cal_prefix + '-total-cost-ind-cutoff-{}-t15.png'.format(int(cutoff)), dpi=400, bbox_inches='tight')

else:
    fig.savefig(basefile + cal_prefix + '-total-cost-ind-cutoff-{}.png'.format(int(cutoff)), dpi=400, bbox_inches='tight')

plt.show()
