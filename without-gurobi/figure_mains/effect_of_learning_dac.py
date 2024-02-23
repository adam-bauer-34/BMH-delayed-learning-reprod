"""Effect of learning figure, but with direct air capture technologies
included.

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

from src.presets import get_presets
from datatree import DataTree, open_datatree

def get_tot_cost(dt_rec, bs='no'):
    """Get total costs from policies by cycling through a datatree.

    Parameters
    ----------
    dt_rec: `datatree` object
        datatree containing uncertainty outputs

    bs: string
        if "bs" (backstop, aka DAC), do a slightly different computation owing
        to scalings in the optimization

    Returns
    -------
    VoL: array
        total cost of policies 
    """
    N_sims = dt_rec['10.0'].width
    learning_times = np.linspace(0.0, 80.0, N_sims+1)
    
    VoL = []

    for lt in learning_times[:-1]:
        if lt > 0 and bs == 'bs':
            VoL.append(dt_rec[str(lt)]['0'].ds.total_cost)
        else:
            VoL.append(dt_rec[str(lt)]['0'].ds.total_cost)
        
    return np.array(VoL)

def get_VoL_tot_cost(ds_base, dt_rec, bs='no'):
    """Get total costs net of certainty from policies by cycling through a datatree.

    Parameters
    ----------
    ds_base: `xarray.Dataset` object
        dataset containing output of base model
    dt_rec: `datatree` object
        datatree containing uncertainty outputs
    bs: string
        if "bs" (backstop, aka DAC), do a slightly different computation owing
        to scalings in the optimization

    Returns
    -------
    VoL: array
        total cost of policies net of certainty
    """
    N_sims = dt_rec['10.0'].width
    learning_times = np.linspace(0.0, 80.0, N_sims+1)
    
    VoL = []

    for lt in learning_times[:-1]:
        if lt > 0 and bs == 'bs':
            VoL.append(dt_rec[str(lt)]['0'].ds.total_cost - ds_base.total_cost)
        else:
            VoL.append(dt_rec[str(lt)]['0'].ds.total_cost - ds_base.total_cost)
        
    return np.array(VoL)

# get presets from source function
presets, basefile = get_presets()
plt.rcParams.update(presets)

# data base path
cwd = os.getcwd()
data_head_path = ''.join([cwd, '/../models/data/output/'])

# initial year for plotting
ti = 2020
tf = 2100 # generally...?

# import data
t15_inv_base = xr.open_dataset(data_head_path + 'ar6_15_inv_output.nc')
t15_inv_rec = open_datatree(data_head_path + 'ar6_15_N1_T40_B6_method1_inv_rp_data.nc')
t15_inv_recbs = open_datatree(data_head_path + 'ar6bs_15_N1_T40_B6_method1_invBS_rp_data.nc')

t17_inv_base = xr.open_dataset(data_head_path + 'ar6_17_inv_output.nc')
t17_inv_rec = open_datatree(data_head_path + 'ar6_17_N1_T40_B6_method1_inv_rp_data.nc')
t17_inv_recbs = open_datatree(data_head_path + 'ar6bs_17_N1_T40_B6_method1_invBS_rp_data.nc')

t2_inv_base = xr.open_dataset(data_head_path + 'ar6_2_inv_output.nc')
t2_inv_rec = open_datatree(data_head_path + 'ar6_2_N1_T40_B6_method1_inv_rp_data.nc')
t2_inv_recbs = open_datatree(data_head_path + 'ar6bs_2_N1_T40_B6_method1_invBS_rp_data.nc')

# total costs
tot_t15_inv = get_tot_cost(t15_inv_rec)
tot_t17_inv = get_tot_cost(t17_inv_rec)
tot_t2_inv = get_tot_cost(t2_inv_rec)

tot_t15_invbs = get_tot_cost(t15_inv_recbs, 'bs')
tot_t17_invbs = get_tot_cost(t17_inv_recbs, 'bs')
tot_t2_invbs = get_tot_cost(t2_inv_recbs, 'bs')

# net of certainty
VoL_t15_inv = get_VoL_tot_cost(t15_inv_base, t15_inv_rec)
VoL_t17_inv = get_VoL_tot_cost(t17_inv_base, t17_inv_rec)
VoL_t2_inv = get_VoL_tot_cost(t2_inv_base, t2_inv_rec)

VoL_t15_invbs = get_VoL_tot_cost(t15_inv_base, t15_inv_recbs, 'bs')
VoL_t17_invbs = get_VoL_tot_cost(t17_inv_base, t17_inv_recbs, 'bs')
VoL_t2_invbs = get_VoL_tot_cost(t2_inv_base, t2_inv_recbs, 'bs')

# make figure
import matplotlib.transforms as mtransforms


fig, ax = plt.subplot_mosaic([['a', 'b']], sharex=True,
                             gridspec_kw={'height_ratios': [1], 'width_ratios': [1, 1]},
                            figsize=(20,8))

# total cost of uncertainty
t15, = ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t15_inv/1000, linestyle=
        'solid', color='#E69F00', label='$T^* = 1.5 \ ^\circ$C')
t17, = ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t17_inv/1000, linestyle='solid', color='#56B4E9', label='$T^* = 1.7 \ ^\circ$C')
t2, = ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t2_inv/1000, linestyle='solid', color='#009E73', label='$T^* = 2 \ ^\circ$C')

ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t15_invbs/1000, linestyle='dashed', color='#E69F00')
ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t17_invbs/1000, linestyle='dashed', color='#56B4E9')
ax['a'].plot(t15_inv_base.time.values[::5] + ti, tot_t2_invbs/1000, linestyle='dashed', color='#009E73')

ax['a'].set_ylabel("Total cost of policy (Trillions of $)")
ax['a'].set_xlabel("Year information is revealed")

ax['a'].set_yticks([5,10,15,20,25,30,35,40])

# additional cost of uncertainty
ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t15_inv/1000, linestyle='solid', color='#E69F00', label='$T^* = 1.5 \ ^\circ$C')
ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t17_inv/1000, linestyle='solid', color='#56B4E9', label='$T^* = 1.7 \ ^\circ$C')
ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t2_inv/1000, linestyle='solid', color='#009E73', label='$T^* = 2 \ ^\circ$C')

ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t15_invbs/1000, linestyle='dashed', color='#E69F00')
ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t17_invbs/1000, linestyle='dashed', color='#56B4E9')
ax['b'].plot(t15_inv_base.time.values[::5] + ti, VoL_t2_invbs/1000, linestyle='dashed', color='#009E73')

ax['b'].set_ylabel("Additional cost of uncertainty\n(Trillions of \$)")
ax['b'].set_xlabel("Year information is revealed")

right = ['a', 'b']
for label in right:
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[label].text(0.9, 1.0, label, transform=ax[label].transAxes + trans, fontsize=22, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
    
fig.legend([t15, t17, t2], ['$T^* = 1.5 \ ^\circ$C', '$T^* = 1.7 \ ^\circ$C', '$T^* = 2 \ ^\circ$C'], 
           bbox_to_anchor=(0.5, -0.1), 
           loc='lower center', ncol=3, fancybox=True, shadow=True, fontsize=22)

fig.subplots_adjust(wspace=0.2, hspace=0.14)

sns.despine(trim=True, offset=10)

fig.savefig(basefile + 'ar6bs-pfig-value-of-learning-duobox-withbs.png', dpi=400, bbox_inches='tight')

plt.show()