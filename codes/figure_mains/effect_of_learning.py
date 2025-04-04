"""Effect of learning on policy figure.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
2.21.2024
"""

import os
import sys 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from src.presets import get_presets
from datatree import DataTree, open_datatree

def get_tot_cost(dt_rec):
    """Get total costs from policies by cycling through a datatree.

    Parameters
    ----------
    dt_rec: `datatree` object
        datatree containing uncertainty outputs

    Returns
    -------
    VoL: array
        total cost of policies 
    """

    N_sims = dt_rec['10.0'].width
    learning_times = np.linspace(0.0, 80.0, N_sims+1)
    
    VoL = []
    for lt in learning_times[:-1]:
        VoL.append(dt_rec[str(lt)]['0'].ds.total_cost)
        
    return np.array(VoL)

def get_VoL_tot_cost(ds_base, dt_rec):
    """Get total costs net of certainty from policies by cycling through a datatree.

    Parameters
    ----------
    ds_base: `xarray.Dataset` object
        dataset containing output of base model
    dt_rec: `datatree` object
        datatree containing uncertainty outputs

    Returns
    -------
    VoL: array
        total cost of policies net of certainty
    """

    N_sims = dt_rec['10.0'].width
    learning_times = np.linspace(0.0, 80.0, N_sims+1)
    
    VoL = []
    #VoL.append(0.0)

    for lt in learning_times[:-1]:
        VoL.append(dt_rec[str(lt)]['0'].ds.total_cost - ds_base.total_cost)
        
    return np.array(VoL)


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

# import data
if run_type == 't15':
    t15_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_15_inv_output.nc')
    t15_inv_rec = open_datatree(data_head_path + cal_prefix + '_15_15N1_T30_B8_method3_inv_rp_data.nc')

    t15_mac_base = xr.open_dataset(data_head_path + cal_prefix + '_15_mac_output.nc')
    t15_mac_rec = open_datatree(data_head_path + cal_prefix + '_15_15N1_T30_B8_method3_mac_rp_data.nc')

    # total costs
    tot_t15_inv = get_tot_cost(t15_inv_rec)
    tot_t15_mac = get_tot_cost(t15_mac_rec)

    # costs net of certainty
    VoL_t15_inv = get_VoL_tot_cost(t15_inv_base, t15_inv_rec)
    VoL_t15_mac = get_VoL_tot_cost(t15_mac_base, t15_mac_rec)

else:
    t17_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_17_inv_output.nc')
    t17_inv_rec = open_datatree(data_head_path + cal_prefix + '_17_N1_T30_B8_method3_inv_rp_data.nc')

    t2_inv_base = xr.open_dataset(data_head_path + cal_prefix + '_2_inv_output.nc')
    t2_inv_rec = open_datatree(data_head_path + cal_prefix + '_2_N1_T30_B8_method3_inv_rp_data.nc')

    t17_mac_base = xr.open_dataset(data_head_path + cal_prefix + '_17_mac_output.nc')
    t17_mac_rec = open_datatree(data_head_path + cal_prefix + '_17_N1_T30_B8_method3_mac_rp_data.nc')

    t2_mac_base = xr.open_dataset(data_head_path + cal_prefix + '_2_mac_output.nc')
    t2_mac_rec = open_datatree(data_head_path + cal_prefix + '_2_N1_T30_B8_method3_mac_rp_data.nc')

    # total costs
    tot_t17_inv = get_tot_cost(t17_inv_rec)
    tot_t2_inv = get_tot_cost(t2_inv_rec)
    tot_t17_mac = get_tot_cost(t17_mac_rec)
    tot_t2_mac = get_tot_cost(t2_mac_rec)

    # costs net of certainty
    VoL_t17_inv = get_VoL_tot_cost(t17_inv_base, t17_inv_rec)
    VoL_t2_inv = get_VoL_tot_cost(t2_inv_base, t2_inv_rec)
    VoL_t17_mac = get_VoL_tot_cost(t17_mac_base, t17_mac_rec)
    VoL_t2_mac = get_VoL_tot_cost(t2_mac_base, t2_mac_rec)

# set number of learning dates
pers = np.arange(0, 80, 5)

# make big figure!
import matplotlib.transforms as mtransforms

fig, ax = plt.subplot_mosaic([['a', 'b', 'c']], sharex=True,
                             gridspec_kw={'height_ratios': [1], 'width_ratios': [1, 1, 1]},
                            figsize=(20, 6))

# total cost of uncertainty
if run_type == 't15':
    # total cost of uncertainty
    t15, = ax['a'].plot(pers + ti, tot_t15_inv/1000, linestyle=
            'solid', color='#E69F00', label='$T^* = 1.5 \ ^\circ$C')
    ax['a'].plot(pers + ti, tot_t15_mac/1000, linestyle='dashed', color='#E69F00')

    dummy_inv, = ax['a'].plot([-100, -99], [np.mean(tot_t15_inv)/1000, np.mean(tot_t15_inv)/1000],
                            linestyle='solid', color='grey', label='Abatement investment model')
    dummy_mac, = ax['a'].plot([-100, -99], [np.mean(tot_t15_inv)/1000, np.mean(tot_t15_inv)/1000],
                            linestyle='dashed', color='grey', label='"Strawman" model')

    # additional cost of uncertainty 
    ax['b'].plot(pers + ti, VoL_t15_inv/1000, linestyle='solid', color='#E69F00', label='$T^* = 1.5 \ ^\circ$C')
    ax['b'].plot(pers + ti, VoL_t15_mac/1000, linestyle='dashed', color='#E69F00')

    # marginal savings
    dt = 5.

    ax['c'].plot(pers + ti, np.gradient(VoL_t15_mac, dt)/1000, linestyle='dashed', color='#E69F00')
    ax['c'].plot(pers + ti, np.gradient(tot_t15_inv, dt)/1000, linestyle='solid', color='#E69F00', label='$T^* = 1.5 \ ^\circ$C')

    fig.legend([t15, dummy_inv, dummy_mac], ['$T^* = 1.5 \ ^\circ$C', 'Abatement investment model', '"Strawman" model'], 
           bbox_to_anchor=(0.5, -0.16), 
           loc='lower center', ncol=2, fancybox=True, shadow=True, fontsize=22)

else:
    t17, = ax['a'].plot(pers + ti, tot_t17_inv/1000, linestyle='solid', color='#56B4E9', label='$T^* = 1.7 \ ^\circ$C')
    t2, = ax['a'].plot(pers + ti, tot_t2_inv/1000, linestyle='solid', color='#009E73', label='$T^* = 2 \ ^\circ$C')

    dummy_inv, = ax['a'].plot([-100, -99], [np.mean(tot_t17_inv)/1000, np.mean(tot_t17_inv)/1000],
                            linestyle='solid', color='grey', label='Abatement investment model')
    dummy_mac, = ax['a'].plot([-100, -99], [np.mean(tot_t17_inv)/1000, np.mean(tot_t17_inv)/1000],
                            linestyle='dashed', color='grey', label='"Strawman" model')

    ax['a'].plot(pers + ti, tot_t17_mac/1000, linestyle='dashed', color='#56B4E9')
    ax['a'].plot(pers + ti, tot_t2_mac/1000, linestyle='dashed', color='#009E73')

    # additional cost of uncertainty
    ax['b'].plot(pers + ti, VoL_t17_inv/1000, linestyle='solid', color='#56B4E9', label='$T^* = 1.7 \ ^\circ$C')
    ax['b'].plot(pers + ti, VoL_t2_inv/1000, linestyle='solid', color='#009E73', label='$T^* = 2 \ ^\circ$C')

    ax['b'].plot(pers + ti, VoL_t17_mac/1000, linestyle='dashed', color='#56B4E9')
    ax['b'].plot(pers + ti, VoL_t2_mac/1000, linestyle='dashed', color='#009E73')

    # marginal savings
    dt = 5.

    ax['c'].plot(pers + ti, np.gradient(VoL_t17_inv, dt)/1000, linestyle='solid', color='#56B4E9', label='$T^* = 1.7 \ ^\circ$C')
    ax['c'].plot(pers + ti, np.gradient(VoL_t2_inv, dt)/1000, linestyle='solid', color='#009E73', label='$T^* = 2 \ ^\circ$C')

    ax['c'].plot(pers + ti, np.gradient(VoL_t17_mac, dt)/1000, linestyle='dashed', color='#56B4E9')
    ax['c'].plot(pers + ti, np.gradient(VoL_t2_mac, dt)/1000, linestyle='dashed', color='#009E73')

    # index: inv vs mac
    fig.legend([t17, t2, dummy_inv, dummy_mac], ['$T^* = 1.7 \ ^\circ$C', '$T^* = 2 \ ^\circ$C', 'Abatement investment model', '"Strawman" model'], 
           bbox_to_anchor=(0.52, -0.14), 
           loc='lower center', ncol=2, fancybox=True, shadow=True, fontsize=18)

ax['a'].set_ylabel("Total cost of policy (Trillions of $)")
ax['b'].set_ylabel("Additional cost of uncertainty\n(Trillions of $)")
ax['c'].set_ylabel("Marginal value of learning the carbon\nbudget (Trillions of \$ per year)")

ax['a'].set_xlabel("Year information is revealed")
ax['b'].set_xlabel("Year information is revealed")
ax['c'].set_xlabel("Year information is revealed")

if run_type == 'low-linear':
    ax['a'].set_yticks([20, 30, 40, 50])
    ax['a'].set_ylim((15, 51))
    ax['b'].set_yticks([0, 5, 10, 15, 20, 25])
    ax['c'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# set x limits to not show dummy lines that give us the grey legend entries 
ax['a'].set_xlim((2020, 2100))
ax['b'].set_xlim((2020, 2100))
ax['c'].set_xlim((2020, 2100))

right = ['a', 'b', 'c']
for label in right:
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[label].text(0.9, 1.0, label, transform=ax[label].transAxes + trans, fontsize=22, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

# fig.subplots_adjust(wspace=0.2, hspace=0.14)
fig.tight_layout()
# sns.despine(trim=True, offset=10)

if run_type == 't15':
    fig.savefig(basefile + cal_prefix + '-pfig-value-of-learning-quadbox-t15.png', dpi=400, bbox_inches='tight')

else:
    fig.savefig(basefile + cal_prefix + '-pfig-value-of-learning-quadbox.png', dpi=400, bbox_inches='tight')    

# plt.show()
