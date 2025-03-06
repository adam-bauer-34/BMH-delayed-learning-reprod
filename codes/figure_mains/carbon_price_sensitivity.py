"""

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
2.25.2025
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
import matplotlib.cm as cmx

from src.presets import get_presets
from datatree import DataTree, open_datatree


def get_carbon_prices(dt, pers):
    """Compute carbon prices for each learing period requested.

    Parameters
    ----------
    dt: `datatree`.DataTree object
        data tree of simulation output

    pers: np.array
        learning dates we want to extract carbon prices for

    Returns
    -------
    mu: np.array
        t=0 carbon prices
    """

    mu = np.zeros_like(pers)

    # loop through periods and pull carbon prices
    for i in range(len(pers)):
        if i == 0:
            tmp_ds = dt[str(pers[i])]['0'].ds
            tmp = tmp_ds.scc.sum('state')[0]

        else:
            tmp = dt[str(pers[i])]['0'].ds.scc.values[0, 0]
        
        if tmp < 0:
            tmp *= -1
        
        mu[i] = tmp

    return mu 


# get presets from source function
presets, basefile = get_presets()
plt.rcParams.update(presets)

# data base path
cwd = os.getcwd()
data_head_path = ''.join([cwd, '/data/output/'])

# number of learning periods we're interested in
pers = np.arange(0.0, 80.0, 5.0)

# import data
## t = 1.7, low linear calibration
t17_inv_rec_lo = open_datatree(data_head_path + 'ar6_17_N1_T30_B8_method3_inv_rp_data.nc')
mu_t17_lo = get_carbon_prices(t17_inv_rec_lo, pers)
print('not yet')
del t17_inv_rec_lo

## t = 1.7, hi linear calibration
t17_inv_rec_hi = open_datatree(data_head_path + 'ar6hi_17_N1_T30_B8_method3_inv_rp_data.nc')
mu_t17_hi = get_carbon_prices(t17_inv_rec_hi, pers)
print('not yet')
del t17_inv_rec_hi

## t = 1.7, nonlinear calibration
t17_inv_rec_pow = open_datatree(data_head_path + 'ar6pow_17_N1_T30_B8_method3_inv_rp_data.nc')
mu_t17_pow = get_carbon_prices(t17_inv_rec_pow, pers)
print('not yet')
del t17_inv_rec_pow

## t = 2, low linear calibration
t2_inv_rec_lo = open_datatree(data_head_path + 'ar6_2_N1_T30_B8_method3_inv_rp_data.nc')
mu_t2_lo = get_carbon_prices(t2_inv_rec_lo, pers)
print('not yet')
del t2_inv_rec_lo

## t = 2, hi linear calibration
t2_inv_rec_hi = open_datatree(data_head_path + 'ar6hi_2_N1_T30_B8_method3_inv_rp_data.nc')
mu_t2_hi = get_carbon_prices(t2_inv_rec_hi, pers)
print('not yet')
del t2_inv_rec_hi

## t = 2, nonlinear calibration
t2_inv_rec_pow = open_datatree(data_head_path + 'ar6pow_2_N1_T30_B8_method3_inv_rp_data.nc')
mu_t2_pow = get_carbon_prices(t2_inv_rec_pow, pers)
print('not yet')
del t2_inv_rec_pow

# make figures
fig, ax = plt.subplots(1, figsize=(6, 6))

# aesthetics
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# plot up the carbon prices
ax.plot(pers+2020, mu_t17_lo, color=color_list[1], linestyle='solid', label=r'$T^* = 1.7 ^\circ$C')
ax.plot(pers+2020, mu_t17_hi, color=color_list[1], linestyle='dashed')
ax.plot(pers+2020, mu_t17_pow, color=color_list[1], linestyle='dashdot')

ax.plot(pers+2020, mu_t2_lo, color=color_list[2], linestyle='solid', label=r'$T^* = 2 ^\circ$C')
ax.plot(pers+2020, mu_t2_hi, color=color_list[2], linestyle='dashed')
ax.plot(pers+2020, mu_t2_pow, color=color_list[2], linestyle='dashdot')

ax.plot([-100, -99], [10, 10], linestyle='solid', color='grey', label='Low-linear')
ax.plot([-100, -99], [10, 10], linestyle='dashed', color='grey', label='High-linear')
ax.plot([-100, -99], [10, 10], linestyle='dashdot', color='grey', label='Nonlinear')

ax.set_xlabel("Year information is revealed")
ax.set_ylabel(r"Carbon price (\$ per tCO$_2$)")
ax.set_yscale('log')

ax.legend(fontsize=14)

# set axis labels
ax.set_xlim((min(pers)+2020, 2050))

# save figure
fig.savefig(basefile + 'carbon-price-sensitivities.png', dpi=400, bbox_inches='tight')

# show plot 
plt.show()