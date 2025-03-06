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

def get_rec_t_dep_obj(dt_rec, pers):
    N_secs = len(dt_rec['0.0']['0'].ds.sector)
    disc = (1 + 0.02)**(-1)
    objs = np.zeros((len(pers), 7, 80))
    a_s = np.zeros((len(pers), 7, 81))
    for i in range(len(pers)):
        tmp_dt = dt_rec[str(pers[i])]
        if i == 0:
            tmp_ds = tmp_dt['0'].ds
            discount = disc**(tmp_ds.time.values)
            avg_inv = (tmp_ds.investment * tmp_ds.B_probs).sum('state')
            avg_a = (tmp_ds.abatement * tmp_ds.B_probs).sum('state').values
            obj = np.array([(tmp_ds.power.values[i]+1)**(-1) * tmp_ds.cbars.values[i]
                            * avg_inv[i].values**(tmp_ds.power.values[i]+1) for i in range(N_secs)])
            a = avg_a
            
        else:
            # loop through pre- and post- periods
            tmp_objs = []
            tmp_as = []
            for j in range(2):
                tmp_ds = tmp_dt[str(j)].ds
                discount = disc**(tmp_ds.time.values)
                avg_inv = (tmp_ds.investment * tmp_ds.B_probs).sum('state')
                avg_a = (tmp_ds.abatement * tmp_ds.B_probs).sum('state')
                obj = np.array([(tmp_ds.power.values[i]+1)**(-1) * tmp_ds.cbars.values[i]
                                * avg_inv[i].values**(tmp_ds.power.values[i]+1) for i in range(N_secs)])
                tmp_objs.append(obj)
                tmp_as.append(avg_a.values)
            obj = np.hstack([tmp_objs[0], tmp_objs[1]])
            a = np.hstack([tmp_as[0][:, :-1], tmp_as[1]])
        
        objs[i] = obj
        a_s[i] = a

    return objs, a_s

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
t17_inv_base = xr.open_dataset(data_head_path + 'ar6_17_inv_output.nc')
t17_inv_rec = open_datatree(data_head_path + 'ar6_17_N1_T30_B8_method3_inv_rp_data.nc')

pers = np.arange(0.0, 35.0, 5.0)

time = np.arange(0, 80, 1) + ti

cstars, astars = get_rec_t_dep_obj(t17_inv_rec, pers)

import matplotlib.transforms as mtransforms

fig, ax = plt.subplots(2,4, figsize=(24, 12))

jet = cm = plt.get_cmap('magma') 
cNorm  = colors.Normalize(vmin=ti, vmax=max(pers)+ti)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

sec = t17_inv_base.sector.values
sec_x = [0,0,0,0,1,1,1]
sec_y = [0,1,2,3,0,1,2]

for j in range(7):
    for i in range(len(pers)):
        colorVal = scalarMap.to_rgba(pers[i]+ti)
        ax[sec_x[j], sec_y[j]].plot(time, cstars[i,j], color=colorVal, linestyle='solid')
        # ax[sec_x[j], sec_y[j]].set_xlim((2020,2150))
        ax[sec_x[j], sec_y[j]].set_title(sec[j])
        
        if sec_y[j] == 0:
            ax[sec_x[j], sec_y[j]].set_ylabel("Investment effort\n(Billions of \$ yr$^{-1}$)")
        
        if sec_x[j] == 1 and sec_y[j] != 3:
            ax[sec_x[j], sec_y[j]].set_xlabel("Year")
        
        elif sec_y[j] == 3:
            ax[sec_x[j], sec_y[j]].set_xlabel("Year")
            
        else:
            continue

ax[1,3].axis('off')

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
for j in range(7):
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[sec_x[j], sec_y[j]].text(0.85, 1.0, labels[j], transform=ax[sec_x[j], sec_y[j]].transAxes + trans, 
                                fontsize=22, fontweight='bold',
                                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

cbar = plt.colorbar(scalarMap, ax=ax)
cbar.set_label("Year information is revealed", rotation=270, labelpad=45)
#sns.despine(trim = True)

#fig.subplots_adjust(wspace=0.2, hspace=0.2)

no_xs = [0,0,0]
no_ys = [0,1,2]
for i in range(len(no_xs)):
    ax[no_xs[i], no_ys[i]].tick_params(axis='x', labelcolor='white')

# ax[0, 1].set_ylim((0, 375))

fig.savefig(basefile + 'ar6-sec-inv-eff-lt-t17.png', dpi=400, bbox_inches='tight')

plt.show()