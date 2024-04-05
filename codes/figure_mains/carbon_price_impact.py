import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

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
tf = 2100 # generally...?

# import data
t15_inv_rec = open_datatree(data_head_path + 'ar6_15_N1_T40_B500_method0_inv_rp_data_short.nc')

# distributions 
scc_dist_2020 = t15_inv_rec['0.0']['0'].ds.scc.values[:,0]/(1/500)
B_dist = t15_inv_rec['0.0']['0'].ds.B_dist.values
scc_dist_2030 = t15_inv_rec['10.0']['1'].ds.scc.values[:,0]/(1/500)

import matplotlib.transforms as mtransforms

fig, ax = plt.subplots(1,2, sharey=True, figsize=(16,7))

ylo = 5
yhi = 10**5

yhist_alpha = 1.0
xhist_alpha = 1.0

for i in range(2):
    ax[i].set_yscale('log')
    ax[i].set_xlabel("Remaining carbon budget when\ninformation is revealed (GtCO$_2$)")
    ax[i].spines[['right', 'top']].set_visible(True)
    ax[i].set_ylim(ylo, yhi)
    ax[i].grid(visible=True, axis='y', alpha=0.4, color='k', which='major', zorder=1)

lf_0, uf_0 = 0.9, 1.1

# 2020
ax[0].scatter(B_dist, scc_dist_2020, label='Simulated data', zorder=10)

ax[0].hist(scc_dist_2020, orientation='horizontal', bins=15, alpha=yhist_alpha,
           bottom=min(B_dist)*lf_0, weights=np.ones_like(scc_dist_2020)*1.75,
           label="Carbon price distribution", color='#56B4E9')

ax[0].hist(B_dist, color='#E69F00', alpha=xhist_alpha, bins=20, weights=np.ones_like(B_dist)*0.22, bottom=ylo,
           label="Carbon budget distribution", zorder=10)

ax[0].hlines(np.mean(scc_dist_2020), min(B_dist)*lf_0, max(B_dist)*uf_0,
             color='#56B4E9', linestyle='dashdot', linewidth=3, label="Average carbon price")

ax[0].legend(loc='upper left')

ax[0].set_xlim((min(B_dist)*lf_0, max(B_dist)*uf_0))
ax0_top = ax[0].twiny()
ax0_top.set_xlim(ax[0].get_xlim())
ax0_top.set_xticks([300,400,500,600,700])
ax0_top.set_xlabel("Remaining carbon budget\nat initialization (GtCO$_2$)", labelpad=20)
ax[0].set_ylabel(r"Carbon price (\$/tCO$_2$)")

# 2030
ax[1].scatter(B_dist-t15_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values, 
         scc_dist_2030, zorder=10)

ax[1].hist(B_dist-t15_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values, 
           color='#E69F00', alpha=xhist_alpha, bins=20, weights=np.ones_like(B_dist)*0.22, bottom=ylo, zorder=10)

ax[1].hist(scc_dist_2030, orientation='horizontal', bins=5000, alpha=yhist_alpha,
           bottom=0, 
           weights=np.ones_like(scc_dist_2020)*1.5, color='#56B4E9')

ax[1].hlines(np.mean(scc_dist_2030), 0,
                max(B_dist-t15_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values) * uf_0,
             color='#56B4E9', linestyle='dashdot', linewidth=3)

ax[1].set_xticks([0,100,200,300,400,500])
ax1_top = ax[1].twiny()
ax1_top.set_xlim(ax[0].get_xlim())
ax1_top.set_xticks([300,400,500,600,700])
ax1_top.set_xlabel("Remaining carbon budget\nat initialization (GtCO$_2$)", labelpad=20)
ax[1].set_xlim((0,
                max(B_dist-t15_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values) * uf_0))

right = ['a', 'b']
tracker = 0
for label in right:
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[tracker].text(0.9, 0.96, label, transform=ax[tracker].transAxes + trans, fontsize=22, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
    tracker += 1

fig.savefig(basefile + 'carbon-price-dists-data-15.png', dpi=300, bbox_inches='tight')

plt.show()