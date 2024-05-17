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
t17_inv_base = open_datatree(data_head_path + 'ar6_17_inv_output.nc')
t17_inv_rec = open_datatree(data_head_path + 'ar6_17_N1_T40_B500_method2_inv_rp_data_short.nc')

# distributions 
scc_dist_2020 = t17_inv_rec['0.0']['0'].ds.scc.values[:,0]/(1/500)

B_dist = t17_inv_rec['0.0']['0'].ds.B_dist.values

scc_dist_2030 = t17_inv_rec['10.0']['1'].ds.scc.values[:,0]/(1/500)

scc_base = t17_inv_base.scc.values[0] * -1000

import matplotlib.transforms as mtransforms

fig, ax = plt.subplots(1,2, sharey=True, figsize=(16,7))

ylo = 1.0
yhi = 10**3

yhist_alpha = 1.0
xhist_alpha = 1.0

for i in range(2):
    ax[i].set_yscale('log')
    ax[i].set_xlabel("Remaining carbon budget when\ninformation is revealed (GtCO$_2$)")
    ax[i].spines[['right', 'top']].set_visible(True)
    #ax[i].set_ylim(ylo, 2000)
    ax[i].grid(visible=True, axis='y', alpha=0.4, color='k', which='major', zorder=1)

lf_0, uf_0 = 0.9, 1.1

# 2020
ax[0].scatter(B_dist, scc_dist_2020, label='Simulated data', zorder=10)

ax[0].hist(scc_dist_2020, orientation='horizontal', bins=20, alpha=yhist_alpha,
           bottom=min(B_dist)*lf_0, weights=np.ones_like(scc_dist_2020)*2.6,
           label="Carbon price distribution", color='#56B4E9')

ax[0].hist(B_dist, color='#E69F00', alpha=xhist_alpha, bins=20, bottom=ylo,
           label="Carbon budget distribution", zorder=10, weights=np.ones_like(B_dist)*0.05)

ax[0].scatter(np.mean(B_dist)-47, np.mean(scc_dist_2020), marker='*', s=600,
           label="Average carbon price - uncertainty", color='#CC79A7', zorder=100)

ax[0].scatter(np.mean(B_dist), scc_base, marker='s', s=200,
           label="Average carbon price - base", color='#009E73', zorder=90)

ax[0].legend(loc='upper left')

ax[0].set_xlim((min(B_dist)*lf_0, max(B_dist)*uf_0))
ax[0].set_xticks([300,600,900,1200])
ax0_top = ax[0].twiny()
ax0_top.set_xlim(ax[0].get_xlim())
ax0_top.set_xticks([300,600,900,1200])
ax0_top.set_xlabel("Remaining carbon budget\nat initialization (GtCO$_2$)", labelpad=20)
ax[0].set_ylabel(r"Carbon price (\$/tCO$_2$)")

# 2030
ax[1].scatter(B_dist-t17_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values, 
         scc_dist_2030, zorder=10)

ax[1].hist(B_dist-t17_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values, 
           color='#E69F00', alpha=xhist_alpha, bins=20, bottom=ylo, zorder=10, weights=np.ones_like(B_dist)*0.05)

ax[1].hist(scc_dist_2030, orientation='horizontal', bins=5000, alpha=yhist_alpha,
           bottom=0, 
           weights=np.ones_like(scc_dist_2020)*0.99, color='#56B4E9')

ax[1].scatter(np.mean(B_dist)-672, np.mean(scc_dist_2030), marker='*', s=600,
           label="Average carbon price", color='#CC79A7', zorder=100)

ax[1].scatter(np.mean(B_dist)-380, scc_base, marker='s', s=200,
           label="Average carbon price - base", color='#009E73', zorder=90)

ax[1].set_xticks([0,300,600,900])
ax1_top = ax[1].twiny()
ax1_top.set_xlim(ax[0].get_xlim())
ax1_top.set_xticks([300,600,900,1200])
ax1_top.set_xlabel("Remaining carbon budget\nat initialization (GtCO$_2$)", labelpad=20)
ax[1].set_xlim((0,
                max(B_dist-t17_inv_rec['10.0']['1'].ds.cumulative_emissions[1,0].values) * uf_0))

right = ['a', 'b']
tracker = 0
for label in right:
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax[tracker].text(0.9, 0.96, label, transform=ax[tracker].transAxes + trans, fontsize=22, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
    tracker += 1

fig.savefig(basefile + 'carbon-price-dists-data-t17.png', dpi=300, bbox_inches='tight')

plt.show()