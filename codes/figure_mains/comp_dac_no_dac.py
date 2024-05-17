import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from src.presets import get_presets
from datatree import DataTree, open_datatree

def get_rec_t_dep_obj(dt_rec, pers, bs=False):
    disc = (1 + 0.02)**(-1)
    if bs:
        N_secs = 8
    else:
        N_secs = 7
    # loop through pre- and post- periods
    tmp_objs = []
    tmp_as = []
    for j in range(2):
        tmp_ds = dt_rec[str(j)].ds
        discount = disc**(tmp_ds.time.values)
        avg_inv = (tmp_ds.investment * tmp_ds.B_probs).sum('state')
        avg_a = (tmp_ds.abatement * tmp_ds.B_probs).sum('state')
        obj = np.array([(2)**(-1) * tmp_ds.cbars.values[i] * avg_inv[i].values**(2) for i in range(N_secs)])
        tmp_objs.append(obj)
        tmp_as.append(avg_a.values)
    obj = np.hstack([tmp_objs[0], tmp_objs[1]])
    a = np.hstack([tmp_as[0][:, :-1], tmp_as[1]])

    return obj, a

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
t17_inv_rec = open_datatree(data_head_path + 'ar6_17_N1_T30_B8_GHQ_TRUNC_invrec_output.nc')
t17_inv_recbs = open_datatree(data_head_path + 'ar6bs_17_GHQ_TRUNC_invrec_output.nc')

t17_inv_costs_mean_sec = get_rec_t_dep_obj(t17_inv_rec, [1.])[0]
t17_invbs_costs_mean_sec = get_rec_t_dep_obj(t17_inv_recbs, [1.], True)[0]

fig, ax = plt.subplots(2,4, figsize=(25, 14), sharex=True)

x_inds = [0,0,0,0,1,1,1,1]
y_inds = [0,1,2,3,0,1,2,3]

track = 1
rec_time = np.arange(0,80,1)

cert, = ax[0,0].plot(t17_inv_base.time + ti, 
                    (0.5 * t17_inv_base.cbars.values[0] * t17_inv_base.investment.values[0]**2))
nodac_avg, = ax[0,0].plot(rec_time + ti, t17_inv_costs_mean_sec[0],
                                      linestyle='solid', color='#E69F00')
dac_avg, = ax[0,0].plot(rec_time + ti, t17_invbs_costs_mean_sec[0],
                                      linestyle='solid', color='#56B4E9')

for i in range(1, len(x_inds)):
    if track != 7:
        ax[x_inds[i], y_inds[i]].plot(t17_inv_base.time + ti, (0.5 * t17_inv_base.cbars.values[i] * t17_inv_base.investment.values[i]**2), label="Certainty policy")
        #ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_inv_costs_med_sec[i], label="Recourse: Median",
        #                              linestyle='dashed', color='#E69F00')
        ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_inv_costs_mean_sec[i], label="Average investment without DAC",
                                      linestyle='solid', color='#E69F00')
        #ax[x_inds[i], y_inds[i]].fill_between(rec_time + ti, t17_inv_costs_5p_sec[i], t17_inv_costs_95p_sec[i], 
        #                                      label=r"Recourse: 5$^{\rm{th}}-$95$^{\rm{th}}$ quantile range",
        #                                      alpha=0.4, color='#E69F00')
    
        #ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_invbs_costs_med_sec[i], label="RecourseBS: Median",
        #                              linestyle='dashed', color='#56B4E9')
        ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_invbs_costs_mean_sec[i],
                                      linestyle='solid', color='#56B4E9', label="Average investment with DAC")
        #ax[x_inds[i], y_inds[i]].fill_between(rec_time + ti, t17_invbs_costs_5p_sec[i], t17_invbs_costs_95p_sec[i], 
        #                                      alpha=0.4, color='#56B4E9')
        
    else:
        #ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_invbs_costs_med_sec[i], label="Recourse: Median",
        #                              linestyle='dashed', color='#56B4E9')
        ax[x_inds[i], y_inds[i]].plot(rec_time + ti, t17_invbs_costs_mean_sec[i], label="Recourse: Mean",
                                      linestyle='solid', color='#56B4E9')
        #ax[x_inds[i], y_inds[i]].fill_between(rec_time + ti, t17_invbs_costs_5p_sec[i], t17_invbs_costs_95p_sec[i], 
        #                                      label=r"Recourse: 5$^{\rm{th}}-$95$^{\rm{th}}$ quantile range",
        #                                      alpha=0.4, color='#56B4E9')
        
    N_periods = 2
    for per in range(N_periods-1):
        tmp_dec_time = t17_inv_rec[str(per)].ds.time.values[-1] + ti
        ax[x_inds[i], y_inds[i]].vlines(tmp_dec_time, 0.0, max(t17_invbs_costs_mean_sec[i]) * 1.02, 
                                        color='grey', linestyle='dotted', linewidth=1.25)
    track += 1

    
fig.legend([cert, nodac_avg, dac_avg], 
           ["Certainty policy", "Average investment without DAC", "Average investment with DAC"], 
           bbox_to_anchor=(0.5, 0.), 
           loc='lower center', ncol=3, fancybox=True, shadow=True, fontsize=22)

#fig.subplots_adjust(wspace=0.4, hspace=0.2)

for i in range(len(x_inds)):
    #ax[x_inds[i], y_inds[i]].set_ylim(0, 1.2 * (0.5 * t17_inv_base.cbars.values[i] * t17_inv_base.abars.values[i]**2))
    ax[x_inds[i], y_inds[i]].set_title(t17_inv_recbs['0'].ds.sector.values[i])
    
for i in range(2):
    ax[i, 0].set_ylabel("Total investment effort\n(Billions of \$ yr$^{-1}$)")

for i in range(3):
    ax[1,y_inds[-3+i]].set_xlabel("Year")

#fig.tight_layout()
sns.despine(trim=True, offset=10)

fig.savefig(basefile + 't17-inv-base-rec-comparision-cost-secs-withbs.png', dpi=300)

plt.show()