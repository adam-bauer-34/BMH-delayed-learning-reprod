"""Marginal abatement cost curve main file.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
2.21.2024
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from src.presets import get_presets
from scipy.optimize import curve_fit

def power_law(x, a, b):
    """Power law fit for data.

    Parameters
    ----------
    x: array
        values we're fitting the curve to

    a: float
        scaling coefficient

    b: float
        power

    Returns
    -------
    a * x**b: array
        power law of data
    """

    return a * x**b

# get presets from source function
presets, basefile = get_presets()
plt.rcParams.update(presets)

# data for each sector
# NOTE: I manually wrote in this data from a very cluttered spreadsheet. I've included the spreadsheet
# in the `data/cal` directory for inquiring minds.
prices = np.array([0, 20, 50, 100, 200])
prices_trunc = prices[:-1] # some sectors have no increase in mitigation potential from 100 -> 200 price

energy = np.array([0, 7.09, 8.98, 10.96, 11.99])
buildings = np.array([0, 1.54, 1.79, 2.04, 3.2]) # direct AND indirect emissions reductions included
ag = np.array([0, 0.85, 1.58, 4.07]) # example of no increase in abatement from 100 -> 200
forestry = np.array([0, 2.81, 3.5, 7.37, 8.25])
transport = np.array([0, 3.27, 3.51, 3.74])
industry = np.array([0, 1.14, 4.11, 5.16, 5.47])
waste = np.array([0, 0.49, 0.59, 0.67, 0.82])

# do power law fit to above data
## set an upper bound to ensure we get nothing higher than a quadratic -- in practice, you could
## get rid of the upper bound or increase it, but the fit arguably gets worse. also, no chance you
## get anything to solve with terms in the objective that are to the power higher than 4 or 5... but
## you can try! 
pow_ub = 2.0
energy_fit = curve_fit(power_law, xdata=energy, ydata=prices, bounds=([0, 1], [np.inf, pow_ub]))
buildings_fit = curve_fit(power_law, xdata=buildings, ydata=prices, bounds=([0, 1], [np.inf, pow_ub]))
ag_fit = curve_fit(power_law, xdata=ag, ydata=prices_trunc, bounds=([0, 1], [np.inf, pow_ub]))
forestry_fit = curve_fit(power_law, xdata=forestry, ydata=prices, bounds=([0, 1], [np.inf, pow_ub]))
transport_fit = curve_fit(power_law, xdata=transport, ydata=prices_trunc, bounds=([0, 1], [np.inf, pow_ub]))
industry_fit = curve_fit(power_law, xdata=industry, ydata=prices, bounds=([0, 1], [np.inf, pow_ub]))
waste_fit = curve_fit(power_law, xdata=waste, ydata=prices, bounds=([0, 1], [np.inf, pow_ub]))

# extract fitting parameters
energy_a, energy_b = energy_fit[0]
buildings_a, buildings_b = buildings_fit[0]
ag_a, ag_b = ag_fit[0]
forestry_a, forestry_b = forestry_fit[0]
transport_a, transport_b = transport_fit[0]
industry_a, industry_b = industry_fit[0]
waste_a, waste_b = waste_fit[0]

# make abatement potentials for each sector
d_pot = 0.05
energy_pot = np.arange(0, energy[-1]+d_pot, d_pot)
buildings_pot = np.arange(0, buildings[-1]+d_pot, d_pot)
ag_pot = np.arange(0, ag[-1]+d_pot, d_pot)
forestry_pot = np.arange(0, forestry[-1]+d_pot, d_pot)
transport_pot = np.arange(0, transport[-1]+d_pot, d_pot)
industry_pot = np.arange(0, industry[-1]+d_pot, d_pot)
waste_pot = np.arange(0, waste[-1]+d_pot, d_pot)

# make figure!
fig, ax = plt.subplots(2,4, figsize=(25, 15), sharey=True)

ax[0,0].set_title("Energy")
ax[0,0].scatter(energy, prices)
ax[0,0].plot(energy_pot, energy_a * energy_pot**energy_b, label='scipy')
ax[0,0].plot(energy_pot, (20.) * energy[1]**(-1) * energy_pot, label='vs18 method')
ax[0,0].plot(energy_pot, (200.) * energy[-1]**(-1) * energy_pot, label='vs18 method')

ax[0,1].set_title("Industry")
ax[0,1].scatter(industry, prices)
ax[0,1].plot(industry_pot, industry_a * industry_pot**industry_b, label='scipy')
ax[0,1].plot(industry_pot, (20.) * industry[1]**(-1) * industry_pot, label='vs18 method')
ax[0,1].plot(industry_pot, (200.) * industry[-1]**(-1) * industry_pot, label='vs18 method')

ax[0,2].set_title("Buildings")
ax[0,2].scatter(buildings, prices)
ax[0,2].plot(buildings_pot, buildings_a * buildings_pot**buildings_b, label='scipy')
ax[0,2].plot(buildings_pot, (20.) * buildings[1]**(-1) * buildings_pot, label='vs18 method')
ax[0,2].plot(buildings_pot, (200.) * buildings[-1]**(-1) * buildings_pot, label='vs18 method')

ax[0,3].set_title("Agriculture")
ax[0,3].scatter(ag, prices_trunc)
ax[0,3].plot(ag_pot, ag_a * ag_pot**ag_b, label='scipy')
ax[0,3].plot(ag_pot, (20.) * ag[1]**(-1) * ag_pot, label='vs18 method')
ax[0,3].plot(ag_pot, (100.) * ag[-1]**(-1) * ag_pot, label='vs18 method')

ax[1,0].set_title("Transport")
ax[1,0].scatter(transport, prices_trunc)
ax[1,0].plot(transport_pot, transport_a * transport_pot**transport_b, label='scipy')
ax[1,0].plot(transport_pot, (20.) * transport[1]**(-1) * transport_pot, label='vs18 method')
ax[1,0].plot(transport_pot, (100.) * transport[-1]**(-1) * transport_pot, label='vs18 method')

ax[1,1].set_title("Forestry")
ax[1,1].scatter(forestry, prices)
ax[1,1].plot(forestry_pot, forestry_a * forestry_pot**forestry_b, label='scipy')
ax[1,1].plot(forestry_pot, (20.) * forestry[1]**(-1) * forestry_pot, label='vs18 method')
ax[1,1].plot(forestry_pot, (200.) * forestry[-1]**(-1) * forestry_pot, label='vs18 method')

ax[1,2].set_title("Waste")
ax[1,2].scatter(waste, prices, label='IPCC AR6 Data')
ax[1,2].plot(waste_pot, waste_a * waste_pot**waste_b, label='Nonlinear')
ax[1,2].plot(waste_pot, (20.) * waste[1]**(-1) * waste_pot, label='Low cost, linear')
ax[1,2].plot(waste_pot, (200.) * waste[-1]**(-1) * waste_pot, label='High cost, linear')


for i in range(2):
    ax[i,0].set_ylabel("Marginal cost (2020 US\$ tCO$_2^{-1}$)")
    for j in range(4):
        ax[i,j].set_xlabel("Abatement pot. (GtCO$_2$ yr$^{-1}$)")
        ax[i,j].tick_params(axis='both')
        ax[i,j].set_ylim(0,210)
        
ax[1,3].axis('off')

ax[1,2].legend(loc='lower right', bbox_to_anchor=(2.1, 0.5), ncol=1, fontsize=18)
fig.subplots_adjust(hspace=0.3)

sns.despine(trim=True, offset=5)

fig.savefig(''.join([basefile, "mac-cal.png"]), dpi=400)

plt.show()