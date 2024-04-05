# Reproduction code package for "How delayed learning about climate uncertainty impacts decarbonization investment strategies"

By: Adam Michael Bauer -- adammb4 [at] illinois [dot] edu

To cite our working paper that uses these codes: [Bauer, A. M., F. McIsaac, S. Hallegatte. *How Delayed Learning about Climate Uncertainty Impacts Decarbonization Investment Strategies*. World Bank Policy Research Working Paper No. WPS10743, World Bank Group, Washington DC, 2024.](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099829103282438373/idu1f2d86d77127091490d1a6df1dc342f15d10b)

## General package overview

This set of codes reproduces all of the figures and analysis carried out in *How delayed learning about climate uncertainty impacts decarbonization investment strategies*. This package uses Gurobi, a commerical nonlinear programming solver that is free for academics, but may not be free for everyone. (It is unclear to me if it's freely available for *all* researchers or just researchers *at universities*. I imagine Gurobi will handle this on a case-by-case basis, so just shoot their customer support staff an email and they can help.)

Each code is assigned a number corresponding to the figure it creates; the number matches the working paper figures. So code `01_xxx.py` makes Figure 1, which shows our calibration of the marginal abatement cost curves, and so on. Here is the full table for both versions:

| Figure Desired | Code to Run |
|----------|----------|
| Figure 1: Calibrating marginal abatement costs | `01_mac_calibration.sh` |
| Figure 2: Effect of delayed learning on aggregate policy cost | `02_effect_of_learning_low_linear.sh` |
| Figure 3: Effect of delayed learning on the temporal distribution of spending | `03_temporal_redistribution_low_linear.sh` |
| Figure 4: Effect of delayed learning on sectoral allocation of abatement investment | `04_sectoral_response.sh`|
| Figure 5: Effect of delayed learning on the carbon price | `05_carbon_price_response.sh` |
| Figure 6: Effect of delayed learning on aggregate policy cost including direct air capture technologies | `06_dac_effect_of_learning.sh` |
| Figure 7: Impact of delayed learning on sectoral allocation of abatement investment when direct air capture technologies are present | `07_dac_vs_no_dac_comp.sh` |
| Figure 8: Effect of delayed learning on aggregate policy cost, growing emissions baseline | `08_effect_of_learning_emis.sh` |
| Figure 9: Effect of delayed learning on the temporal distribution of spending, growing emissions baseline | `09_temporal_redistribution_emis.sh` |
| Figure 10: Effect of delayed learning on aggregate policy cost, high-bound calibration | `10_effect_of_learning_high_linear.sh` |
| Figure 11: Effect of delayed learning on the temporal distribution of spending, high-bound calibration | `11_temporal_redistribution_high_linear.sh` |
| Figure 12: Effect of delayed learning on aggregate policy cost, nonlinear calibration | `12_effect_of_learning_pow.sh` |
| Figure 13: Effect of delayed learning on the temporal distribution of spending, nonlinear calibration | `13_temporal_redistribution_pow.sh` |

If you're an academic, you can email Gurobi customer support to get a free academic license. It's easy to install, and once it's installed, I believe you'll be good to go to run the codes.

A final note is that you should consider using the `.yml` file provided in this directory to establish a virtual python environment that should include all of the necessary dependencies for the code to run smoothly. I recommend using `conda` to do this. 

## How to run the code

To run the codes, simply navigate to the `codes` directory and run the numbered code to recreate the desired figure. If you want to run the program `script_name`, you may need to execute:
```
    chmod +x script_name
```
to grant execution permissions (hence the `+x`) to the script you want to run.

As an example, if you want to recreate Figure 1 which shows our calibration of the marginal abatement cost curves, you would simply run:
```
./01_mac_calibration.sh
```
Notice the first bit of the above program name, `01_mac_calibration.sh`, matches the figure number we wanted to create, Figure 1.

All figures will be deposited into the `codes/figs` folder. To run indiviudal simulations, you can run any of the files in `simulation_mains`, and to make individual figures, you can run any file in the `figure_mains` folder. **Note:** You should run all scripts from the `codes` directory. As an example, let's say you want to run the `invBase_cvxpy_main.py` file in the `ar6_15` calibration, but not save the output. Then in your command line, you'd use:
```
python simulation_mains/invBase_cvxpy_main.py ar6_15 1 0
```

**Note:** You should be operating in the Python environment provided at the head directory. Without it, I make no guarantees any of this will run on your machine (and even then, well, mileage may vary...).

Last edited: 5 April, 2024.

