#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base model...'

python simulation_mains/invBase_cvxpy_main.py ar6_15 1 1

echo 'Models with uncertainty...'
echo 'No DAC...'

python simulation_mains/invRec_cvxpy_main.py ar6_15 N1_T40_B6 6 1 1

echo 'With DAC...'
python simulation_mains/invRecBS_cvxpy_main.py ar6bs_15 N1_T40_B6 6 1 1

echo 'Making Figure 7...'

python figure_mains/comp_dac_no_dac.py