#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base model...'

python simulation_mains/invBase_cvxpy_main.py ar6_17 1 1

echo 'Models with uncertainty...'
echo 'No DAC...'

python simulation_mains/invRec_cvxpy_main.py ar6_17 N1_T30_B8 3 1

echo 'With DAC...'
python simulation_mains/invRecBS_cvxpy_main.py ar6bs_17 N1_T30_B8 3 1

echo 'Making Figure 7...'

python figure_mains/comp_dac_no_dac.py