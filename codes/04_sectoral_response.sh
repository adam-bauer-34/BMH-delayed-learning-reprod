#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBase_cvxpy_main.py ar6_15 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_15 N1_T40_B6 6 1 1

echo 'Making Figure 4...'

python figure_mains/sectoral_response.py