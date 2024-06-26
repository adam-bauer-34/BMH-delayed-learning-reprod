#!/bin/bash

echo 'Carrying out simulations...'
echo 'Base models...'

python simulation_mains/invBaseEmis_cvxpy_main.py ar6emis_17 1 1
python simulation_mains/invBaseEmis_cvxpy_main.py ar6emis_2 1 1

echo 'Models with uncertainty...'

python simulation_mains/invRecEmis_RiskPrem_cvxpy_main.py ar6emis_17 N1_T30_B8 8 3 1
python simulation_mains/invRecEmis_RiskPrem_cvxpy_main.py ar6emis_2 N1_T30_B8 8 3 1

echo 'Making SI Figure 4...'

python figure_mains/temporal_redist.py emis
