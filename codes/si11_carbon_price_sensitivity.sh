#!/bin/bash

echo 'Carrying out simulations...'

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_17 N1_T30_B8 8 3 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_2 N1_T30_B8 8 3 1

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6hi_17 N1_T30_B8 8 3 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6hi_2 N1_T30_B8 8 3 1

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6pow_17 N1_T30_B8 8 3 1
python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6pow_2 N1_T30_B8 8 3 1

echo 'Making SI Figure 11...'

python figure_mains/carbon_price_sensitivity.py
