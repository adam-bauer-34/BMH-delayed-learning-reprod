#!/bin/bash

echo 'Carrying out simulations...'

echo 'Models with uncertainty...'

python simulation_mains/invRec_RiskPrem_cvxpy_main.py ar6_17 N1_T30_B8 8 3 1

echo 'Making Figure 5...'

python figure_mains/sectoral_response.py
