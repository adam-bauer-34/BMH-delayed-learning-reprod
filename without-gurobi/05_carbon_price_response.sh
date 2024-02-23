#!/bin/bash

echo 'Carrying out simulations...'
echo 'Models with uncertainty...'
echo 'WARNING: This one might take a while, since we have to draw 500 samples from the RCB distirbution rather than use an approximation method!'

python simulation_mains/invRec_RiskPremShort_cvxpy_main.py ar6_15 N1_T40_B500 500 0 1

echo 'Making Figure 5...'

python figure_mains/carbon_price_impact.py