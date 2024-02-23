# Reproduction code package for "How delayed learning about climate uncertainty impacts decarbonization investment strategies"

By: Adam Michael Bauer -- adammb4 [at] illinois [dot] edu

This set of codes reproduces all of the figures and analysis carried out in *How delayed learning about climate uncertainty impacts decarbonization investment strategies*. There are two main folders in the head directory: `with-gurobi` and `without-gurobi`. Gurobi is a commerical nonlinear programming solver that is free for academics, but may not be free for everyone. (It is unclear to me if it's freely available for *all* researchers or just researchers *at universities*. I imagine Gurobi will handle this on a case-by-case basis, so just shoot their customer support staff an email and they can help.) Hence, I've made a version of the code that should (emphasis on *should*) work for everyone. I cannot guarantee that the output of both sets of codes will be identical, but the overall story of our paper should still hold. Note that in my preliminary testing of the `without-gurobi` codes, the programs ran much slower than their `with-gurobi` counterparts. 

Operationallly speaking, running both versions is identical. Each code is assigned a number corresponding to the figure it creates; the number matches the working paper figures. So code `01_mac_cal.py` makes Figure 1, which shows our calibration of the marginal abatement cost curves. 

If you're an academic, you can email Gurobi customer support to get a free academic license. It's easy to install, and once it's installed, I believe you'll be good to go to run the `with-gurobi` codes. If you're not using Gurobi, I did preliminary testing on the `without-gurobi` codes to make sure they all still work and generate the results, but your mileage may vary, as I had access to Gurobi during the course of this project via my affiliation with the University of Illinois Urbana-Champaign. My apologies for any headaches with the `without-gurobi` codes, and please do reach out to the above email if any problems to arise.

To cite our working paper that uses these codes: Bauer, A. M., F. McIsaac, S. Hallegatte. *In preparation*, 2024. (Will update once we have an actual working paper number, and hopefully a volume and page number from a journal!)

Last edited: February 2024.
