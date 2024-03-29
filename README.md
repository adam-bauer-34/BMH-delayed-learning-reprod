# Reproduction code package for "How delayed learning about climate uncertainty impacts decarbonization investment strategies"

By: Adam Michael Bauer -- adammb4 [at] illinois [dot] edu

This set of codes reproduces all of the figures and analysis carried out in *How delayed learning about climate uncertainty impacts decarbonization investment strategies*. There are two main folders in the head directory: `with-gurobi` and `without-gurobi`. Gurobi is a commerical nonlinear programming solver that is free for academics, but may not be free for everyone. (It is unclear to me if it's freely available for *all* researchers or just researchers *at universities*. I imagine Gurobi will handle this on a case-by-case basis, so just shoot their customer support staff an email and they can help.) Hence, I've made a version of the code that should (emphasis on *should*) work for everyone. I cannot guarantee that the output of both sets of codes will be identical owing to the various approximations a given solver would make along the way towards finding the optimal solution, but the overall story of our paper should still hold. Note that in my preliminary testing of the `without-gurobi` codes, the programs ran much slower than their `with-gurobi` counterparts. 

Operationallly speaking, running both versions is identical. Each code is assigned a number corresponding to the figure it creates; the number matches the working paper figures. So code `01_xxx.py` makes Figure 1, which shows our calibration of the marginal abatement cost curves. 

If you're an academic, you can email Gurobi customer support to get a free academic license. It's easy to install, and once it's installed, I believe you'll be good to go to run the `with-gurobi` codes. If you're not using Gurobi, I did preliminary testing on the `without-gurobi` codes to make sure they all still work and generate the results, but your mileage may vary, as I had access to Gurobi during the course of this project via my affiliation with the University of Illinois Urbana-Champaign. My apologies for any headaches with the `without-gurobi` codes, and please do reach out to the above email if any problems to arise.

A final note is that you should consider using the `.yml` file provided in this directory to establish a virtual python environment that should include all of the necessary dependencies for the code to run smoothly. I recommend using `conda` to do this. 

To cite our working paper that uses these codes: [Bauer, A. M., F. McIsaac, S. Hallegatte. *How Delayed Learning about Climate Uncertainty Impacts Decarbonization Investment Strategies*. World Bank Policy Research Working Paper No. WPS10743, World Bank Group, Washington DC, 2024.](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099829103282438373/idu1f2d86d77127091490d1a6df1dc342f15d10b)

Last edited: 29 March, 2024.
