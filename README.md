The aim of this project is to calibrate a one-factor Hull-White model to market data.
To implement this project, one needs to download one of the csv files in this repository, each one corresponding to bond data.

Some computations useful for the project can be found in the file Project_2__Financial derivatives. 
In particular, one will find :
- the SDE of the short rate r(t) and the forward rate f(t,T)
- the expression of theta(t) in terms of the forward curve
- the pricing formula for caplets and swaptions using zero-bond put options


The calculation of the term structure p(t,T), using a Monte Carlo takes quite some time (10 hours for 1000 trajectories, with a daily discretisation for a 30-year time horizon), despite using vector operations.
To reduce calculation time, one can modify the time horizon (for example one can take a 5-year or 10-year time horizon).
