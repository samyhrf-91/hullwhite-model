# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from numpy import random
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm
from math import *
from tqdm.notebook import tqdm

#%% Fix the seed to get plots for the same path
random.seed(4)

#%% Data import


yield_df = pd.read_csv(r'C:\Users\samyh\OneDrive\Documents\Financial derivatives\Project 2\yield_curve_spline.csv')

yield_rates = yield_df.to_numpy()

#%% Hull_White parameters

a = 0.10
sigma = 0.01

first_date = yield_rates[0,0]
last_date = yield_rates[10958,0]

f_date = datetime.strptime(first_date, '%Y-%m-%d')
l_date = datetime.strptime(last_date, '%Y-%m-%d')

# T = (l_date - f_date).days / 365 # gives 30.02 instead of exactly 30
T = 30
deltaT = 1/365

N = len(yield_rates)

#%%

# Yield values
y_curve = np.array(yield_rates[:,1], dtype = float)
ymax = y_curve[-1]

timesN = np.linspace(0,T,N)

#Forward rates
fwd_curve = (ymax*T - y_curve * timesN)
fwd_curve[0:-1] /= (T-timesN[0:-1])

# Prices
prices = np.exp(-integrate.cumulative_trapezoid(fwd_curve,dx=1/365))

#%% Plot rate curves

plt.figure()
plt.plot(timesN,y_curve, label = 'Yield')
plt.plot(timesN,fwd_curve, label = 'Forward')
#plt.plot(timesN,prices)
plt.xlabel('Time (years)')
plt.ylabel('Rates')
plt.title('Yield and forward rates up to the final time')
plt.legend()

plt.figure()
plt.plot(timesN[0:-1],prices, label = 'prices')
plt.title('Prices up to the final time')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.show()

plt.figure()
plt.plot(timesN[0:50],prices[0:50], label = 'prices')
plt.xlabel('Time (years)')
plt.ylabel('Prices')
plt.title('Prices for the next 50 days')
plt.show()


#%% theta in terms of the forward rate curve

dfdT = np.diff(fwd_curve)/deltaT
def theta(idx, fwd):
    return a*fwd[idx] + dfdT[idx] + sigma**2 / (2*a) * (1- np.exp(-2*a*(idx*deltaT)))

#def theta(idx, fwd):
#    return a*fwd[idx] + dfdT[idx] + sigma**2 / (2*a) * (1 - 2 * np.exp(-a*(idx*deltaT)) + np.exp(-2*a*(idx*deltaT))) + sigma**2 / 2 * np.exp(-a*(idx*deltaT))



#%% Simulation of the short rate

def simulate_r(r0, N, thetas):
    r = np.zeros(N)
    r[0] = r0
    
    for i in range(1,N):
        W = random.normal()
        r[i] = r[i-1] + (thetas[i]-a*r[i-1])*deltaT + sigma*W *np.sqrt(deltaT)
    return r
times = np.linspace(0,T-deltaT,N-1)

thetas = np.zeros(N-1)
for i in range(0,N-1):
    thetas[i] = theta(i,fwd_curve)

r0 = fwd_curve[0]
r = simulate_r(r0,N-1, thetas)

plt.figure()
plt.plot(times,r, label = 'Short rate')
plt.plot(times,thetas/a, label = 'Theta(t)/a')
plt.legend()
plt.show()

#%% Monte carlo method for price computation
M = 1000

# Generate samples
r_list = np.zeros((N-1,M))

for i in range(M):
    r_sample = simulate_r(r0,N-1, thetas)
    r_list[:,i] = r_sample
    
# Monte-Carlo method
price = 0
price_list = np.zeros(M)
for i in range(M):
    integ = integrate.simpson(r_list[:,i],dx=deltaT)
    price_sample = np.exp(-integ)
    price += price_sample
    price_list[i] = price/(i+1)
    
price /= M

samples = range(0,M)
plt.figure()
plt.plot(samples,price_list, label = 'Price approximation')
plt.plot(samples,prices[-1]*np.ones(M), label = 'Price from data')
plt.title('Evolution of the price approximation as more samples are generated')
plt.legend()
plt.show()

#%% price in terms of forward curve

#forward curve for different maturities
def simulate_fwd(N,Tj, thetas):
    fwd = np.zeros(N-1)
    fwd[0] = fwd_curve[Tj]  # Initialisation of the forward curve f(0, Tj)
    exp_a = np.exp(-a * np.arange(Tj))  
    sqrt_deltaT = np.sqrt(deltaT)  
    sigma2_a = (sigma ** 2) / a  
    W = np.random.normal(size=Tj)
    
    for i in range(1, Tj):
        qt = exp_a[Tj - i]
        drift = sigma2_a * (qt - qt**2) + 2 * thetas[i-1] * qt
        diffusion = sigma * qt * W[i-1]
        fwd[i] = fwd[i-1] + drift * deltaT + diffusion * sqrt_deltaT

    return fwd
#%% discretisation to decrease the computation time
M=1000
N_bis=ceil(N/2)
times_bis = times[::2]
thetas_bis = thetas[::2]




#%% monte carlo simulation for forward rate curve

def forward_rate(N,times,thetas,M):
    structure = np.zeros((N-1,N-1)) #dtypes=np.float32
    for Tj in tqdm(range(len(times)-1)):
        fwd_sum = np.zeros(N-1)
        for i in range(M):
            fwd_sum += simulate_fwd(N,Tj,thetas)
        structure[:,Tj] = fwd_sum/M
    return structure

structure = forward_rate(N, times,thetas, M)
term_struct=np.zeros((N-1,1))

#%% plot of the term structure

for i in range (N-1):
    term_struct[i]= integrate.simpson(structure[i,:],dx=deltaT)
    term_struct[i]=np.exp(-term_struct[i])
    
plt.figure()
plt.plot(times,term_struct, label = 'prices')
plt.title('Term structure p(t,T*) for Hull-White model')
plt.xlabel('Time (years)')
plt.ylabel('Spot price')
plt.show()



#%% caplets and swaptions
def zbp_price(p,t,K,F,P):
    C = (1 / (2 * a)) * (np.exp(-2 * a * t[P] + 2 * a * t[F]) + 1 - np.exp(-2 * a * t[P]) - np.exp(-2 * a * t[F])) - (2 / a) * (np.exp(-2 * a * t[P] + 2 * a * t[F]) - np.exp(-2 * a * t[P] - 2 * a * t[F]))
    pP=p[P]
    pF=p[F]
    phi_d=norm.cdf((a/(sigma*C))*(((sigma**2*C)/(2*a**2))+np.log(K*pF/pP)))
    return K*pF*phi_d-pP*(1-phi_d)

def caplet(p,t,i,K):
    Kprime=1+K*deltaT
    
    return Kprime*zbp_price(p,t,1/Kprime,i-1,i)


K=0.8

def caplets(K,prices,timesN):
    caplets=np.zeros((N,1))
    for i in range(1,N-1):
        caplets[i]=caplet(prices,timesN,i,K)
    caplets[0]=(1+K*deltaT)*zbp_price(prices,timesN,K,0,0)
    caplets[N-1]=(1+K*deltaT)*zbp_price(prices,timesN,K,N-2,N-2)
    return caplets