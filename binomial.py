import numpy as np
from functools import wraps

S0 = 100      # initial stock price
K = 110       # strike price
T = 0.5       # time to maturity in years
r = 0.06      # annual risk-free rate
N = 1000       # number of time steps
sigma = 0.3   # Annualised stock price volatility
opttype = 'P' # Option Type 'C' or 'P'

def europeanF(K, T, S0, r, N, sigma, opttype):
    #precompute constants
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    q = (np.exp(r*dt)-d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise asset prices at maturity - Time step N
    S = S0*d**(np.arange(N,-1,-1))*u**(np.arange(0,N+1,1))

    # initialise option values at maturity
    if opttype == 'C':
        C = np.maximum(0, S - K)
    else:
        C = np.maximum(0, K - S)

    # step backwards through tree
    for i in np.arange(N, 0, -1):
        C = disc*(q*C[1:i+1]+(1-q)*C[0:i])

    return C[0]

def americanF(K, T, S0, r, N, sigma, opttype):
    #precompute constants
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    q = (np.exp(r*dt)-d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise stock prices at maturity
    S = S0*d**(np.arange(N, -1, -1))*u**(np.arange(0, N+1, 1))

    # option payoff
    if opttype == 'P':
        C = np.maximum(0, K-S)
    else:
        C = np.maximum(0, S-K)

    # backward recursion through the tree
    for i in np.arange(N-1, -1, -1):
        S = S0*d**(np.arange(i, -1, -1))*u**(np.arange(0, i+1, 1))
        C[:i+1] = disc*(q*C[1:i+2]+(1-q)*C[0:i+1])
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K-S)
        else:
            C = np.maximum(C, S-K)

    return C[0]

print(americanF(K, T, S0, r, N, sigma, opttype))
print(europeanF(K, T, S0, r, N, sigma, opttype))
