import numpy as np

S0 = 100      # initial stock price
K = 110       # strike price
T = 0.5       # time to maturity in years
r = 0.06      # annual risk-free rate
N = 1000       # number of time steps
sigma = 0.3   # Annualised stock price volatility
opttype = 'P' # Option Type 'C' or 'P'

def European(K, T, S0, r, N, sigma, opttype):
    dt = T/N
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    pu = ((np.exp(r*dt/2)-np.exp(-sigma*np.sqrt(dt/2))) / (np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2))))**2
    pd = ((np.exp(sigma*np.sqrt(dt/2))-np.exp(r*dt/2)) / (np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2))))**2
    pm = 1 - pu - pd
    disc = np.exp(-r*dt)

    logu = np.log(u)
    logd = -logu
    center = N

    logS = np.array([np.log(S0)+(j-center)*logu for j in range(2*N+1)])
    S = np.exp(logS)
    V = np.maximum(K-S, 0)

    for step in range(N, 0, -1):
        nextV = np.zeros(2*step-1)
        for i in range(2*step-1):
            nextV[i] = disc*(pu*V[i]+pm*V[i+1]+pd*V[i+2])
        V = nextV
    
    return V[0]

def American(K, T, S0, r, N, sigma, opttype):
    dt = T/N
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    pu = ((np.exp(r*dt/2)-np.exp(-sigma*np.sqrt(dt/2))) / (np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2))))**2
    pd = ((np.exp(sigma*np.sqrt(dt/2))-np.exp(r*dt/2)) / (np.exp(sigma*np.sqrt(dt/2))-np.exp(-sigma*np.sqrt(dt/2))))**2
    pm = 1 - pu - pd
    disc = np.exp(-r*dt)

    logu = np.log(u)
    logd = -logu
    center = N

    logS = np.array([np.log(S0)+(j-center)*logu for j in range(2*N+1)])
    S = np.exp(logS)
    V = np.maximum(K-S, 0)

    for step in range(N, 0, -1):
        newV = np.zeros(2*step-1)
        for i in range(2*step-1):
            newS = np.exp(np.log(S0)+(i-(step-1))*logu)
            cont =  disc*(pu*V[i]+pm*V[i+1]+pd*V[i+2])
            intrinsic = max(K-newS, 0)
            newV[i] = max(cont, intrinsic)
        V = newV
    
    return V[0]

print(European(K, T, S0, r, N, sigma, opttype))
print(American(K, T, S0, r, N, sigma, opttype))
