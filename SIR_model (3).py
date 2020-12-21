#!/usr/bin/env python
# coding: utf-8

# In[271]:


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# # Let’s run the basic SIR model

# In[272]:



def deriv(y, t, N, beta, gamma, delta, alpha, rho):
    S, E, I, R, D = y
    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


# In[273]:


#Based on Swedish data
L = 30 #Days before restrictions
N = 10_000_000 #Population
D = 6.0 # infections lasts six days
gamma = 1.0 / D  #Recovers per time unit
delta = 1.0 / 5.0  # incubation period of five days
def R_0(t): #Infected per person before and after restrictions
    return 5.0 if t < L else 2
def beta(t): #Infected per time unit
    return R_0(t) * gamma

alpha = 0.021  # 2.1% death rate
rho = 1/9  # 9 days from infection until death
S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed


# In[274]:


def plotsir(t, S, E, I, R, D):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
  ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
  ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
  ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')
  #Dashed line = Total population
    
  ax.set_xlabel('Time (days)')
  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
    
  plt.savefig("Plot.png")
  plt.show();


# In[275]:


t = np.linspace(0, 314, 315) # Grid of time points (in days)
y0 = S0, E0, I0, R0, D0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
S, E, I, R, D = ret.T #D = deaths


# plot the graph

# In[276]:


plotsir(t, S, E, I, R, D)

