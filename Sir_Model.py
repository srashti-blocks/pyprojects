   
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the SIR model equations
def sir_model(y, t, beta, gamma, mu):
    S, I, R = y
    N = S + I + R  # Total population 
    dS_dt =    mu * N - beta * S * I / N - mu * S

    dI_dt = beta * S * I / N - gamma * I - mu * I
    dR_dt = gamma * I - mu * R

    return [dS_dt, dI_dt, dR_dt]

# Parameters99
beta = 0.2 # Transmission rate
gamma = 0.08 # Recovery rate
mu = 0.01    # Birth and death rate (endemic factor)

# Initial conditions
S0 = int(input("no. of suspectable: "))     # Initial susceptible individuals
I0 =  int(input("no. of infectious: "))      # Initial infected individuals
R0 =  int(input("no. of recovered: "))       # Initial recovered individuals
y0 = [S0, I0, R0]

# Time grid
t = np.linspace(0, 500, 500)  # 500 days

# Solve the differential equations
solution = odeint(sir_model, y0, t, args=(beta, gamma, mu))
S, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible', color='blue')
plt.plot(t, I, label='Infected', color='red')
plt.plot(t, R, label='Recovered', color='green')
plt.title('SIR Model with Endemic Equilibrium')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()  