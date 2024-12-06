# this script computes chemical equilibrium state for given atmospheric composition, 
# following White, Johnson & Dantzig 1958
import numpy as np
import yaml
from matplotlib import pyplot as plt

### Constants SI units
NAvo = 6.022E23 #Avogadro's number
M_H2 = 2.016 # molar weight of H2 in g mol^-1
M_H2O = 18.015 # molar weight of H2 in g mol^-1
NA = 6.022E23 #Avogadro's number
R_U = 8.3145 # universal gas constant in KJ mol^-1 K^-1
k_B = 1.3806E-23 #Boltzmann's constant in J K^-1
### reference state
P_not = 1.0E5 # Pa

#def chem_eq(P, T, species, conc, reactions, thermo):
# inputs:
#       P, T: P-T profile
#       species: input species, list of species number
#       conc: input mixing ratios of species, n_species*n_z matrix 
#       reactions: reactions
#       thermo: thermo data

# find all the species present in the atmosphere after reactions
# read in the reaction file
reaction_list = np.genfromtxt("Reactions.dat", dtype="int")
#print(reaction_list)

# loop through all reactions, if all reactants exist, append both reactants and products to a new list
# major species: CO2, CO, SO2, N2
species = np.array([10,11,18,13]) # species number, not index
species_all = np.array([10,11,18,13])
print(species_all)

# read in P-T profile
data = np.genfromtxt('PT.dat')
height = data[:,0] # altitude in km
delta_z = (height[1] - height[0])*1.0E3 # vertical resolution in m
P = 10**data[:,1] # pressure in Pa
T = data[:,2] # temperature in K
#print('T:',T)
#print('P:',P)

# compute delta_f_G
# read in thermo data from inputs.yaml file
with open('Inputs.yaml', 'r') as file:
    species_data = yaml.safe_load(file)


# compute delta_H_m
# unit: KJ mol^-1
def compute_delta_H_m(T, shomate):
    t = T/1000.
    delta_H_m = shomate[0]*t + 1./2.*shomate[1]*t**2 + 1./3.*shomate[2]*t**3 + 1./4.*shomate[3]*t**4 - shomate[4]/t + shomate[5] - shomate[7]
    return delta_H_m

# compute S_m
# unit: J mol^-1 K^-1
def compute_S_m(T, shomate):
    t = T/1000.
    S_m = shomate[0]*np.log(t) + shomate[1]*t + 1./2.*shomate[2]*t**2 + 1./3.*shomate[3]*t**3 - 1./2.*shomate[4]/(t**2) + shomate[6]
    return S_m

# compute delta_f_H_m
# shomate coefficients for reference states of CHONS: C, H2, O2, N2,S
shomate_ref = np.zeros((5,8))
ind_ref = [27,1,3,12,28] # indices of ref: [27,1,3,12,18]

for i in range(0,len(ind_ref)):
    shomate_ref[i,:] = species_data['species'][ind_ref[i]]["thermo"]["data"][0]

# read in stoichiometric coefficients for each species
n_ref = np.genfromtxt("stoichiometric_coefficients.dat")
#print(n_ref)
mu = n_ref[species_all-1]
print(mu[0])

def compute_delta_f_H_m(delta_f_H_m_0, T, shomate, shomate_ref, mu):
    # mu is the stoichiometric coefficients required to form one mole of the chemical from reference state
    ### unit: kJ mol^-1
    t = T/1000.
    sum = 0.
    delta_H_m = compute_delta_H_m(T, shomate)
    for i in range(len(mu)):
        sum += mu[i]*compute_delta_H_m(T, shomate_ref[i,:])
    delta_f_H_m = delta_f_H_m_0 + delta_H_m - sum
    return delta_f_H_m

# compute delta_S_m
def compute_delta_f_S_m(T, shomate, shomate_ref, mu):
    ### unit: J mol^-1 K^-1
    t = T/1000.
    sum = np.zeros(len(height))
    S_m = compute_S_m(T, shomate)
    #print("S_m:", S_m)
    for i in range(len(mu)):
        sum += mu[i]*compute_S_m(T, shomate_ref[i,:])
    delta_f_S_m = S_m - sum
    #print("sum:", sum)
    #print("delta_f_S_m:", delta_f_S_m)
    return delta_f_S_m

### compute sum of delta_f_G_m in J mol^-1 at each altitude
### delta_f_G_m = delta_f_H_m*1E3 - T*delta_f_S_m + R_U*T*ln(mr*P/P0)
def sum_compute_delta_f_G_m(T, P, mr, delta_f_H_m, delta_f_S_m):
    sum = 0
    for i in range(len(mr)):
        delta_f_G_m = delta_f_H_m[i]*1.E3 - T*delta_f_S_m[i]+ R_U*T*np.log(mr[i]*P/P_not)
        sum += mr[i]*delta_f_G_m
    return sum

### compute delta_f_H_m and delta_f_S_m for each species at each altitude
delta_f_H_m_0 = np.zeros(len(species_all))
delta_f_H_m = np.zeros((len(height),len(species_all)))
delta_f_S_m = np.zeros((len(height),len(species_all)))

for i in range(0,len(species_all)):
    #print(mu[i])
    # read in the shomate coefficients for each species
    shomate = species_data['species'][species_all[i]-1]['thermo']['data'][0]
    # read in delta_f_H_m_0 for each species
    delta_f_H_m_0[i] = species_data['species'][species_all[i]-1]["thermo"]["ref_delta_H"]
    delta_f_H_m[:,i] = compute_delta_f_H_m(delta_f_H_m_0[i], T, shomate, shomate_ref, mu[i])
    delta_f_S_m[:,i] = compute_delta_f_S_m(T, shomate, shomate_ref, mu[i])
print("delta_f_H_m",delta_f_H_m[0,:])
print("delta_f_S_m",delta_f_S_m[0,:])

element = ['C', 'H', 'O', 'N', 'S']

# minimize sum(delta_f_G_m) using grid search
mr_eq = np.zeros((len(height),len(species))) # output mr for all species
mr_grid = np.logspace(-6,0,100) # search grid for mr
# loop over altitude
for i in range(0, len(height)):
    sum_delta_f_G = np.zeros((len(mr_grid), len(mr_grid), len(mr_grid)))
    for j in range(0, len(mr_grid)):
        for k in range(0, len(mr_grid)):
            for m in range(0, len(mr_grid)):
                if 1-(mr_grid[j] + mr_grid[k] + mr_grid[m]) >= 0: #+ mr_grid[m]
                    mr = [mr_grid[j], mr_grid[k], mr_grid[m], 1-(mr_grid[j] + mr_grid[k] + mr_grid[m])] # mr_grid[m]
                    sum_delta_f_G[j,k,m] = sum_compute_delta_f_G_m(T[i], P[i], mr, delta_f_H_m[i,:], delta_f_S_m[i,:])
    #print("min G:", sum_delta_f_G)
    mr_eq[i,0] =  mr_grid[np.where(sum_delta_f_G == np.min(sum_delta_f_G))[0]]
    mr_eq[i,1] =  mr_grid[np.where(sum_delta_f_G == np.min(sum_delta_f_G))[1]]
    mr_eq[i,2] =  mr_grid[np.where(sum_delta_f_G == np.min(sum_delta_f_G))[2]]
    mr_eq[i,-1] = 1 - np.sum(mr_eq[i,0:-1])       
print("mr eq:", mr_eq[0,:])

colors = ['deeppink', 'orange', 'purple', 'blue']
# plot mr profiles for main species at chemcial equilibrium
fig, ax = plt.subplots()
ax.set_xscale('log')
for i in range(0,len(species_all)):
    ax.plot(mr_eq[:,i], height, color = colors[i], linewidth = 2,  label = species_data['species'][species_all[i]-1]['name'])
ax.set_xlabel("Mixing ratio")
ax.set_ylabel('Height (km)')
plt.legend()
plt.show()
