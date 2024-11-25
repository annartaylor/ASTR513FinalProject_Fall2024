# this script computes chemical equilibrium state for given atmospheric composition, 
# following White, Johnson & Dantzig 1958
import numpy as np
import yaml

### Constants
NAvo = 6.022E23 #Avogadro's number
M_H2 = 2.016 # molar weight of H2 in g mol^-1
M_H2O = 18.015 # molar weight of H2 in g mol^-1
NA = 6.022E23 #Avogadro's number
R_U = 8.3145 # universal gas constant in J mol^-1 K^-1
k_B = 1.3806E-23 #Boltzmann's constant in J K^-1
### reference state
P_not = 1.0E5 # hPa

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
species = np.array([2,3,4,7,9,10,11,13,14,18,27])
species_append = species
species_all = []

for i in range(0, len(reaction_list)):
    if reaction_list[i][2] == 0:
        if (reaction_list[i][0] in species) and (reaction_list[i][1] in species):
            species_append = np.append(species_append,reaction_list[i][-2:])
    else:
        if (reaction_list[i][0] in species) and (reaction_list[i][1] in species) and (reaction_list[i][2] in species):
            species_append = np.append(species_append,reaction_list[i][-2:])
#print(species)
species_all = list(set(species_append[np.where(species_append>0)]))
print(species_all)

# compute delta_f_G
# read in thermo data from inputs.yaml file
with open('Inputs.yaml', 'r') as file:
    species_data = yaml.safe_load(file)

shomate = species_data['species'][species[0]]["thermo"]["data"][0]

# compute delta_H_m
def compute_delta_H_m(T, shomate):
    t = T/1000.
    delta_H_m = shomate[0]*t + 1./2.*shomate[1]*t**2 + 1./3.*shomate[2]*t**3 + 1./4.*shomate[3]*t**4 - shomate[4]/t + shomate[5] - shomate[7]
    return delta_H_m

# compute S_m
def compute_S_m(T, shomate):
    t = T/1000.
    S_m = shomate[0]*np.log(t) + shomate[1]*t + 1./2.*shomate[2]*t**2 + 1./3.*shomate[3]*t**3 - 1./2.*shomate[4]/(t**2) + shomate[6]
    return S_m

# compute delta_f_H_m
# shomate coefficients for reference states of CHONS: C, H2, O2, N2,S
# indices of ref: [27,1,3,12,18]
shomate_ref = np.zeros((5,8))
ind_ref = [27,1,3,12,28]

for i in range(0,len(ind_ref)):
    shomate_ref[i,:] = species_data['species'][ind_ref[i]]["thermo"]["data"][0]

#print(shomate_ref)

delta_f_H_m_0 = species_data['species'][0]["thermo"]["ref_delta_H"]
#print(delta_f_H_m_0)

# read in stoichiometric coefficients for each species
n_ref = np.genfromtxt("stoichiometric_coefficients.dat")
#print(n_ref)
mu = n_ref[species]

def compute_delta_f_H_m(delta_f_H_m_0, T, shomate_ref, mu):
    # mu is the stoichiometric coefficients required to form one mole of the chemical from reference state
    ### unit: kJ mol^-1
    t = T/1000.
    sum = 0.
    delta_H_m = compute_delta_H_m(T, shomate_ref, mu)
    for i in range(len(mu)):
        sum += mu[i]*compute_delta_H_m(T, shomate_ref[i,0], shomate_ref[i,1], shomate_ref[i,2], shomate_ref[i,3], shomate_ref[i,4], shomate_ref[i,5], shomate_ref[i,7])
    delta_f_H_m = delta_f_H_m_0 + delta_H_m - sum
    return delta_f_H_m

# compute delta_S_m
def compute_delta_f_S_m(T, shomate_ref, mu):
    ### unit: J mol^-1
    t = T/1000.
    sum = 0.
    S_m = compute_S_m(T, shomate_ref)
    for i in range(len(mu)):
        sum += mu[i]*compute_S_m(T, shomate_ref[i,0], shomate_ref[i,1], shomate_ref[i,2], shomate_ref[i,3], shomate_ref[i,4], shomate_ref[i,6] )
    delta_f_S_m = S_m - sum
    return delta_f_S_m

### compute delta_f_G_m at P, T
### delta_f_G_m = delta_f_H_m - T*delta_f_S_m*1E-3 + R_U*T*ln(mr*P/P0)
def compute_delta_f_G_m(T, P, mr, delta_f_H_m, delta_f_S_m):
    delta_f_G_m = delta_f_H_m  - T*delta_f_S_m *1E-3 + R_U*T*np.log(mr*P/P_not)
    return delta_f_G_m

# minimize delta_f_G to get mixing ratios of all speceis
# sum_delta_f_G = sum(mr_i*delta_f_G_m)
# conservation of element
mr_ini = np.array([1.e-6,1.e-6,1.e-5,1.e-5,5.e-6,0.965,3.e-5,0.034,3.e-9,1.5e-4,5.e-6])

element = ['C', 'H', 'O', 'N', 'S']

def sum_mr_element(species, mr, element):
    # compute the sum of mr for each element: CHONS
    sum = np.zeros(len(element))

    for i in range(0,len(element)):
        for j in range(0,len(species)):
            sum_mr_element[i] +=  mr[j]*species_data['species'][species[j]]['composition'].get(element[i])
    return sum

# initialize the mixing ratios at equilibrium of all speceis using the initial condition
mr_eq_ini = np.zeros_like(species_all, dtype = 'float64')

for i in range(0, len(species_all)):
    if species_all[i] in species:
        mr_eq_ini[i] = mr_ini[np.where(species == species_all[i])[0][0]]

print(mr_eq_ini)

# minimize sum(delta_f_G_m) using gradient descent
def gradient_descent(P, T, delta_f_H_m, delta_f_S_m):
    # inputs:
    #        P, T
    #        delta_f_H_m and delta_f_S_m at each altitude
    # initialization of variables, step, threshold for convergence
    n_iter = 1000
    stepsize = 1.e-10
    threshold = 1e-20
    mr = mr_eq_ini # mr at last iteration
    mr_new = np.zeros_like(mr) # current mr
    gradient = np.zeros_like(species_all, dtype = 'float64')

    # interation
    for i in range(0, n_iter):
        # compute gradient and stepping
        for i in range(0, len(species_all)):
            gradient[i] = delta_f_H_m[i]-T*delta_f_S_m[i]+R_U*T*np.log(mr[i]*P/P_not)
            mr_new[i] = mr[i] - stepsize*gradient
            # compare with the threshold for convergence
            if np.max(abs(mr_new - mr)) < threshold:
                break
    # check conservation of element/mass balance
    sum_mr_element_eq = sum_mr_element(species_all, mr_new, element)
    if sum_mr_element > threshold:
        print("ERROR:Conservation of element not satisfied!")
