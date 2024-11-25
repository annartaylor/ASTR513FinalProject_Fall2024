#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:47:30 2024

Fitting the Carbon Enthaply Values to a Shomate Equation 

@author: annartaylor
"""

import numpy as np
import matplotlib.pyplot as py
from scipy.optimize import curve_fit

# Read Janaff Table
carbon_janaff = np.genfromtxt("/Users/annartaylor/Library/CloudStorage/OneDrive-UniversityofArizona/Desktop/Fall 2024 Classes/ASTR513/JANAF_C.txt")
temperature = carbon_janaff[:,0]/1000 #K
enthalpy = carbon_janaff[:,4] #kJ/mol

#Define Shomate Equation
def shomate(t, A, B, C, D):
    return A*t + (B*t**2)/2 + (C*t**3)/3 + (D*t**4)/4 #- E/t + F - H

#Using scipy curve_fit to fit carbon data to shomate equation
popt, pcov = curve_fit(shomate, temperature, enthalpy)

#Plotting Fit 
fig = py.figure(figsize=(8, 6))
py.plot(temperature*1000, enthalpy, 'o', linewidth = 3, color = 'k' ,label = "JANAF Data")
py.plot(temperature*1000, shomate(temperature, *popt), color = 'deeppink', linewidth = 3, label = "Curve Fit")
py.xlabel("Temperature (K)")
py.ylabel("Standard Enthalpy (kJ/mol)")
py.legend()
py.show()

#Printing Shomate Coefficients
print("Shomate Coefficients: A = ", popt[0], ", B = ", popt[1], ", C = ", popt[2], ", D = ", popt[3], ", E = ", 0.0,  ", F = ", 0.0,  ", H = ", 0.0 )