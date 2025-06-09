# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:58:26 2025

@author: User
"""

from scipy.special import hyp2f1
import numpy as np

class CFT_2D:
    def __init__(self, d_phi):
        self.d_phi = d_phi

    def g_spec(self, z, zb, spec_list):
        # pre_f = 1 if s !=0 else 1/2
        def pre_f(s):
            return 1 if s !=0 else 1/2
        
        # Vectorized calculation using numpy
        return np.array([pre_f(s)*
                (z**((d+s)/2) * zb**((d-s)/2) * hyp2f1((d+s)/2, (d+s)/2, (d+s), z) * hyp2f1((d-s)/2, (d-s)/2, (d-s), zb) +
                zb**((d+s)/2) * z**((d-s)/2) * hyp2f1((d+s)/2, (d+s)/2, (d+s), zb) * hyp2f1((d-s)/2, (d-s)/2, (d-s), z)) 
                for s, d in spec_list
                ])
    
    def G(self, z_list, zb_list, spec_list):
        return np.array((np.abs(z_list-1)**(2*self.d_phi)) * self.g_spec(z_list, zb_list, spec_list) - 
                        np.abs(z_list)**(2*self.d_phi) * self.g_spec(1-z_list, 1-zb_list, spec_list)).T
    
    def vacuum(self, z_list):
        return np.array(np.abs(z_list-1)**(2*self.d_phi)-np.abs(z_list)**(2*self.d_phi))
    
    def crossing_eq(self, z_list, zb_list, ope_list, spec_list):
        ope_list = np.array(ope_list)
        spec_list = np.array(spec_list)
        
        return np.sum(ope_list * self.G(z_list, zb_list, spec_list), axis=1) + self.vacuum(z_list)
    
    
    

if __name__ == '__main__':
    d_phi = 1/8
    x_list = np.linspace(0.2, 0.5, 400)
    xb_list = np.linspace(0.0j, 0.3j, 70)
    
    z_list = np.array(x_list)
    zb_list = np.array(x_list)
    ope_list = [0.25, 0.000244141, 0.015625, 3.43323e-6, 0.000219727, 0.0000152588, 6.81196e-6]
    spec_list = [[0, 1], [0, 4], [2, 2], [2, 6], [4, 4], [4, 5], [6, 6]]
    cft = CFT_2D(d_phi)
    
    print(cft.crossing_eq(z_list, zb_list, ope_list, spec_list))
    