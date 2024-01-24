import numpy as np
import yaml
from scipy.optimize import brentq
import matplotlib.pyplot as plt

class Generator():
    def __init__(self, struc_file = 'structure.yml', set_file = "sets.yml", cap_file = "cap.yaml" ):
        self.Ns = 1e15
        self.Eo = 8.8541878128e-12 # [F/m]
        self.q = 1.60218e-19 # [C]
        self.Kb = 1.38064852e-23 #m2 kg s-2 K-1 -> [eV/K]
        self.Navo = 6.023e23 # [mol-1]
        self.T = 298 # [K] Room temperature
        self.B = (self.Kb*self.T)/self.q # [V]
        self.conc = 0.001
        self.Io = self.conc/0.001 #mol/l=mol/0.001m^3 - 
        self.Ew = (78.3810 * self.Eo) #F/m -> Electrolyte
        self.Qo = np.sqrt(8*self.Ew*self.B*self.Io*self.Navo*self.q) #C/m^2
        self.c_s = 0.8 # use ctern capacitance for corresponding oxide
        self.n = 0
        self.m = 0
        self.chain = None
        with open(struc_file, 'r') as outfile:
            self.db_dict = yaml.safe_load(outfile)
        with open(set_file, 'r') as outfile:
            self.sets = yaml.safe_load(outfile)
        with open(cap_file, 'r') as outfile:
            self.cap = yaml.safe_load(outfile)
    def get_aas(self):
        return [*self.sets[0], *self.sets[1],*self.sets[2]]
    def get_kvals(self, chain):
        aa = None
        if chain is None:
            raise ValueError('Chain sequence is required')
        if type(chain) == str:
            chain = [*chain]
        if type(chain) != list:
            raise ValueError('Chain must be string or list')
        aa = chain
        self.chain = chain
        k_vals = None
        # ind 1 is for amine attached
        # ind 0 is for carb attached
        
        ind = 0
               # N  M
        # DYK -  2, 2
        # DYKA - 2 ,2
        # DYND - 1, 3
        
        
        self.n = 1-ind #Amines
        self.m = ind #Carboxyl
        connection = (ind)%2
        for i, k in enumerate(aa):
            if k in self.sets[1]:
                self.m+=1
            elif k in self.sets[2]:
                self.n+=1
            if k_vals is None:
                k_vals = self.db_dict[k]
                k_vals = np.delete(k_vals, [connection], axis=0)
            else:
              
                k_vals = np.concatenate((np.delete(k_vals, [len(k_vals)-2], axis=0), np.delete(self.db_dict[k], [connection], axis=0) ))

        k_vals.sort()
        k_vals = k_vals[k_vals != 0]
        # print("N:",self.n, "M:",self.m)
        
        # print(k_vals)
        k_vals = np.power(10,-k_vals)
        k_vals = np.insert(k_vals,0,1)
        return k_vals
    def get_tokens(self, k_vals):
        pot_plot = []
        pot = 0.0
        pot_plot = np.zeros(1400)
        cnt = 0
        def f(x):
            sig_dl = self.Qo*np.sinh(x/(2*self.B))
            pot_o = sig_dl/self.c_s + x
            H_s = H_bulk*np.exp(-pot_o/self.B)
            num = 0
            for i in range(self.n):
                num+=np.power(H_s,self.n+self.m-i)*np.prod(k_vals[0:i+1])
            num*=self.n
            num1 = 0
            for i in range(self.m):
                num1+=np.power(H_s,i)*np.prod(k_vals[0:self.n+self.m+1-i])
            num1*=self.m
            den = 0
            for i in range(self.n+self.m+1):
                den+=np.power(H_s,self.n+self.m-i)*np.prod(k_vals[0:i+1])
            sig_o = self.q*self.Ns*((num-num1)/den)
            return sig_dl-sig_o
            
        def get_cap(zetaPot):
            curr_cap = 0
            curr_len = 0
            for j in range(len(self.chain)):
                mol_per, mol_length = self.cap[self.chain[j]]
                mol_per*=self.Eo
                mol_length*=1e-10
                C = mol_per/mol_length # permittivity/ length from amino xls
                if j == 0:
                    curr_cap = C # second capacitance {lists}
                    curr_len = mol_length; # length of each aa
                else:
                    curr_cap = (curr_cap*C) / (curr_cap + C) # parallel equation for total cap
                    curr_len = curr_len + mol_length

                #Last value of Cap list is the intrinsic capacitance of sequence

            diffuseLayerCap = (self.Qo /( 2*self.B )*np.cosh(zetaPot/(2*self.B)))

            diffLayerCap = (diffuseLayerCap*self.c_s)/(diffuseLayerCap+self.c_s)

            capBio2 = curr_cap*self.Ns*1e-16
            # return curr_cap*curr_len
            return diffLayerCap + capBio2
        
        cap_plot = []
        for pH in np.linspace(0,14,1400):
            H_bulk = 10**(-pH)
    
            # Minimum and maximum of the entire curve
            
            pot = brentq(f,-0.28,0.28) # this is zeta pot for capacitance calc
            cap_plot.append(get_cap(pot))
            sig_dl = self.Qo*np.sinh(pot/(2*self.B))
            pot_o = sig_dl/self.c_s + pot # surface potential for capacitance
            pot_plot[cnt] = pot_o
            cnt+=1

        
        
        out = np.gradient(np.gradient(pot_plot))
        tokens_pot = np.where(np.logical_and(np.abs(np.diff(out)) >= 1e-10, np.diff(np.sign(out)) != 0))[0]
        # for i in range(0,len(tokens_pot)):
        #     plt.scatter(tokens_pot[i], out[i], s = 8, c='b')
        tokens_pot = np.append(tokens_pot, 0)

        out = np.gradient(np.gradient(cap_plot))
        tokens_cap = np.where(np.logical_and(np.abs(np.diff(out)) >= 1e-10, np.diff(np.sign(out)) != 0))[0]
        tokens_cap = np.append(tokens_cap, 0)

        #plt.show()
        return tokens_pot, tokens_cap
    
    def get(self, chain = None):
        k_vals = self.get_kvals(chain[:-1])
        # print(k_vals)
        return self.get_tokens(k_vals)
    
    