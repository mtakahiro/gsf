import numpy as np

class Basic:
    '''
    # Function used in function_class.py
    '''
    def __init__(self, MB):
        self.ZZ = MB.Zall
        self.NZ = len(self.ZZ)
        self.delZ = MB.delZ #ZZ[1] - ZZ[0]
    def Z2NZ(self, Z):
        '''
        # Conversion from Z to NZ
        '''
        NZ = np.argmin(np.abs(self.ZZ-Z))
        return NZ

class Basic_tau:
    '''
    '''
    def __init__(self, MB):
        self.ZZ = MB.Zall
        self.NZ = len(self.ZZ)
        self.delZ = MB.delZ
        self.TT = MB.tau
        self.delT = MB.deltau            
        self.LA = MB.ageparam
        self.delLA = MB.delage
        
    # Conversion from Z to NZ
    def Z2NZ(self, Z, tau, age):

        NZ = np.argmin(np.abs(Z-self.ZZ))
        NT = np.argmin(np.abs(tau-self.TT))    
        NA = np.argmin(np.abs(age-self.LA))

        return NZ, NT, NA
