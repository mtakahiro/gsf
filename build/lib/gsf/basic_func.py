import numpy as np

class Basic(object):
    '''
    '''
    def __init__(self, MB):
        self.ZZ = MB.Zall
        self.NZ = len(self.ZZ)
        self.delZ = MB.delZ

    def Z2NZ(self, Z):
        '''
        Critical function to infer NZ from Z.
        '''
        NZ = np.argmin(np.abs(self.ZZ-Z))
        return NZ


class Basic_tau(object):
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
        
    def Z2NZ(self, Z, tau, age):
        '''
        Critical function to infer NZ from Z.
        '''
        NZ = np.argmin(np.abs(Z-self.ZZ))
        NT = np.argmin(np.abs(tau-self.TT))    
        NA = np.argmin(np.abs(age-self.LA))

        return NZ, NT, NA
