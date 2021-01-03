import numpy as np

class Basic:
    '''
    # Function used in function_class.py
    '''
    def __init__(self, MB):
        self.ZZ = MB.Zall
        self.NZ = len(self.ZZ)
        try:
            self.delZ = ZZ[1] - ZZ[0]
        except:
            self.delZ = 0.0001
    def Z2NZ(self, Z):
        '''
        # Conversion from Z to NZ
        '''
        NZ = np.argmin(np.abs(self.ZZ-Z))
        return NZ
