import numpy as np
# Function which is also used in function_class.py
class Basic:
    def __init__(self, ZZ):
        self.ZZ = ZZ
        self.NZ = len(ZZ)
        try:
            self.delZ = ZZ[1] - ZZ[0]
        except:
            self.delZ = 0.0001
    # Conversion from Z to NZ
    def Z2NZ(self, Z):
        if self.NZ>1:
            Zmax = np.max(self.ZZ)
            Zmin = np.min(self.ZZ)
            NZ   = int((Z - Zmin + self.delZ/2.0) / self.delZ)
        else:
            NZ = 0
        return NZ
