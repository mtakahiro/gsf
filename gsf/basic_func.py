import numpy as np
# Function which is also used in function_class.py
class Basic:
    def __init__(self, MB):
        self.ZZ = MB.Zall
        self.NZ = len(self.ZZ)
        try:
            self.delZ = ZZ[1] - ZZ[0]
        except:
            self.delZ = 0.0001
    # Conversion from Z to NZ
    def Z2NZ(self, Z):
        NZ = np.argmin(np.abs(self.ZZ-Z))
        return NZ
