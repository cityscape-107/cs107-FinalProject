import numpy as np
from AD.ADmulti import AD



class NTRF:

    def __init__(self, f, order, tol=1e-6, iters=200):
        self.f = f
        self.order = order
        self.tol = tol
        self.iters = iters


    def find_roots(self, initial_point):
        i = 0
        x0 = initial_point
        while i < self.iters:
            #   xn+1=xn-Jac^-1*f(xn)
            print('Current variable', x0)
            print('Update', (np.linalg.pinv(x0.der)*self.f(x0))[0][0])
            x0.sort(self.order)
            x1 = x0 - (np.linalg.pinv(x0.der)*self.f(x0))[0][0]
            x0 = x1
            # print(x0)
            value = self.f(*x0.val)
            if np.sum(value**2) < self.tol**2:
                break
            i += 1
        return x0


"""
def NTRF(init_vals, tol=1e-2, iters=200, resetter=1):
    #initiate variable list
    x0=[value for key,value in init_vals.items()]
    vals=init_vals
    #print(x0)
    #print(vals)
    
    i=0
    while i < iters:        
        
    #   xn+1=xn-Jac^-1*f(xn)
        x1=x0-np.dot(np.linalg.pinv(_function(vals).der),_function(vals).val)
        #print(x1)
        
        x0=x1
        #print(x0)

                
        for q in range(len(vals)):            
            vals[list(vals)[q]]=x0[q]
            
        cache=np.array(_function(vals).val)

        check=np.sum(cache**2)

        if check < tol:
            break
        i+=1
        
        
    return x0
"""


