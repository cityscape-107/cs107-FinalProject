import numpy as np
from ADmulti import AD

class NT:
    def __init__(self, f, order, tol=1e-6, iters=200):
    self.f = f
    self.order = order
    self.tol = tol
    self.iters = iters
    
    def NTRF(init_vals,f, tol=1e-2, iters=2200, resetter=1):
        #initiate variable list
        xn=[ value for key,value in init_vals.items() ]
        vals=init_vals
        v=f
        #print(xn)
        #print(vals)
        
        i=0
        while i < iters:        
            
        #   xn+1=xn-Jac^-1*f(xn)
            xn1=xn-np.dot(np.linalg.pinv(_function(vals,v).der),_function(vals,v).val)
            xn=xn1
    
            #print(xn)
                    
            for q in range(len(vals)):            
                vals[list(vals)[q]]=xn[q]
                
            cache=np.array(_function(vals).val)
    
    
            if all(_ <= tol for _ in cache):            
                print('Root Found:{}'.format(xn))
            else:
                i+=1
            if i == iters:
                print('Max Iteration Reached, No Root Found')
            
        return 
    
    
    
    def _function(vals,*x):
        for k,v in vals.items():
            globals()[k]=AD(v,1,k)
        
        u= lambda *x: AD([*x])
        
        return u

# f= lambda x1, x2, x3: x1*2 + x2 + x3/2
# g= lambda x1, x2, x3: x1*2 + x2 + x3/2
# h= lambda x1, x2, x3: x1*2 + x2 + x3/2
# v= lambda f, g, h: AD([f,g,h])


# init_vals={'x1':1,'x2':2,'x3':3}

# NTRF(init_vals,[f,g,h])