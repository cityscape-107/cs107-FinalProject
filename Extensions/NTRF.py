import numpy as np
from ADmulti import AD




def NTRF(init_vals, tol=1e-2, iters=200, resetter=1):
    #initiate variable list
    x0=[ value for key,value in init_vals.items() ]
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

init_vals={'x1':1,'x2':2,'x3':3}
# for k,v in init_vals.items():
#     globals()[k]=AD(v,1,k)

def _function(init_vals):
    for k,v in init_vals.items():
        globals()[k]=AD(v,1,k)
        

    f = x1*2 + x2 + x3/2
    g = x1 + x2 + x3
    h = x1*0 + x2*0 + x3
    v = AD([f,g,h])
    
    return v

#print(NTRF(init_vals))