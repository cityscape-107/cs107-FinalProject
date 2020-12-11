import numpy as np
from ADmulti import AD


def NTRF(init_vals, tol=1e-10, iters=2000, path_freq=1):

    xn=[ value for key,value in init_vals.items() ]
    vals=init_vals

    global path
    path =[]
    i=0   
    while i < iters:        
        
        # WARNING
        # TODO: MAKE SURE AD object.der and AD object.val outputs ordered results
        xn1=xn-np.dot(np.linalg.pinv(_function(vals).der),_function(vals).val)
        xn=xn1
              
        for q in range(len(vals)):            
            vals[list(vals)[q]]=xn[q]
            
        cache=np.array(_function(vals).val)
        
        # users can choose to record path
        # by default, it records path every 50 iterations
        if i % path_freq == 0:
            path.append(cache)


        if all(_ <= tol for _ in cache):            
            result=('Root Found:{}'.format(xn))
            break
        else:
            i+=1
        if i == iters:
            result=('Max Iteration Reached, No Root Found')
        
        
    return result

def _function(init_vals):
    for k,v in init_vals.items():
        globals()[k]=AD(v,1,k)

# users set their function system here
    f = x1*2 + x2
    g = x1 - x2*2 + 2
    v = AD([f,g])
    
    return v


# users set their wanted starting points here
init_vals={'x1':1,'x2':2}

# users get their result
print(NTRF(init_vals))
print('Value Path: \n',path)