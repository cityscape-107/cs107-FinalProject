### Handling N-Dimensions
In this release, our package handles 1D-in-1D-out situations (scalar) only. In the next update, we will add R^n^-in-R^n^-out (matrix) compatibility. 
```
# define variables
>>> x=AD(1,1)
>>> y=AD(2,1)
>>> z=AD(3,1)

#define functions
>>> f = f(x)
>>> g = g(x)
>>> z = AD.combine(f,g)

#Jacobian
>>> z.jac
```

### Visualizing Forward and Reverse Mode
We aim to provide a visualization tool that is comparable to "Auto-eD." Our GUI will be able to take function inputs directly, and users no longer need to do tedious input by button clicking.