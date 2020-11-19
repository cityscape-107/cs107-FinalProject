
# How to use

### The url to the project is: https://github.com/cityscape-107/cs107-FinalProject



- Install Anaconda, then create and activate a virtual environment.

```
#in command line
conda -V
conda create -n Cityscape
activate Cityscape
```

- Install dependencies

```
#in command line
conda install numpy
conda install math

```

- Download Cityscape AD from https://github.com/cityscape-107/cs107-FinalProject, alternatively, you can clone the repository

- Unzip downloaded file, copy `ADbase2.py` into your current working directory

- If you are unsure of current working directory, run:
```
import os

os.getcwd()
```

- Then, in your script, import `ADbase2`
```
from ADbase2 import *

```


- Define 1D scalar variable

```
# x=a, and is of 1 dimension
x = AD(a,1)
```

- Define function

```
f = f(x)
```
- Find value and 1D Jacobian (derivative)

```
# value
f.val

# Jacobian
f.der
```

- Demo: ℝ1→ℝ1

Consider the case of $f(x)=2 x^{3}$ at $x=4$

```
#define variable
x=AD(4,1)

#define function
f=2*x**3



#value
f.val
>>> 128



#Jacobian
f.der
>>> 96

```





