

#Milestone 1

# Introduction

Automatic Differentiation, or Algorithmic Differentiation, is a term used to describe a collection of techniques that can be used to calculate the derivatives of complicated functions. Because derivatives play a key role in computational analyses, statistics, and machine and deep learning algorithms, the ability to quickly and efficiently take derivatives is a crucial one. Other methods for taking derivatives, however, including Finite Differentiation and Symbolic Differentiation have drawbacks, including extreme slowness, precision errors, inaccurate in high dimensions, and memory intensivity.  Automatic differentiation addresses many of these concerns by providing an exact, high-speed, and highly-applicable method to calculate derivatives. Its importance is evidenced by the fact that it is used as a backbone for TensorFlow, one of the most widely-used machine learning libraries. In this project, we will be implementing an Automatic Differentiation library that can be used as the basis for analysis methods, including a Newton’s Method extension that we will illustrate. 


# Background


The principal concept that is going to be leveraged through automatic differentiation is that we will construct the point derivative of every function based on how a function can be decomposed into elementary operations. Computing the derivatives of these different atoms will subsequently enable us to be able to construct derivatives of a wide range of real-valued functions. Therefore, the derivative of every function can be deduced from simple laws : how to derive basic functions (or atoms) and how to handle the derivates on basic operations of functions. THis is explained in the following table. We first present the atoms and then present how to handle the derivative on basic operations on functions. Here, x is a real variable and u and v are functions. 

|Atom function   |   Derivative |
|$x^r$           | r*x^{r-1}    *|
|$ln(x)$|$\frac{1}{x}$|
|e^x|e^x|
|cos(x)|-sin(x)|
|sin(x)|cos(x)|
|u+v|u'+v'|
|uv|u'v+uv'|
|\frac{u}{v}|\frac{u'v-uv'}{v^2}|


Now that we know how to compute the derivatives of atoms and how to handle derivatives on basic operations of functions, we want to visualize how can a function be decomposed into thse basic operations.
An important visualization of how a function can be decomposed into several elementary operations is the computational graph. 

For instance, we are going to draw the graph of the function $f(x,y) =exp(-(sin(x)-cos(x))**2) $ 

![Computational_graph.jpeg](Computational_graph.jpeg)

Therefore, the resulting quantity of interest can be explicitely expressed as a composition of several functions. In order to compute the derivative of these successive compositions, we are going to leverage a powerful mathematical tool: the **chain rule**.
A simple version of the chain rule can be expressed as follows : for $f$ and $g$ two functions, 

$[f(g)]' = g'*f'(g)$*

Therefore, from the computational graph we have seen above, we can express the derivative of the function encoded at every node by computing the derivative of this elementary operation and multiplyingby the derivative of the inner function. We know that we are able to compute the derivative of the elementary operation from the derivative of the different atom functions. 
Now, the question is to get the derivative of the inner function, that represents all the composition of the different operations encoded at every node until the current node. We do this iteratively, by applying at every node the chain rule with the previous composition operations. 
This suite of operations is encoded on the trace table. 


![Evaluation_table.png](Evaluation_table.png)

Therefore, from the previous points, we see that we will be able to compute value of the gradient of a function evaluated on a point by iteratively applying the chain rule at every operation node and leveraging a set of basic derivatives and operation on derivatives.


 



# How to use

### The url to the project is: https://pseudo.link



- Create and activate a virtual environment.

```
#test if you have Anaconda installed, our package relies on it
#in command line
conda -V
conda create -n Cityscape
activate Cityscape
```

- Install and test Cityscape® AD tool using pip.

```
#Installation 
conda install -n CityscapeAD

#test
pytest tests
```


#### Demos

```
import CityscapeAD as  ad
import numpy as np
```


- Variables input

```
x=ad.function.val(1)

```

- define function

```
f=ad.function(f(x))

```

- Demo 1: ℝ1→ℝ1

Consider the case of f(x)= x, we calculate the value and the first derivative at x=1.

```
#value input
x=ad.function.val(1)

#define function
f=ad.function( x )

#derivative
f.get_der()

#value
f.get_value()

```

- Demo 2: ℝ𝑚→ℝ1

Consider the case of f(x)= y-x, we calculate the value and the first derivative at x=1, y=1.

```
#value input
x=ad.function.val(1)
y=ad.function.val(1)

#define function
f=ad.function( y - x )

#derivative
f.get_der()

#value
f.get_value()

```

- Demo 3: ℝ1→ℝ𝑛

Consider the case of f(x)= (x^2, x+1), we calculate the value and the first derivative at x=1.

```
#value input
x=ad.function.val(1)

#define function
f=ad.function( [x^2] , [x+1] )

#derivative
f.get_der()

#value
f.get_value()

```

- Demo 4: ℝ𝑚→ℝ𝑛

Consider the case of f(x,y,z)= (x+1 , z+y ), we calculate the value and the first derivative at x=y=z=1.


```
#value input
x=ad.function.val(1)
y=ad.function.val(1)
z=ad.function.val(1)

#define function
f=ad.function( [x+1] , [z+y] )

#derivative
f.get_der()

#value
f.get_value()

```










# Implementation

From the background part, there are several questions that need to be dealt with during implementation:

1. How can we encode the structure of the computational graph, that allows us to construct the trace table, which is an illustration of how we will perform the operations?
2. How can we represent an elementary operation between functions ? 
3. What will be the data structure in order to represent the value of a function and its derivative ? 
4. How to represent the derivatives of elementary functions ? 



1. We could use a tree structure in order to encode the structure of the computational graph. Every node would encode an elementary operation. Howwe would further encode these elementary operations is another question (structure to be defined further). The tree root will be the final function and the leaves would the atom operations, when we assign values. The children of a node would be the several components composing a elementary operation. 
Therefore, in order to construct the entire function, we would be able to leverage the recursive construction of a tree: in order to construct the operation at a specfic node, we would need to recursively get the operations made on every child and unite them with the operation encoded on the node. 
The node would contain several attributes, their children (links) and the elementary operation (structure to be defined later on).
This class would be called computational_graph(). An instance of this class will have two attributes: a string (the initial function to Differentiate) and a link to the root of the computational tree. 
We will define methods in order to build a Computational tree from a string, that would be a recursive function (explicitely or not, using a stack). 
2. When will we be in the need of an elementary operation ? What utility do we need from it ? We will use elementary operations (+, -, x, /) when we will be given two functions, alogside their derivatives. According to the trace table, we will need from these elementary operations their action on values **and** on derivatives. Therefore, we would implement a class of elementary operations that would be called operations(), that would contain as attributes:
- the left part of the operation
- the elementary operation at sake 
- the right hand side of the operation. 
The methods we will implement are resulting_value, which will compute the value of the resulting operation and resulting_derivatives which will compute the value of the resulting derivatives (leveraging the class of elementayr functions and how we compute their derivatives). Another idea might be to override the existing __add__ and __radd__ in order to have a more comprehensive coding practice. 
3. At every step of the computational graph, the value of a function is defined by the actual value of the function until then, and the value of its gradient. Therefore, we could use a tuple in order to encode (value, gradient). Every element of a tuple might be a list, or an array, in order to account for high-dimension settings. We might also want to define two attributes for an instance of function() class, like f.values and f.derivatives.
This class function() will contain a constructor and we will let the class operations() handle all the operations. Furthermore, using a tuple enables us to keep a immutable sequence and making sure to not perform unwanted changes to our objects.
4. We will define a class of elementary functions. Several things need to be dealt with here. We will leverage these elementary functions when having dealt with the entire initial function, computing the computational graph, and when we need to evaluate an atomic expression and its derivative (as part of the flow described when presenting the operations: we need to work on both sides of the operations, and then work on combining what we got). Therefore, we will need these elementary operations in order to evaluate the value of a function and the value of a function's derivative. 
Therefore, we need to construct a wide basis of elementary functions, having identifiers for each of them and define the derivatives of these functions. An instance of this class elementary() will have several attributes:
- an unique identifier in order to define which elementary function it is
- an instance of the class function (defined thanks to implemented functions that allow to compute the value and the derivative of an elementary function)


We will use a constructor so set values representing the first row of the computational table discussed above (i.e. the first layer of the computational tree).


Handling of invalid inputs: as we will be defining a class for elementary operations that will override the usual dunder methods we will also need to ensure that the inputs into these are valid. Notably, since we are only working with real numbers, we will not define these operations for imaginary inputs and will need to implement checks to ensure only real values are passed through.


The library will have the following external dependencies: numpy, math
Elementary functions such as cos, sin, exp etc. will be implemented using the the numpy and math dependences.
The goal for the library is also to be able to implement the AD differention on an input array, similar to how one can apply the np.exp() function to both a single value and an array.

Also included, will be an application using the AD library to implement Newton's Root Finding Method for vector valued functions of vector variables using the AD library. This will be held in a seperate library to the AD library.
 



## Software organization
#### 1. Modules 
Our automatic differentiation package (named `Cityscape`) will consist of probably three modules:
 - A main module (`autodiff`) for the basic requirements of automatic differentiation. 
 - Probably two additional modules (`rootfinder` and `optimization`) as extensions of the basic requirements. The names here are tentative since we have not made a final decision on which will be the extensions to our project.
 
#### 2. Directory Structure 
All the modules will be found in the directory `Cityscape`, under subdirectories with the name of the module. There will also be a directory for `tests`, as well as `examples` and `documentation`. Additional documentation will be also provided for each individual module.

The main directory will also include files like the `.travis.yml`, `.codecov.yml`, `setup.py`, `README.md`, `LICENSE.txt` and any other necessary files.

The structure will be similar to the following example:


```python
Cityscape/
        __init__.py  
        Cityscape/ #directory with the modules
                __init__.py
                autodiff/ #main module
                        __init__.py
                        autodiff.py
                        
                        documentation/ #docs of main module (.txt, .md, .tex, ...)
                                doc_autodiff.txt 
                
                rootfinder/ #extension 1
                        __init__.py
                        rootfinder.py
                        
                        documentation/
                                doc_rootfinder.txt
                
                optimization/ #extension 2
                        __init__.py
                        optimization.py
                        
                        documentation/
                                doc_optimization.txt
                ...
        
        tests/ #tests
                __init__.py
                tests_autodiff.py
                tests_rootfinder.py
                tests_optimization.py
                ...
        
        examples/
                example1.py
                example2.py
                example3.py
                ...
                
        documentation/ #general documentation
                doc_general.txt
                ...
        #We could also place the documentation in the main directory     
            
        .travis.yml
        .codecov.yml
        setup.py
        setup.cfg
        README.md # description or the module / goal of the project
        LICENSE.txt #terms of distribution
        ...
```

#### 3. Distribution
We will distribute our package using `PyPI`. The files  `setup.py`, `setup.cfg`, `LICENSE.txt` and `README.md` that are outside of the `Cityscape` package folder are necessary for PyPI to work. 

The file `setup.py` contains important information like:
 - the `name` and `version` of the package.
 - `download_url` (GitHub url).
 - `install_requires` (list of dependencies).
 
By uploading our package to `PyPI` it will be easy to install just by simply writting:

       $ pip install Cityscape

#### 4. Testing 
We will use the continuous integration tool `Travis-CI` linked to our GitHub project to automatically test changes before integrating them into the project. This will ensure that new changes are merged only if they pass the tests and do not break our code. 

Additionally, `Codecov` will provide coverage reports of the tests performed i.e. the percentage of our code that the tests actually tested. After tests are successfully run by `Travis-CI` a report is sent to `Codecov`, which will show the test coverage of the code in our project repository. 

#### 5. Packaging: How will you package your software? Will you use a framework? If so, which one and why? If not, why not?

NEED HELP :(
