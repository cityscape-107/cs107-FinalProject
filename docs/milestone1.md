

#Milestone 1

# Introduction

Automatic Differentiation, or Algorithmic Differentiation, is a term used to describe a collection of techniques that can be used to calculate the derivatives of complicated functions. Because derivatives play a key role in computational analyses, statistics, and machine and deep learning algorithms, the ability to quickly and efficiently take derivatives is a crucial one. Other methods for taking derivatives, however, including Finite Differentiation and Symbolic Differentiation have drawbacks, including extreme slowness, precision errors, inaccurate in high dimensions, and memory intensivity.  Automatic differentiation addresses many of these concerns by providing an exact, high-speed, and highly-applicable method to calculate derivatives. Its importance is evidenced by the fact that it is used as a backbone for TensorFlow, one of the most widely-used machine learning libraries. In this project, we will be implementing an Automatic Differentiation library that can be used as the basis for analysis methods, including a Newton’s Method extension that we will illustrate. 


# Background


The principal concept that is going to be leveraged through automatic differentiation is that we will construct the point derivative of every function based on how a function can be decomposed into elementary operations. Computing the derivatives of these different atoms will subsequently enable us to be able to construct derivatives of a wide range of real-valued functions. Therefore, the derivative of every function can be deduced from simple laws : how to derive basic functions (or atoms) and how to handle the derivates on basic operations of functions. THis is explained in the following table. We first present the atoms and then present how to handle the derivative on basic operations on functions. Here, x is a real variable and u and v are functions. 

|Atom function   |   Derivative |
|:--------------:|:-------------:|
|$x^r$           | r*x^{r-1}    |
|$ln(x)$|$\frac{1}{x}$|
|e^x|e^x|
|cos(x)|-sin(x)|
|sin(x)|cos(x)|
|u+v|u'+v'|
|uv|u'v+uv'|
|\frac{u}{v}|\frac{u'v-uv'}{v^2}|
_Table 1._


Now that we know how to compute the derivatives of atoms and how to handle derivatives on basic operations of functions, we want to visualize how can a function be decomposed into thse basic operations.
An important visualization of how a function can be decomposed into several elementary operations is the computational graph. 

For instance, we are going to draw the graph of the function $f(x,y) =exp(-(sin(x)-cos(y))**2) $ 

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

- Install Cityscape® AD tool using pip.

```
#Installation 
conda install -n CityscapeAD

```


#### Demos

```
import CityscapeAD as  ad
import numpy as np
```


- define function

```
f=ad.function(f(x))

```

- Value input

```
f.val(1)

```


- Demo 1: ℝ1→ℝ1

Consider the case of f(x)= x, we calculate the value and the first derivative at x=1.

```
#define function
f=ad.function( x )

#value input
f.val(1)

#derivative
f.get_der()
>>> 1

#value
f.get_value()
>>> 1

```

- Demo 2: ℝ𝑚→ℝ1

Consider the case of f(x)= y-x, we calculate the value and the first derivative at x=1, y=1.

```
#define function
f=ad.function( y - x )

#value input
f.val( 1,1 )

#derivative
f.get_der()
>>> -1
>>> 1

#value
f.get_value()
>>> 0


```

- Demo 3: ℝ1→ℝ𝑛

Consider the case of f(x)= (x^2, x+1), we calculate the value and the first derivative at x=1.

```
#define function
f=ad.function( [x^2] , [x+1] )

#value input
f.val(1)

#derivative
f.get_der()
>>> 2x , 1

#value
f.get_value()
>>> 1 , 2

```

- Demo 4: ℝ𝑚→ℝ𝑛

Consider the case of f(x,y,z)= (x+1 , z+y ), we calculate the value and the first derivative at x=y=z=1.


```
#define function
f=ad.function( [x+1] , [z+y] )

#value input
f.val( 1,1,1 )

#derivative
f.get_der()
>>> 1
>>> 1
>>> 1

#value
f.get_value()
>>> 2 , 2

```








# Implementation


From the background part, there are several questions that need to be dealt with during implementation:

1. How can we encode the structure of the computational graph, that allows us to construct the trace table, which is an illustration of how we will perform the operations?
2. How can we represent an elementary operation between functions ?
3. What will be the data structure used to represent the value of a function and its derivative ?
4. How to represent the derivatives of elementary operations ?


Our library will contain the following classes, which will be the primary data structures used. Classes will contain various methods and have attributes like "kind of function".

class `function()`:

- The attributes of this function class: a string representing the analytical expression, an input with the point at sake, a value which is the output of the function at this point and the value of the gradient at this point
- Wrapper class that leverages all of the classes defined below. 
- Takes as an input the vector function 
- Has a method that allows us to introduce the input points (array or scalar).
- Calls upon the classes discussed below in order to compute the evaluation of the derivatives
- Output a tuple containing the value of the function and the derivative at the input points


class `comp_graph()`:

- Attributes of this class:  a function for which we wish to calculate the graph and a link to the root of the computational tree, which contains the final expression.
- Arguments of the constructor: a string representing the final function. 
- Data structure used for this class: a binary tree. Every node of the tree would encapsulate an operation instance and the children of this node would be the left hand and right hand sides of this operation. The leaves of this tree would be the atomic 'assignment' operations. 
- Methods: a constructor, which would encode the recursive construction of a binary tree. 
- Outputs a series of groups of consisting of at least 2 elements - one node and one elementary operation, with the optional third element for elementary operation requiring 2 input nodes as indicated for the operations() class # i do not understand this section



class `operations()`:

- Defines elementary operations, including the number of arguments required and the action of an operation on the current value of the function AND its derivative 
- Attributes of this class: left and right hand sides of the operations, from the function class and a string operation (might be better to use something else than a string).
- Operations such as +,-,*,/ will be defined as methods within the class. This will be done by overwriting the dunder methods already existing. For these operations, we will also define the methods that implement their range of actions on the value and the derivative. FOr instance, the action on the derivative of __mul__(f1, f2) will be something like f1.der*f2.val + f1.val*f2.der. 

class `elementary_functions()`:

- The class will define values and derivatives for atomic inputs, as defined in _Table 1_.
- Elementary functions such as sin, cos, exp, etc. will be imported from numpy
- The attributes of this class will be : expression (which will contain the reference to the kind of elementary function it is), value and derivative



Handling of invalid inputs: as we will be defining a class for elementary operations that will override the usual dunder methods we will also need to ensure that the inputs into these are valid. Notably, since we are only working with real numbers, we will not define these operations for imaginary inputs and will need to implement checks to ensure only real values are passed through.


The library will have the following external dependencies: numpy, math

Elementary functions such as cos, sin, exp etc. will be implemented using the numpy and math dependencies.

The goal for the library is also to be able to implement the AD differention on an input array, similar to how one can apply the np.exp() function to both a single value and an array.

We will also include an application using the AD library to implement Newton's Root Finding Method for vector valued functions of vector variables. This will be held in a seperate library.
 



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

For ease of user implementation and due to the extensive documentation available on the software, we will use the software framework PyScaffold. Using a software framework will allow for a more coherent organization of code and more straightforward user interaction with the library. Additionally, PyScaffold is a very commonly implemented framework that has been shown to be consistently effective and could be considered like a standar for Python.





# Milestone 1 Feedback


2/2 Introduction
  - Looks good!

2/2 Background
  - Looks good!
  - I think there's a typo in your f(x,y) example, if not then you will need to fix your computational graph.

3/3 How to use
  - Add the expected output from f.get_der() and f.get_value() in your examples.

2.5/3 Software Organization
  - What is the proposed basic functionality of the root finder and optimization modules?
  - What are some example inputs/outputs/functions expected in your modules?
  - Where will your test suite live? Even though its shown in your directory structure, explicitly write it out.

5/5 Implementation
  - Looks good, but your additional considerations section is a bit hard to read.
       - You can consider using more subcategories (e.g. each data structure along with its application and corresponding classes).

Note. https://latex.codecogs.com/ is a helpful rendering service for showing equations on markdown.
