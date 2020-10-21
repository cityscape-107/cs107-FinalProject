

#Milestone 1

# Introduction

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

For instance, we are going to draw the graph of the function $f(x) = $ 

"insert the computational graph"

Therefore, the resulting quantity of interest can be explicitely expressed as a composition of several functions. In order to compute the derivative of these successive compositions, we are going to leverage a powerful mathematical tool: the **chain rule**.
A simple version of the chain rule can be expressed as follows : for $f$ and $g$ two functions, 

$[f(g)]' = g'*f'(g)$*

Therefore, from the computational graph we have seen above, we can express the derivative of the function encoded at every node by computing the derivative of this elementary operation and multiplyingby the derivative of the inner function. We know that we are able to compute the derivative of the elementary operation from the derivative of the different atom functions. 
Now, the question is to get the derivative of the inner function, that represents all the composition of the different operations encoded at every node until the current node. We do this iteratively, by applying at every node the chain rule with the previous composition operations. 
This suite of operations is encoded on the trace table. 


"insert the forward trace table related to the computational graph"




Therefore, from the previous points, we see that we will be able to compute value of the gradient of a function evaluated on a point by iteratively applying the chain rule at every operation node and leveraging a set of basic derivatives and operation on derivatives.

## How to use


## Software organization

# Implementation

From the background part, there are several questions that need to be dealt with during implementation:

1. How can we encode the structure of the computational graph, that allows us to construct the trace table, which is an illustration of how we will perform the operations?
2. How can we represent an elemental operation between functions ? 
3. What will eb the data structure in order to represent the value of a function and its derivative ? 
4. How to represent the derivatives of elementary functions ? 



1. We could use a tree structure in order to encore the structure of the computational graph. Every node would encode an elementary operation, that we would further encode into a class of elementary operations (structure to be defined further). The node will be the final function and the leaves would be when we assigned values. The children of a node would be the several components composing a elementary operation. The node would contain several attributes, their children (links) and the elementary operation.
2. Now, in order to represent the elementary operation that would be encapsulated inside every node of the computational tree, we would define a class of elementary functions. This class would override the usual dunder methods in order to specify how we want to handle the addition, multiplication...
3. To be done, how to deal with elementary operations how to define addition ..
4. Leverage he datastructures defined in 3, and define a class of elementary functions where we define explicitely the derivatives.


External dependencies: numpy, math
Elementary functions: use numpy
Ability to act on an array

 


