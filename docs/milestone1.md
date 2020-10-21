#Milestone 1

##Introduction


##Background


##How to use


##Software organization

##Implementation
 

We will use a constructor so set values representing the first row of the computational table discussed above (i.e. the first layer of the computational tree).


Handling of invalid inputs: as we will be defining a class for elementary operations that will override the usual dunder methods we will also need to ensure that the inputs into these are valid. Notably, since we are only working with real numbers, we will not define these operations for imaginary inputs and will need to implement checks to ensure only real values are passed through.


The library will have the following external dependencies: numpy, math
Elementary functions such as cos, sin, exp etc. will be implemented using the the numpy and math dependences.
The goal for the library is also to be able to implement the AD differention on an input array, similar to how one can apply the np.exp() function to both a single value and an array.

Also included, will be an application using the AD library to implement Newton's Root Finding Method for vector valued functions of vector variables using the AD library. This will be held in a seperate library to the AD library.


 
