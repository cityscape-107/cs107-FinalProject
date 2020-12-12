## Introduction

Automatic Differentiation, or Algorithmic Differentiation, is a term used to describe a collection of techniques that can be used to calculate the derivatives of complicated functions. Because derivatives play a key role in computational analyses, statistics, and machine and deep learning algorithms, the ability to quickly and efficiently take derivatives is a crucial one. Other methods for taking derivatives, however, including Finite Differentiation and Symbolic Differentiation have drawbacks, including extreme slowness, precision errors, inaccurate in high dimensions, and memory intensivity.  Automatic differentiation addresses many of these concerns by providing an exact, high-speed, and highly-applicable method to calculate derivatives. Its importance is evidenced by the fact that it is used as a backbone for TensorFlow, one of the most widely-used machine learning libraries. In this project, we will be implementing an Automatic Differentiation library that can be used as the basis for analysis methods, including an Optimization and Newton’s Method extension that we will illustrate. 

## Background

The principal concept that is going to be leveraged through automatic differentiation is that we will construct the point derivative of every function based on how a function can be decomposed into elementary operations. Computing the derivatives of these different atoms will subsequently enable us to be able to construct derivatives of a wide range of real-valued functions. Therefore, the derivative of every function can be deduced from simple laws : how to derive basic functions (or atoms) and how to handle the derivates on basic operations of functions. This is explained in the following table. We first present the atoms and then present how to handle the derivative on basic operations on functions. Here, x is a real variable and u and v are functions. 

|Atom function   |   Derivative |
|:--------------:|:-------------:|
|<img src="https://render.githubusercontent.com/render/math?math=x^r">|<img src="https://render.githubusercontent.com/render/math?math=r*x^{r-1}">|
|<img src="https://render.githubusercontent.com/render/math?math=ln(x)">|<img src="https://render.githubusercontent.com/render/math?math=\frac{1}{x}">|
|e^x|e^x|
|cos(x)|-sin(x)|
|sin(x)|cos(x)|
|u+v|u'+v'|
|uv|u'v+uv'|
|<img src="https://render.githubusercontent.com/render/math?math=\frac{u}{v}">|<img src="https://render.githubusercontent.com/render/math?math=\frac{u'v-uv'}{v^2}">|
_Table 1._


Now that we know how to compute the derivatives of atoms and how to handle derivatives on basic operations of functions, we want to visualize how can a function be decomposed into thse basic operations.
An important visualization of how a function can be decomposed into several elementary operations is the computational graph. 

For instance, we are going to draw the graph of the function <img src="https://render.githubusercontent.com/render/math?math=[f(x,y) =exp(-(sin(x)-cos(y))**2)]">


![images/imagesComputational_graph.jpeg](images/Computational_graph.jpeg)

Therefore, the resulting quantity of interest can be explicitely expressed as a composition of several functions. In order to compute the derivative of these successive compositions, we are going to leverage a powerful mathematical tool: the **chain rule**.
A simple version of the chain rule can be expressed as follows : for <img src="https://render.githubusercontent.com/render/math?math=$f$"> and <img src="https://render.githubusercontent.com/render/math?math=$g$"> two functions, 

<img src="https://render.githubusercontent.com/render/math?math=$[f(g)]' = g'*f'(g)$">

Therefore, from the computational graph we have seen above, we can express the derivative of the function encoded at every node by computing the derivative of this elementary operation and multiplyingby the derivative of the inner function. We know that we are able to compute the derivative of the elementary operation from the derivative of the different atom functions. 
Now, the question is to get the derivative of the inner function, that represents all the composition of the different operations encoded at every node until the current node. We do this iteratively, by applying at every node the chain rule with the previous composition operations. 
This suite of operations is encoded on the trace table. 


![images/Evaluation_table.png](images/Evaluation_table.png)

Therefore, from the previous points, we see that we will be able to compute value of the gradient of a function evaluated on a point by iteratively applying the chain rule at every operation node and leveraging a set of basic derivatives and operation on derivatives.


Let's now move on to a brief background of our extension: optimization. Specifically, Gradient Descent. Gradient Descent is the preferred way to optimize neural networks and other Machine Learning algorithms. There are tens of varients of the algorithm - we have demonstrated 3 of these, namely Adam, Stochastic Gradient Descent, and RMS Prop. 

Gradient Descent, in general, is a way to minimize a certain objective function <img src="https://render.githubusercontent.com/render/math?math=$J(\theta)$"> that is parametrized by a model's parameters <img src="https://render.githubusercontent.com/render/math?math=$\theta \in R^d$">. Essentially, we follow the direction of the slope of the objective function surface downwards until we reach a local minimum ( a valley). More specifically, it updates the parameters in the _opposite_ direction of the gradient of the objective function <img src="https://render.githubusercontent.com/render/math?math=$\Delta_\theta J(\theta)$"> with respect to our parameters. We define a _learning rate_ <img src="https://render.githubusercontent.com/render/math?math=$\eta$"> that determines the size of the steps we take in this direction to reach a local minimum. You can see a basic visualization of this process in the graphic below. 

![images/gradientDescentPic.jpeg](images/gradientDescentPic.jpeg)

Additionally, a more 3D visualization of this process can be seen in the graphic below. 

![images/3dgradientDescentPic.jpg](images/3dgradientDescentPic.jpg)


  A version of this algorithm is present in almost every advanced machine learning and deep learning library, and it's strength is a testament to how powerful Automatic Differentiation is. 


## How to use

## Note - Jim will edit this further
- The url to the project is: https://github.com/cityscape-107/cs107-FinalProject


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
>>> import os

>>> os.getcwd()
```

- Then, in your script, import `ADbase2`
```
>>> from ADbase2 import *

```


- Define 1D scalar variable

```
# x=a, and is of 1 dimension
>>> x = AD(a,1)
```

- Define function

```
>>> f = f(x)
```
- Find value and 1D Jacobian (derivative)

```
# value
>>> f.val

# Jacobian
# 1D Jacobian is just derivative, so we use .der here. 
# Use .jac in nD situations
>>> f.der
```

- Demo: ℝ1→ℝ1

Consider the case of <img src="https://render.githubusercontent.com/render/math?math=f(x)=2*x^{3}$"> at <img src="https://render.githubusercontent.com/render/math?math=x=4$">.


```
#define variable
>>> x=AD(4,1)

#define function
>>> f=2*x**3



#value
>>> f.val
128



#Jacobian
>>> f.der
96

```




## Software organization
#### 1. Modules 
Our automatic differentiation package (named `Cityscape-107`) will consist of two modules:
 - A main module (`AD`) for the basic requirements of automatic differentiation. 
 - An additional module(`optimization`) will be an extension of the basic requirements. The optimization method will be used to maximize or minimize a given function making use of its derivative and maybe also making use of the rootfinder method in order to find the values for which the sderivatives become zero. 


The `AD` module has a main class called AD. It is initialized with a two arguments (value and value for the derivative) and they are stored as `self.val` and `self.der` attributes. There are several functions including the overloaded dunder methods/operations `__add__`, `__radd__`,`__mul__`, `__sub__`, `__pow__`, etc. as well as some basic functions `sin`, `cos`, `exp`, etc.


#### 2. Directory Structure 
All the modules will be found in the directory `Cityscape-107`, under subdirectories with the name of the module. There will also be a directory for `tests`, as well as `examples` and `documentation`. Additional documentation will be also provided for each individual module.

The main directory will also include files like the `.travis.yml`, `.codecov.yml`, `setup.py`, `README.md`, `LICENSE.txt` and any other necessary files.

The structure will be similar to the following example:


```python
Cityscape-107/
        cs107-FinalProject/ #directory with the modules' folders and files
                __init__.py  
                AD/ #main module
                        __init__.py
                        ADmulti.py
                        test_ADmulti.py

                optimization/ #extension 
                        __init__.py
                        optimizer.py
                        Test_Optimizer.py

                docs/ #documentation
                        images/ #will contain images used in documentation
                        prev_milestone_docs/ #storage of previous milestone docs 
                        documentation.md

                tests/ #tests
                        __init__.py
                        
                        
                        ...

                examples/
                        ex_driver.py
                        NTRF_driver.py
                        ...

                .travis.yml
                README.md
                .codecov.yml
                requirements.txt
                setup.py
                LICENSE.txt #terms of distribution
                ...
```

#### 3. Distribution
We will distribute our package using `PyPI`. The files  `setup.py`, `setup.cfg`, `LICENSE.txt` and `README.md` that are outside of the `Cityscape-107` package folder are necessary for PyPI to work. 

The file `setup.py` will contain important information like:
 - the `name` and `version` of the package.
 - `download_url` (GitHub url).
 - `install_requires` (list of dependencies).
 
By uploading our package to `PyPI` it will be easy to install just by simply writting:

       $ pip install Cityscape

#### 4. Testing 
We will use the continuous integration tool `Travis-CI` linked to our GitHub project to automatically test changes before integrating them into the project. This will ensure that new changes are merged only if they pass the tests and do not break our code. Our directory `tests` consists of tests for our functionality. The `test_ADmulti.py` file, for example, contains many tests for the AD function.  

Additionally, `Codecov` will provide coverage reports of the tests performed i.e. the percentage of our code that the tests actually tested. After tests are successfully run by `Travis-CI` a report is sent to `Codecov`, which will show the test coverage of the code in our project repository. 

#### 5. Packaging: How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
## jim, if you want to include any more PyPi details here, maybe


As mentioned above, our software is packaged so that it can be downloaded with pip! 








## Implementation 

#### 1. Core Class

In order to implement our Forward Mode, our core class was the AD class. 
The AD class was a representation of a Node in our computational graph. It holds as attributes a value 
and a derivative, which are computed as in the trace table. 

#### 2. Core Data Structure 

For now, our data structure only supports 1D input and 1D output. We leveraged numpy arrays as data structures
for our values and derivatives because of their convenience in term of memory and time efficiency. 
This choice has a counterpart though, we will need to handle the rigidity and the immutable aspect of these data 
structures. Furthermore, in higher dimensions, gradients are arrays and Jacobians are matrices. This 
is why we wanted our code to be adapted to numpy arrays as of now. 

#### 3. Important attributes of the class

The important attributes of the AD class are value and derivative. We decided to define the default value of 
the derivative for a new instance to be 0. Therefore, a user could implement a *constant* via only specifying its value. However, 
this choice required that when defining a *variable*, the user should input a 1 value for the value of the 
derivative. 

For now, the Jacobian and the derivative are the same value so there is no Jacobian attribute or function.

#### 4. External Dependencies

We tried to keep the external dependencies at the lowest possible. There are two reasons for that:
- User Convenience (the user does not need to install 100 packages to run our code)
- Implementation convenience (every external dependency has its own syntax in a way, and we wanted to be consistent 
in our implementation regarding design)

Therefore, the only two external dependencies needed in order to run our code are: math and numpy libraries.

#### 5. Elementary functions

We defined several elementary functions in order to define the way AD variables would interact between each other.
This has been done via overloading the elementary operations: addition, substraction, multiplication, division and power functions. 
For the division operation, we needed to pay extra attention to the __rtruediv__ operation, because of the asymmetry of this operation. 
Last, the power overloading was also delicate because of forbidden cases and the derivation of a function which exponent being another function.   
We also defined the elementary functions: trig functions, inverse trig functions, hyperbolic functions, a logistic function, exp, and log. 



## Our Extension: Applications to Optimization, Gradient Descent, and Machine Learning 
There are several computational applications that implemenent a type of gradient descent that could harness our automatic differentiation tool. In order to apply our AD tool to a neural network, for instance, we would need to have variables that stored the weights for the layers of the network, and our AD tool would be used as a step of the backpropogation methodology. 

As we discussed above, Gradient Descent is an algorithm that involves a set of parameters that will minimize a loss. It's equation looks something like this: <img src="https://render.githubusercontent.com/render/math?math=$\theta_{t+1} = \theta_t - \alpha * \Delta_\theta* J$">

The algorithm that involves updating a set of parameters to minimize a loss, and is typically in the form of 𝜃_𝑡+1=𝜃_𝑡−𝛼∇_𝜃𝐽. The gradient here is the gradient of the loss with respect to the parameters - Automatic differentiation allows us to automate the calculation of this step / these derivatives. Our file would repeatedly make use of the AD() class to calculate the derivatives! The disadvantages of automatic differentiation outweigh the advantages in this situation. 

![images/gradientDescentPic2.jpg](images/gradientDescentPic2.jpg)

With that basic background built up (for more of an intuitive understanding, see the `Background` section of the documentation above), 
let's describe an example of where gradient descent might be used, namely the problem of predicting housing prices, i.e. the ML equivelent of "Hello World". Every machine learning model needs a problem T, a performance measure P, and a dataset E, from where our model can learn patterns. Let us say that our dataset has 781 data records, each of which has 3 features - size (square feet), school district (integer from 1-20) and price. 
In order to measure accuracy, we have to create a performance measure. We can use, for example, the Mean Squared Error (formula shown below - don't worry about the variables, the colored labels should sort out their meaning!)

![images/meanSquaredErrorFormula.jpg](images/meanSquaredErrorFormula.jpg)

Now, our goal is to build a model (a function whose parameters we are trying to find) that can take in a size and predict a housing price.
A simplified version of our end goal function is the following: 

<img src="https://render.githubusercontent.com/render/math?math=$housingPrice = weight1 * squareFootage %2B weight2*schoolDistrict">

To get started, we randomly initialize values of weight1 and weight2. Then, at each iteration, we take our dataset, feed it into our current function, and use our results to calculate our error. With that error, we can calculate the partial derivatives of the error with respect to each weight, and adjust our weights accordingly. 

<img src="https://render.githubusercontent.com/render/math?math=$\Delta Err = [\frac{\partial}{\partial W_0}, \frac{\partial}{\partial W_1}]^T$">

We want our weights to update so that they lower our error in our next iteration, so we need to make them follow the _opposite direction_ of each respective gradient. We are going to change the weights by taking a small step of size <img src="https://render.githubusercontent.com/render/math?math=$\eta$"> in this opposite direction. Putting these steps together, we have our update step formula as: 

<img src="https://render.githubusercontent.com/render/math?math=$W_O = W_0 - \eta * \frac{\partial}{\partial W_0}$ ">


<img src="https://render.githubusercontent.com/render/math?math=$W_1 = W_1 - \eta * \frac{\partial}{\partial W_1}$
"> 

We continue with these iterations until we have an error that is either 0 or below a certain threshold that we set for ourselves. 


Recall the graphic that we used to visualize 3D gradient descent from above. If we imagine <img src="https://render.githubusercontent.com/render/math?math=$\theta_0$"> 
and <img src="https://render.githubusercontent.com/render/math?math=$\theta_1$"> to be our two input features, square footage and school district, this graphic makes it very easy to visualize the process we just covered!

![images/3dgradientDescentPic.jpg](images/3dgradientDescentPic.jpg)

Now that we've built a solid understanding of the process of gradient descent, let us explore our extension and how we built it. Our `Optimizer.py` file contains 4 classes: an `Optimizer()` class, along with three Gradient Descent Algorithms that inherit the `Optimizer()` class: `Adam(Optimizer)`, `stg(Optimizer)` (Stochastic Gradient Descent), and `RMSProp(Optimizer)`. 

Our `Optimizer()` class essentially builds a base gradient descent framework whose parameters (i.e. batch data sets, adaptive learning rate) can then be modified in classes that inherit it for a more customized experience. 
`Optimizer()` consists of 5 methods: 
- `__init__()`, which initializes and returns an object of the class Optimizer, allowing a user to optimize a function based on a descent method of their choice
- `__str__()`, a tostring method
- `produce_random_points()`, a function which allows a user to produce random initialziation points in order to start the process of gradient descent when they don't wish to individually specify them. It infers dimensionality of the inputs and returns points sampled from a gaussian distribution. 
- `annealing()`, a function which essentially allows the user to fine tune their original points so that they start off on the right foot, accelerating the optimization algorithm. At tthe moment, it only supports the quadratic function. 
- `descent()`, a function which actually runs our gradient descent algorithm to minimize a function. 

We then inherit `Optimizer()` into 3 other classes: 
- `Adam(Optimizer)`, which uses the default values we define for the parameters. By default, we have an adaptive learning rate for each weight that changes depending on the accumulated squared gradients until that iteration. Adam also keeps an exponentially decaying average of mast gradients, similar to momentum (a method that essentially helps accelerate gradient descent and reduces the oscillation). 
- `sgd(Optimizer)`, which reverts some of our default parameters that implemented "momentum"  for a standard, classic gradient descent method. This is stochastic gradient descent without any of our special parameters, like momentum, adaptive learning rate, etc. 

- `RMSProp(Optimizer)`, which also (1) removes momentum and (2) adds an adaptive learning rate that changes depending on the accumulated squared gradients until that iteration (like in Adam). 

Each of these classes have `__init__()` methods and `__str__()` methods. 


#### Other Examples: 
We also implemented a driver for Newton's Root Finding Method for vector valued functions of vector variables. You can find this in the `examples` folder of our codebase. 

## ^^^ Maybe add more about newton root finder? 




## Broader Impact:
As we discussed above, automatic differentiation has an incredible and a wide spread array of applications. And while there might not be a way to “misuse” the simple practice of taking derivatives, the applications in which Automatic Differentiation is used, especially Artificial Intelligence and Optimization, are ripe for misuse. Much in the same way that biased datasets lead to biased ML models and biased predictions, if our automatic differentiation library gives incorrect values or values that don’t have a high accuracy, the use of those values in real-life algorithms can lead to issues.  AI and Machine Learning, is, at its core, simply math - derivatives and gradients taken across a dataset to minimize a certain loss. If we incorrectly calculate these values, our algorithm could arrive at results that could have harmful effects on the very people they are meant to help. These algorithms range from weather prediction models to algorithms that determine bail amounts and the length of someone’s prison sentence. This comes back to an idea that our society is grappling with at the moment - while Automatic Differentiation and its uses in optimization (the extension that we implemented) and machine learning algorithms are powerful, we must take careful steps that every major decision the algorithm makes is human-reviewed by a diverse and responsible board with a knowledge of the subject area. Algorithms are only as powerful as the people who design them, and we can ensure that they are used in the best way possible by creating a culture of responsibility and ethical-based review in our codebase and community. 

## Software Inclusivity: 
We developed this project using github version control, a system that is often quite difficult and confusing to understand in the very beginning of stages of trying to use it. People who might not have access to a robust and in-depth computer science education might find it difficult to contribute to this project because they aren’t familiar with branching, committing, pushing, and pulling from github. These groups include several underrepresented minorities, including women, Black, Latinx, and Native Americans. This also includes people in rural and urban areas who might not have access to a CS education that includes concepts like git and version control. As someone who learned about these concepts quite late in her CS education as well, I can testify to the fact that “software development” as a practice can often be quite intimidating and can keep bright and talented people from pursuing the field because of how much prior knowledge of “arcane” (but necessary) things like git. 

Once a developer has made themselves comfortable with the codebase and with the practice of version control, we have a very fair and open system of code contribution. After making a branch, developers can create Pull Requests that are approved by either a few members of the team (if the PR involves a small change, such as a bug fix or comment) or all members of the team (if the PR involves major changes such as a new feature, test, or method). If reviewers have any questions, they leave them in the Comments space of the PR and the developer can respond to them as needed. If our organization was bigger, we would make sure to hire developers who are diverse in age, race, gender, and sexual orientation in order to make everyone feel completely comfortable adding radical features, challenging their peers and having respectful disagreements, and overall contributing to a more robust and dynamic codebase and product. 


## Future
In line with the tech community's growing emphasis on / acknowledgement of the fact that as developers and researchers, we must be acutely aware of the impacts of our work, it would be very cool to create another extension of our AD package that creates a "Effects of Biased Data" visualizer. Although there is a growing interest in post-processing bias mitigation methods, the most common and spoken-about cause of AI and ML model bias is rooted in biased, unrepresentative datasets. Google, IBM, and Microsoft all came under fire for releasing computer vision models (which also use backpropogation and automatic differentiation, another possible future extension) that were unable to recognize african american women. The main reason for this inability? A dataset that didn't have nearly enough women of color as compared to white men and women. 

An "Effects of Biased Data Visualizer" could use an Automatic Differentiation core to visually show the changing accuracies of computer vision models as the datasets on which they are trained become more diverse and representative of the world around us. In the same way that we built an "Optimizer" extension with an AD core, we could build a Computer Vision extension and then create a visualization that shows (perhaps in graph form) how a model's ability to perform on images of underrepresented minorities changes as its dataset becomes more diverse. 

Additionally, thinking beyond extensions rooted in computer science, Automatic Differentiation has a whole host of applications in the natural sciences! We could implement a specific automatic differentiation technique called "interface contraction" that is commonly used in biostatistical analysis. Interface contraction makes derivative computation more effecient by taking advantage of the fact that the number of variables passed between subroutines is often very small compared to the number of variables with respect to which you want to differentiate. It has been famously used in a study analyzing the relationships of age and dietary intake with risk of developing breast cancer in a cohort of 89,538 nurses. Implementing this type of AD for use in biostatistics would be fascinating and an excellent way to show the many applications of Automatic Differentiation. 
















