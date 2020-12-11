## Introduction

Automatic Differentiation, or Algorithmic Differentiation, is a term used to describe a collection of techniques that can be used to calculate the derivatives of complicated functions. Because derivatives play a key role in computational analyses, statistics, and machine and deep learning algorithms, the ability to quickly and efficiently take derivatives is a crucial one. Other methods for taking derivatives, however, including Finite Differentiation and Symbolic Differentiation have drawbacks, including extreme slowness, precision errors, inaccurate in high dimensions, and memory intensivity.  Automatic differentiation addresses many of these concerns by providing an exact, high-speed, and highly-applicable method to calculate derivatives. Its importance is evidenced by the fact that it is used as a backbone for TensorFlow, one of the most widely-used machine learning libraries. In this project, we will be implementing an Automatic Differentiation library that can be used as the basis for analysis methods, including an Optimization and Newton’s Method extension that we will illustrate. 

## Background

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
|$\frac{u}{v}$|$\frac{u'v-uv'}{v^2}$|
_Table 1._


Now that we know how to compute the derivatives of atoms and how to handle derivatives on basic operations of functions, we want to visualize how can a function be decomposed into thse basic operations.
An important visualization of how a function can be decomposed into several elementary operations is the computational graph. 

For instance, we are going to draw the graph of the function $[f(x,y) =exp(-(sin(x)-cos(y))**2)$]

![Computational_graph.jpeg](Computational_graph.jpeg)

Therefore, the resulting quantity of interest can be explicitely expressed as a composition of several functions. In order to compute the derivative of these successive compositions, we are going to leverage a powerful mathematical tool: the **chain rule**.
A simple version of the chain rule can be expressed as follows : for $f$ and $g$ two functions, 

$[f(g)]' = g'*f'(g)$*

Therefore, from the computational graph we have seen above, we can express the derivative of the function encoded at every node by computing the derivative of this elementary operation and multiplyingby the derivative of the inner function. We know that we are able to compute the derivative of the elementary operation from the derivative of the different atom functions. 
Now, the question is to get the derivative of the inner function, that represents all the composition of the different operations encoded at every node until the current node. We do this iteratively, by applying at every node the chain rule with the previous composition operations. 
This suite of operations is encoded on the trace table. 


![Evaluation_table.png](Evaluation_table.png)

Therefore, from the previous points, we see that we will be able to compute value of the gradient of a function evaluated on a point by iteratively applying the chain rule at every operation node and leveraging a set of basic derivatives and operation on derivatives.


 

## Broader Impact:
As we discussed above, automatic differentiation has an incredible and a wide spread array of applications. And while there might not be a way to “misuse” the simple practice of taking derivatives, the applications in which Automatic Differentiation is used, especially Artificial Intelligence and Optimization, are ripe for misuse. Much in the same way that biased datasets lead to biased ML models and biased predictions, if our automatic differentiation library gives incorrect values or values that don’t have a high accuracy, the use of those values in real-life algorithms can lead to issues.  AI and Machine Learning, is, at its core, simply math - derivatives and gradients taken across a dataset to minimize a certain loss. If we incorrectly calculate these values, our algorithm could arrive at results that could have harmful effects on the very people they are meant to help. These algorithms range from weather prediction models to algorithms that determine bail amounts and the length of someone’s prison sentence. This comes back to an idea that our society is grappling with at the moment - while Automatic Differentiation and its uses in optimization (the extension that we implemented) and machine learning algorithms are powerful, we must take careful steps that every major decision the algorithm makes is human-reviewed by a diverse and responsible board with a knowledge of the subject area. Algorithms are only as powerful as the people who design them, and we can ensure that they are used in the best way possible by creating a culture of responsibility and ethical-based review in our codebase and community. 

## Software Inclusivity: 
We developed this project using github version control, a system that is often quite difficult and confusing to understand in the very beginning of stages of trying to use it. People who might not have access to a robust and in-depth computer science education might find it difficult to contribute to this project because they aren’t familiar with branching, committing, pushing, and pulling from github. These groups include several underrepresented minorities, including women, Black, Latinx, and Native Americans. This also includes people in rural and urban areas who might not have access to a CS education that includes concepts like git and version control. As someone who learned about these concepts quite late in her CS education as well, I can testify to the fact that “software development” as a practice can often be quite intimidating and can keep bright and talented people from pursuing the field because of how much prior knowledge of “arcane” (but necessary) things like git. 

Once a developer has made themselves comfortable with the codebase and with the practice of version control, we have a very fair and open system of code contribution. After making a branch, developers can create Pull Requests that are approved by either a few members of the team (if the PR involves a small change, such as a bug fix or comment) or all members of the team (if the PR involves major changes such as a new feature, test, or method). If reviewers have any questions, they leave them in the Comments space of the PR and the developer can respond to them as needed. If our organization was bigger, we would make sure to hire developers who are diverse in age, race, gender, and sexual orientation in order to make everyone feel completely comfortable adding radical features, challenging their peers and having respectful disagreements, and overall contributing to a more robust and dynamic codebase and product. 
