# Machine-learning-roadMap

[Machine Learning is the] field of study that gives computers the ability to learn
without being explicitly programmed.
—Arthur Samuel, 1959


Repository containing a portfolio of machine learning projects and weekly progress towards becoming better at Machine Learning algorithms.

# Week 1

### Pick a language, and in my opinion, Python is the best for this. 
Why? Easy syntax, extensive documentation, great community, availability of resources. 

### Create a GitHub profile and learn Git

Git is a powerful version control system that every single person in the STEM field should learn. GitHub is a way of showing the world you know something. Simply writing, I know Python will not be enough for companies. You have to prove you know the Python by showing your projects. 

# Week 2

### Data Analyse Libraries 
Learn Pandas and NumPy, the two most important data analysis libraries that will help you to deal with the data preprocessing step. Machine Learning is not all about applying the algorithms to the given dataset. Sometimes, you will want to alter the data according to your needs, or even the data you have will include unnecessary information. Thus, these two libraries will help you to prepare the data. In week 3, we will choose a data visualization library. 


### Learn the following Machine Learning related terms

Supervised Learning | Unsupervised Learning | Semi-supervised Learning | Reinforcement Learning   
Feature Extraction | Feature Selection    
Noise | Outliers | Garbage Data   
Batch Learning | Online Learning   
Overfitting | Regularization | Underfitting   
Training Dataset | Testing Dataset   



### Go over the basic calculus
One big misconsumption on Machine Learning that discourage people is Calculus. You don't have to be Calculus expert in order to start ML. 

# Week 3

### Go over the Basic Statistics

Mean | Standard Deviation | Variance   
Covariance | Correlation | Normal Distribution  

### Go over the Basic Linear Algebra

You don't need to be a master in linear algebra. Here are the basic skills that everyone who are interested in ML should know.

Matrix multiplication, Addition and Subtraction   
Linear Transformation
Transpose, identiy matrix, inverse 


### Artificial Neurons and Perceptron

### Implementing Perceptron algorithm and fitting Irish dataset to train it

![SLP](images/single-layer-perceptron-in-tensorflow2.png)

Get yourself familiar with terms: bias, weight, activation function, threshold, convergence.

## Common Question
### What is the need of bias in the perceptron algorithm?

Weights define how each feature is affective on the classifier, i.e., it only changes the shape of your activation function. However, to be able to shift the activation function along the axis without effecting steepness of it, we need a term bias. 

Imagine a situation where both of your features are zero; however, you need an output of 1. No matter which weight you choose, the calculation of the result will be zero. This is where the importance of the bias comes to place. When you have a bias that is always equal to one, you can adjust the weight to get the result of 1. 

You can think it as a y-intercept on the line equation: y = mx + b

The weight parameter is 'm' where 'b' stands for the y-intercept of the given x. If you don't have a 'b', your line will always cross the origin which will not be effective in every classifier. To shift the classifier to left or right, we need a y-intercept. 

# Week 4

### Gradient Descent Algorithm

Gradient descent is a famous optimization algorithm that can be used in many areas in machine learning such as clustering, logistic and linear regression. In linear regression, we use GD to find optimal line that fits the given poins in 2D. The main idea is to find the best m(slope) and b(y-intercept) that minimizes the objective function!

![SLP](images/gradient_descent.png)


### Linear Regression 

Linear Regression, perhaps one of the fundemental supervised machine learning examples, is an algorithm that uses statistics to predict the corresponding output to the given input based on the previous input-output pairs. The goal is to find best fit line to the given points. The objective function of the algorithm is sum of squared errors which we try to minimize it.  



# Week 5

### Adaptive Linear Neurons

It is an improvement over Perceptron algorithm which uses obj function to update weight rather than a threshold function. 

![SLP](images/adaline.png)



### Stochastic Gradient Descent

It is an optimization technique like Batch Gradient Descent;however, it is more applicable to big dataset. Unlike batch gradient descent, it only picks sample from the given set of features where BGD uses whole dataset. 
