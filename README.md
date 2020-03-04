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

It is an improvement over Perceptron algorithm which uses activation function to update the weights where Perceptron uses unit step function.

In perceptron algorithm, weights are updated after ealuating each sample where ADALINE uses gradient of the whole dataset (objective function) to update the weights.

![SLP](images/adaline.png)



### Stochastic Gradient Descent

It is an optimization technique like Batch Gradient Descent;however, it is more applicable to big dataset. Unlike batch gradient descent, it only picks sample from the given set of features where BGD uses whole dataset. 

### One-Vs-Rest Strategy

Adaline and Perceptron are binary classifier which can classify 2 classes. Using the one-vs-rest strategy, we can use these binary classifier to a multivariate classifier. 
For example:

If we have 3 classes (A,B,C), train 3 classifiers.
classifierA => class A = 1, classes (B and C) = -1
classifierB => class B = 1, classes (A and C) = -1
classifierC => class C = 1, classes (A and B) = -1

Where classifierA, classifierB, classifierC are the same algorithm (i.e., perceptron)

And to predict the class to which an instance x belongs, it is necessary to predict for each of these n classifiers.
classifierA.predict (x)
classifierB.predict (x)
classifierC.predict (x)

Then, we will take as the class of x where we get the 1.

For example :

classifierA.predict (x) ==> -1
classifierB.predict (x) ==> 1
classifierC.predict (x) ==> -1

The class of x is B.


# Week 6

This week, we will get back to Linear Regression, especially to the Multivariate Regression where we have more than one feature.

## Label Encoder
Unfortunately, not all the features will have a continious values all the time. In case of categorical feature, we need to conver this into the numbers to be able to use it in our model. Imagine a case where one feature vector contains following countries: 'Turkey' and 'USA'. Obviously, you will not able to able to feed this feature to the model. Using the label encoder, we will represent these countries as 0 or 1. For the sake of example, we can represent Turkey as 0 and USA as 1. 

## One-Hot Encoder

Giving this 0 and 1 representation can mislead the model. It might give the sense of the representation has an some sort of order/strength. To prevent from this problem, we use one-hot encoder, which will seperate the Turkey and USA into two columns. This is what one-hot encoding is. 

## Feature Extraction-Backward Elimination

In real life, what do we do when we have garbage? Well, I personally throw it away. As in real life, when we have a dataset to feed the Machine Learning algorithm, we need to separate the garbage features/columns from the good one. Because garbage in garbage out.    


In machine learning linguistic, this process is called feature extraction. Using the FE, we can extract the data that will have no impact/bad impact on the output of the model. One of the simple ways to do this is Backward Elimination, which uses the hypothesis testing, specifically p-value, to determine what is garbage data.   

But why do we need Feature Extraction? 
Here are a couple of reasons. 
Decrease the complexity, improved accuracy and performance, and faster training time. 

## Classification Algorithms

## Logistic Regression
Logistic Regression is an classification algorithm that produces limited range continious variable using the probability theory(0-1). 

## K-Nearest Neighbor

Knn, perhaps one of the most straightforward Machine Learning algorithm, is a classification algorithm works in the following steps:

1) Choose the number of neighbors, 
2) Get a data point you want to classify, 
3) Measure the **distance** between the data point and the neighbors from the closes neighbors
4) Among these K neighbors, count the number of data points in each category
5) Assign new instance to the majority category


## Norms and Distance Metrics

Distance metric is used in both supervised and unsupervised machine learning algorithms to measure the distance between the data point. Using this distance metric, we can assign a new data instances to the corresponding categories. Before getting details of it, we need to understand the norm. 

In linear algebra, norm is a function which takes a vector as an input and produces scalar value that shows how big the vector is.In other words, it is a distance between the vector and origin. Using norms, we can define distance metrics. The most commonly used norms are L1 and L2. 

## Week 7
## Support Vector Machines

The goal of the SVM is to find the best line that classifies the data points.

SVM has three cases.   

The first case is when two classes can be separable with an optimal hyperplane. Pretty boring   


The second case is when we relax our model and accept some misclassified instances. This is also known as smooth SVM   


The third case is when there is absolutely no way to find an optimal hyper line. In this case, we use kernel tricks to expand the two-dimension into third (Z). When we transform data points to the third dimension, we can find a line to separate them. (Mad respect). Actually, with Radial kernel, even transformation into an infinite dimension possible, but let's not talk about it. 


## Week 8
## Decision Tree

Decision Tree is a supervised learning algorithm that can utilize both prediction and classification. The main goal of the algorithm is to divide until you reach **pure** data points. 

In the form of a tree structure where each node is either:
(1) a leaf node, indicating a class of instance or   
(2) a decision node, specifying a test to be carried out on a single attribute value, with one branch and a sub-tree for each possible outcome of the test    

### Overfitting

In machine learning algorithms, we usually fit the training dataset to the created model. Overfitting occurs when the model works well with the training dataset; however, it produces poor performance on the new instances, a.k.a. test dataset. The signal is the underlying pattern that we want our classifier to learn, whereas noise is irrelevant data points. When your classifier learns the noise rather than a signal, it can't do well on the test dataset.   

The decision tree algorithm very likely to get affected by overfitting issues. There are a couple of techniques to overcome this issue. 
The size of the tree has great impact on the accuracy of the model. One of the optimization technique used in DT is to find the optimal **max_leaf_nodes** which will help our us to get better accuracy. Here is a useful python script. 

    def get_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeClassification(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        acc = accuracy_score(val_y, preds_val)
        
                
### Pruning
