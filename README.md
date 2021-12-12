# Data Preprocessing

Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.

### [Complete Code](https://github.com/anupam215769/Data-Preprocessing-ML/blob/main/Data%20Preprocessing.ipynb)

### OR Run by Yourself Here

<a href="https://colab.research.google.com/github/anupam215769/Data-Preprocessing-ML/blob/main/Data%20Preprocessing.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

>Don't forget to add Data.csv file in colab. Otherwise it won't work.

## Step 1 - Importing the libraries

In order to perform data preprocessing using Python, we need to import some predefined Python libraries. These libraries are used to perform some specific jobs. There are three specific libraries that we will use for data preprocessing, which are:

**Numpy: Numpy Python library is used for including any type of mathematical operation in the code. It is the fundamental package for scientific calculation in Python. It also supports to add large, multidimensional arrays and matrices. So, in Python, we can import it as:**

```
import numpy as np
```

**Matplotlib: The second library is matplotlib, which is a Python 2D plotting library, and with this library, we need to import a sub-library pyplot. This library is used to plot any type of charts in Python for the code. It will be imported as below:**

```
import matplotlib.pyplot as plt
```

**Pandas: The last library is the Pandas library, which is one of the most famous Python libraries and used for importing and managing the datasets. It is an open-source data manipulation and analysis library. It will be imported as below:**

```
import pandas as pd
```

## Step 2 - Importing the dataset

Now we need to import the datasets which we have collected for our machine learning project.

Now to import the dataset, I will use **read_csv()** function of pandas library, which is used to read a csv file and performs various operations on it.

```
dataset = pd.read_csv('Data.csv')
```

#### Extracting dependent and independent variables: 

In machine learning, it is important to distinguish the matrix of features (independent variables) and dependent variables from dataset. In our dataset, there are three independent variables that are ***Country***, ***Age***, and ***Salary***.

```
X = dataset.iloc[:, :-1].values
```

In the above code, the first colon(:) is used to take all the rows, and the second colon(:) is for all the columns. Here I have used :-1, because I don't want to take the last column as it contains the dependent variable. So by doing this, we will get the matrix of features.

#### Extracting dependent variable:

In our dataset, there is a dependent variable which is ***Purchased***.
To extract dependent variables, again, I will use Pandas .iloc[] method.

```
y = dataset.iloc[:, -1].values
```

## Step 3 - Taking care of missing data

The next step of data preprocessing is to handle missing data in the datasets. If our dataset contains some missing data, then it may create a huge problem for our machine learning model. Hence it is necessary to handle missing values present in the dataset.

#### Ways to handle missing data:

There are mainly two ways to handle missing data, which are:

**By deleting the particular row:** The first way is used to commonly deal with null values. In this way, we just delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give the accurate output for the small dataset. But works fine for large dataset, where deleting few rows won't affect the result.

**By calculating the mean:** In this way, we will calculate the mean of that column or row which contains any missing value and will put it on the place of missing value. This strategy is useful for the features which have numeric data such as age, salary, year, etc. Here, in my code I have used this approach.

To handle missing values, we will use Scikit-learn library in our code, which contains various libraries for building machine learning models

```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

## Step 4 - Encoding categorical data

Categorical data is data which has some categories such as, in our dataset; there are two categorical variable, ***Country***, and ***Purchased***.

Since machine learning model completely works on mathematics and numbers, but if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.

#### Dummy Variables:

Dummy variables are those variables which have values 0 or 1. The 1 value gives the presence of that variable in a particular column, and rest variables become 0. With dummy encoding, we will have a number of columns equal to the number of categories.


### 4.1 Encoding the Independent Variable

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

### 4.2 Encoding the Dependent Variable

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

## Step 5 - Splitting the dataset into the Training set and Test set

In machine learning data preprocessing, we divide our dataset into a training set and test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.

Suppose, if we have given training to our machine learning model by a dataset and we test it by a completely different dataset. Then, it will create difficulties for our model to understand the correlations between the models.

If we train our model very well and its training accuracy is also very high, but we provide a new dataset to it, then it will decrease the performance. So we always try to make a machine learning model which performs well with the training set and also with the test dataset. Here, we can define these datasets as:

![dataset](https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-5.png)

**Training Set:** A subset of dataset to train the machine learning model, and we already know the output.

**Test set:** A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.

For splitting the dataset, we will use the below lines of code:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```

* x_train: features for the training data
* x_test: features for testing data
* y_train: Dependent variables for training data
* y_test: Independent variable for testing data

In **train_test_split()** function, we have passed four parameters in which first two are for arrays of data, and test_size is for specifying the size of the test set. The **test_size** maybe .5, .3, or .2, which tells the dividing ratio of training and testing sets.

The last parameter **random_state** is used to set a seed for a random generator so that you always get the same result, and I have used value for this is 1.

## Step 6 - Feature Scaling

Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.

As we can see, the age and salary column values are not on the same scale. A machine learning model is based on **Euclidean distance**, and if we do not scale the variable, then it will cause some issue in our machine learning model.

Euclidean distance is given as:

![distance](https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-8.png)

If we compute any two values from age and salary, then salary values will dominate the age values, and it will produce an incorrect result. So to remove this issue, we need to perform feature scaling for machine learning.

There are two ways to perform feature scaling in machine learning:

#### Standardization

![distance](https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-9.png)

#### Normalization

![distance](https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-10.png)

>Here, I have used the standardization method for the dataset.

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```


## References

https://scikit-learn.org/stable/

https://numpy.org/doc/stable/

https://pandas.pydata.org/docs/


## Related Repositories

### [Regression](https://github.com/anupam215769/Regression-ML)

### [Classification](https://github.com/anupam215769/Classification-ML)

### [Clustering](https://github.com/anupam215769/Clustering-ML)

## Credit

**Coded By**

[Anupam Verma](https://github.com/anupam215769)

<a href="https://github.com/anupam215769/Data-Preprocessing-ML/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=anupam215769/Data-Preprocessing-ML" />
</a>

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anupam-verma-383855223/)
