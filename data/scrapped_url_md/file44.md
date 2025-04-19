- [Data Science](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Data Science Projects](https://www.geeksforgeeks.org/top-data-science-projects/)
- [Data Analysis](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Data Visualization](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-projects/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Computer Vision](https://www.geeksforgeeks.org/computer-vision/)
- [Artificial Intelligence](https://www.geeksforgeeks.org/artificial-intelligence/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/implementation-of-lasso-ridge-and-elastic-net/?type%3Darticle%26id%3D383976&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
ML \| MultiLabel Ranking Metrics - Coverage Error\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ml-multilabel-ranking-metrics-coverage-error/)

# Implementation of Lasso, Ridge and Elastic Net

Last Updated : 15 May, 2021

Comments

Improve

Suggest changes

Like Article

Like

Report

In this article, we will look into the implementation of different regularization techniques. First, we will start with multiple linear regression. For that, we require the python3 environment with sci-kit learn and pandas preinstall. We can also use google collaboratory or any other jupyter notebook environment.

First, we need to import some packages into our environment.

- Python3

## Python3

|     |
| --- |
| `import` `pandas as pd`<br>`import` `numpy as np`<br>`import` `matplotlib.pyplot as plt`<br>`from` `sklearn ` `import` `datasets`<br>`from` `sklearn.model_selection ` `import` `train_test_split`<br>`from` `sklearn.linear_model ` `import` `LinearRegression` |

```

```

```

```

We are going to use the Boston house prediction dataset. This dataset is present in the **datasets** module of sklearn (scikit-learn) library. We can import this dataset as follows.

- Python3

## Python3

|     |
| --- |
| `# Loading pre-defined Boston Dataset`<br>`boston_dataset ` `=` `datasets.load_boston()`<br>`print` `(boston_dataset.DESCR)` |

```

```

```

```

**Output:**

![Boston dataset description](https://media.geeksforgeeks.org/wp-content/uploads/20200227171407/Capture389.png)

We can conclude from the above description that we have 13 independent variable and one dependent (House price) variable. Now we need to check for a correlation between independent and dependent variable. We can use scatterplot/corrplot for this.

- Python3

## Python3

|     |
| --- |
| `# Generate scatter plot of independent vs Dependent variable`<br>`plt.style.use(` `'ggplot'` `)`<br>`fig ` `=` `plt.figure(figsize ` `=` `(` `18` `, ` `18` `))`<br>`for` `index, feature_name ` `in` `enumerate` `(boston_dataset.feature_names):`<br>`    ``ax ` `=` `fig.add_subplot(` `4` `, ` `4` `, index ` `+` `1` `)`<br>`    ``ax.scatter(boston_dataset.data[:, index], boston_dataset.target)`<br>`    ``ax.set_ylabel(` `'House Price'` `, size ` `=` `12` `)`<br>`    ``ax.set_xlabel(feature_name, size ` `=` `12` `)`<br>`plt.show()` |

```

```

```

```

The above code produce scatter plots of different independent variable with target variable as shown below

![scatter plots](https://media.geeksforgeeks.org/wp-content/uploads/20200227174719/scatters.jpg)

We can observe from the above scatter plots that some of the independent variables are not very much correlated (either positively or negatively) with the target variable. These variables will get their coefficients to be reduced in regularization.

**Code : Python code to pre-process the data.**

- Python3

## Python3

|     |
| --- |
| `# Load the dataset into Pandas Dataframe`<br>`boston_pd ` `=` `pd.DataFrame(boston_dataset.data)`<br>`boston_pd.columns ` `=` `boston_dataset.feature_names`<br>`boston_pd_target ` `=` `np.asarray(boston_dataset.target)`<br>`boston_pd[` `'House Price'` `] ` `=` `pd.Series(boston_pd_target)`<br>`# input `<br>`X ` `=` `boston_pd.iloc[:, :` `-` `1` `]`<br>`#output`<br>`Y ` `=` `boston_pd.iloc[:, ` `-` `1` `]`<br>`print` `(boston_pd.head())` |

```

```

```

```

Now, we apply train-test split to divide the dataset into two parts, one for training and another for testing. We will be using 25% of the data for testing.

- Python3

## Python3

|     |
| --- |
| `x_train, x_test, y_train, y_test ` `=` `train_test_split(`<br>`    ``boston_pd.iloc[:, :` `-` `1` `], boston_pd.iloc[:, ` `-` `1` `], `<br>`    ``test_size ` `=` `0.25` `)`<br>`print` `(` `"Train data shape of X = % s and Y = % s : "` `%` `(`<br>`    ``x_train.shape, y_train.shape))`<br>`print` `(` `"Test data shape of X = % s and Y = % s : "` `%` `(`<br>`    ``x_test.shape, y_test.shape))` |

```

```

```

```

**Multiple (Linear) Regression**

Now it’s the right time to test the models. We will be using multiple Linear Regression first. We train the model on training data and calculate the MSE on test.

- Python3

## Python3

|     |
| --- |
| `# Apply multiple Linear Regression Model`<br>`lreg ` `=` `LinearRegression()`<br>`lreg.fit(x_train, y_train)`<br>`# Generate Prediction on test set`<br>`lreg_y_pred ` `=` `lreg.predict(x_test)`<br>`# calculating Mean Squared Error (mse)`<br>`mean_squared_error ` `=` `np.mean((lreg_y_pred ` `-` `y_test)` `*` `*` `2` `)`<br>`print` `(` `"Mean squared Error on test set : "` `, mean_squared_error)`<br>`# Putting together the coefficient and their corresponding variable names `<br>`lreg_coefficient ` `=` `pd.DataFrame()`<br>`lreg_coefficient[` `"Columns"` `] ` `=` `x_train.columns`<br>`lreg_coefficient[` `'Coefficient Estimate'` `] ` `=` `pd.Series(lreg.coef_)`<br>`print` `(lreg_coefficient)` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20200228133932/lregcoefficient.png)

Let’s plot a bar chart of above coefficients using matplotlib plotting library.

- Python3

## Python3

|     |
| --- |
| `# plotting the coefficient score`<br>`fig, ax ` `=` `plt.subplots(figsize ` `=` `(` `20` `, ` `10` `))`<br>`color ` `=` `[` `'tab:gray'` `, ` `'tab:blue'` `, ` `'tab:orange'` `, `<br>`'tab:green'` `, ` `'tab:red'` `, ` `'tab:purple'` `, ` `'tab:brown'` `, `<br>`'tab:pink'` `, ` `'tab:gray'` `, ` `'tab:olive'` `, ` `'tab:cyan'` `, `<br>`'tab:orange'` `, ` `'tab:green'` `, ` `'tab:blue'` `, ` `'tab:olive'` `]`<br>`ax.bar(lreg_coefficient[` `"Columns"` `], `<br>`lreg_coefficient[` `'Coefficient Estimate'` `], `<br>`color ` `=` `color)`<br>`ax.spines[` `'bottom'` `].set_position(` `'zero'` `)`<br>`plt.style.use(` `'ggplot'` `)`<br>`plt.show()` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20200228152832/lreg_coefficient_prime.png)

As we can observe that lots of the variables have an insignificant coefficient, these coefficients did not contribute to the model very much and need to regulate or even eliminate some of these variables.

**Ridge Regression:**

Ridge Regression added a term in ordinary least square error function that regularizes the value of coefficients of variables. This term is the sum of squares of coefficient multiplied by the parameter The motive of adding this term is to penalize the variable corresponding to that coefficient not very much correlated to the target variable. This term is called **L2** regularization.

**Code : Python code to use Ridge regression**

- Python3

## Python3

|     |
| --- |
| `# import ridge regression from sklearn library`<br>`from` `sklearn.linear_model ` `import` `Ridge`<br>`# Train the model `<br>`ridgeR ` `=` `Ridge(alpha ` `=` `1` `)`<br>`ridgeR.fit(x_train, y_train)`<br>`y_pred ` `=` `ridgeR.predict(x_test)`<br>`# calculate mean square error`<br>`mean_squared_error_ridge ` `=` `np.mean((y_pred ` `-` `y_test)` `*` `*` `2` `)`<br>`print` `(mean_squared_error_ridge)`<br>`# get ridge coefficient and print them`<br>`ridge_coefficient ` `=` `pd.DataFrame()`<br>`ridge_coefficient[` `"Columns"` `]` `=` `x_train.columns`<br>`ridge_coefficient[` `'Coefficient Estimate'` `] ` `=` `pd.Series(ridgeR.coef_)`<br>`print` `(ridge_coefficient)` |

```

```

```

```

**Output:** The value of MSE error and the dataframe with ridge coefficients.

![RidgRegcoefficient](https://media.geeksforgeeks.org/wp-content/uploads/20200228163329/lregcoefficient1.png)

The bar plot of above data is:

[![rigdgeatAlpha1](https://media.geeksforgeeks.org/wp-content/uploads/20200228165206/rigdgeatAlpha1.png)](https://media.geeksforgeeks.org/wp-content/uploads/20200228165206/rigdgeatAlpha1.png)

Ridge Regression at ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)=1

In the above graph we take ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 1.

Let’s look at another bar plot with ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 10

[![](https://media.geeksforgeeks.org/wp-content/uploads/20200228170029/ridgeatalpha10.png)](https://media.geeksforgeeks.org/wp-content/uploads/20200228170029/ridgeatalpha10.png)

Ridge regression at ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 10

As we can observe from the above plots that ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)helps in regularizing the coefficient and make them converge faster.

Notice that the above graphs can be misleading in a way that it shows some of the coefficients become zero. In Ridge Regularization, the coefficients can never be 0, they are just too small to observe in above plots.

**Lasso Regression:**

Lasso Regression is similar to Ridge regression except here we add Mean Absolute value of coefficients in place of mean square value. Unlike Ridge Regression, Lasso regression can completely eliminate the variable by reducing its coefficient value to 0. The new term we added to Ordinary Least Square(OLS) is called **L1** Regularization.

**Code : Python code implementing the Lasso Regression**

- Python3

## Python3

|     |
| --- |
| `# import Lasso regression from sklearn library`<br>`from` `sklearn.linear_model ` `import` `Lasso`<br>`# Train the model`<br>`lasso ` `=` `Lasso(alpha ` `=` `1` `)`<br>`lasso.fit(x_train, y_train)`<br>`y_pred1 ` `=` `lasso.predict(x_test)`<br>`# Calculate Mean Squared Error`<br>`mean_squared_error ` `=` `np.mean((y_pred1 ` `-` `y_test)` `*` `*` `2` `)`<br>`print` `(` `"Mean squared error on test set"` `, mean_squared_error)`<br>`lasso_coeff ` `=` `pd.DataFrame()`<br>`lasso_coeff[` `"Columns"` `] ` `=` `x_train.columns`<br>`lasso_coeff[` `'Coefficient Estimate'` `] ` `=` `pd.Series(lasso.coef_)`<br>`print` `(lasso_coeff)` |

```

```

```

```

**Output:** The value of MSE error and the dataframe with Lasso coefficients.

[![lassowithalpaha11](https://media.geeksforgeeks.org/wp-content/uploads/20200228175828/lassowithalpaha11-268x300.png)](https://media.geeksforgeeks.org/wp-content/uploads/20200228175828/lassowithalpaha11.png)

Lasso Regression with ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 1

The bar plot of above coefficients:

[![Lasso Regression Chart](https://media.geeksforgeeks.org/wp-content/uploads/20200228173913/lassowithalpha1.png)](https://media.geeksforgeeks.org/wp-content/uploads/20200228173913/lassowithalpha1.png)

Lasso Regression with ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)=1

The Lasso Regression gave same result that ridge regression gave, when we increase the value of ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg). Let’s look at another plot at ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 10.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200228180411/lassowithalpha10.png)

**Elastic Net :**

In elastic Net Regularization we added the both terms of L1 and L2 to get the final loss function. This leads us to reduce the following loss function:

![L_{elastic-Net}\left ( \hat\beta \right )= \left ( \sum \left ( y - x_i^J\hat{\beta} \right )^2 \right )/2n+\lambda \left ( (1 -\alpha )/2 * \sum_{j=1}^{m} \hat{\beta_{j}^{2}}+\alpha * \sum_{j=1}^{m} \left \| \hat{\beta_{j}} \right \| \right) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1a041331098799f7f662249273e25b08_l3.svg)

where ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)is between 0 and 1. when ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 1, It reduces the penalty term to L1 penalty and if ![\alpha ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-05b37788c6a2f1f1c5e67ba691bc1606_l3.svg)= 0, it reduces that term to L2

penalty.

**Code : Python code implementing the Elastic Net**

- Python3

## Python3

|     |
| --- |
| `# import model`<br>`from` `sklearn.linear_model ` `import` `ElasticNet`<br>`# Train the model`<br>`e_net ` `=` `ElasticNet(alpha ` `=` `1` `)`<br>`e_net.fit(x_train, y_train)`<br>`# calculate the prediction and mean square error`<br>`y_pred_elastic ` `=` `e_net.predict(x_test)`<br>`mean_squared_error ` `=` `np.mean((y_pred_elastic ` `-` `y_test)` `*` `*` `2` `)`<br>`print` `(` `"Mean Squared Error on test set"` `, mean_squared_error)`<br>`e_net_coeff ` `=` `pd.DataFrame()`<br>`e_net_coeff[` `"Columns"` `] ` `=` `x_train.columns`<br>`e_net_coeff[` `'Coefficient Estimate'` `] ` `=` `pd.Series(e_net.coef_)`<br>`e_net_coeff` |

```

```

```

```

**Output:**

Elastic\_Net

**Bar plot of above coefficients:**

![Elastic Net Plot](https://media.geeksforgeeks.org/wp-content/uploads/20200302152322/elasticNetco.png)

**Conclusion :**

From the above analysis we can reach the following conclusion about different regularization methods:

- Regularization is used to reduce the dependence on any particular independent variable by adding the penalty term to the Loss function. This term prevents the coefficients of the independent variables to take extreme values.

- Ridge Regression adds L2 regularization penalty term to loss function. This term reduces the coefficients but does not make them 0 and thus doesn’t eliminate any independent variable completely. It can be used to measure the impact of the different independent variables.

- Lasso Regression adds L1 regularization penalty term to loss function. This term reduces the coefficients as well as makes them 0 thus effectively eliminate the corresponding independent variable completely. It can be used for feature selection etc.

- Elastic Net is a combination of both of the above regularization. It contains both the L1 and L2 as its penalty term. It performs better than Ridge and Lasso Regression for most of the test cases.


Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ml-multilabel-ranking-metrics-coverage-error/)

[ML \| MultiLabel Ranking Metrics - Coverage Error](https://www.geeksforgeeks.org/ml-multilabel-ranking-metrics-coverage-error/)

[P](https://www.geeksforgeeks.org/user/pawangfg/)

[pawangfg](https://www.geeksforgeeks.org/user/pawangfg/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-Regression](https://www.geeksforgeeks.org/tag/ml-regression/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Implementation of Elastic Net Regression From Scratch\\
\\
\\
Prerequisites: Linear RegressionGradient DescentLasso & Ridge RegressionIntroduction: Elastic-Net Regression is a modification of Linear Regression which shares the same hypothetical function for prediction. The cost function of Linear Regression is represented by J. \[Tex\]\\frac{1}{m} \\sum\_{i=1}^\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)
[Lasso vs Ridge vs Elastic Net \| ML\\
\\
\\
Regularization methods such as Lasso, Ridge and Elastic Net are important in improving linear regression models by avoiding overfitting, solving multicollinearity and feature selection. These methods enhance the model's predictive accuracy and robustness. Below is a concise explanation of how each t\\
\\
5 min read](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/)
[Implementation of KNN using OpenCV\\
\\
\\
KNN is one of the most widely used classification algorithms that is used in machine learning. To know more about the KNN algorithm read here KNN algorithm Today we are going to see how we can implement this algorithm in OpenCV and how we can visualize the results in 2D plane showing different featu\\
\\
3 min read](https://www.geeksforgeeks.org/implementation-of-knn-using-opencv/)
[Implementation of Lasso Regression From Scratch using Python\\
\\
\\
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that combines prediction with feature selection. It does this by adding a penalty term to the cost function shrinking less relevant feature's coefficients to zero. This makes it effective for high-dim\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/)
[Implementation of Teaching Learning Based Optimization\\
\\
\\
The previous article Teaching Learning Based Optimization (TLBO) talked about the inspiration of teaching learning-based optimization, it's mathematical modeling and algorithms. In this article we will implement Teaching learning-based optimization (TLBO) for two fitness functions 1) Rastrigin funct\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-teaching-learning-based-optimization/)
[ANN - Implementation of Self Organizing Neural Network (SONN) from Scratch\\
\\
\\
Prerequisite: ANN \| Self Organizing Neural Network (SONN) Learning Algorithm To implement a SONN, here are some essential consideration- Construct a Self Organizing Neural Network (SONN) or Kohonen Network with 100 neurons arranged in a 2-dimensional matrix with 10 rows and 10 columns Train the netw\\
\\
4 min read](https://www.geeksforgeeks.org/ann-implementation-of-self-organizing-neural-network-sonn-from-scratch/)
[Implementation of Locally Weighted Linear Regression\\
\\
\\
LOESS or LOWESS are non-parametric regression methods that combine multiple regression models in a k-nearest-neighbor-based meta-model. LOESS combines much of the simplicity of linear least squares regression with the flexibility of nonlinear regression. It does this by fitting simple models to loca\\
\\
3 min read](https://www.geeksforgeeks.org/implementation-of-locally-weighted-linear-regression/)
[Implement Canny Edge Detector in Python using OpenCV\\
\\
\\
In this article, we will learn the working of the popular Canny edge detection algorithm developed by John F. Canny in 1986. Usually, in Matlab and OpenCV we use the canny edge detection for many popular tasks in edge detection such as lane detection, sketching, border removal, now we will learn the\\
\\
5 min read](https://www.geeksforgeeks.org/implement-canny-edge-detector-in-python-using-opencv/)
[Comparison between L1-LASSO and Linear SVM\\
\\
\\
Within machine learning, linear Support Vector Machines (SVM) and L1-regularized Least Absolute Shrinkage and Selection Operator (LASSO) regression are powerful methods for classification and regression, respectively. Although the goal of both approaches is to locate a linear decision boundary, they\\
\\
3 min read](https://www.geeksforgeeks.org/comparison-between-l1-lasso-and-linear-svm/)
[Blind source separation using FastICA in Scikit Learn\\
\\
\\
FastICA is the most popular method and the fastest algorithm to perform Independent Component Analysis. It can be used to separate all the individual signals from a mixed signal. Independent Component Analysis(ICA) is a method where it performs a search for finding mutually independent non-Gaussian\\
\\
8 min read](https://www.geeksforgeeks.org/blind-source-separation-using-fastica-in-scikit-learn/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/implementation-of-lasso-ridge-and-elastic-net/)

Improvement

Suggest changes

Suggest Changes

Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.

![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)

Create Improvement

Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.

![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)

Suggest Changes

min 4 words, max Words Limit:1000

## Thank You!

Your suggestions are valuable to us.

## What kind of Experience do you want to share?

[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=778866795.1745055852&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1267543490)