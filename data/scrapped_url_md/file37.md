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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-linear-regression/?type%3Darticle%26id%3D225468&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Ordinary Least Squares (OLS) using statsmodels\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)

# Linear Regression in Machine learning

Last Updated : 05 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It provides valuable insights for prediction and data analysis. This article will explore its types, assumptions, implementation, advantages and evaluation metrics.

## **Understanding Linear Regression**

**Linear regression** is also a type of [**supervised machine-learning algorithm**](https://www.geeksforgeeks.org/supervised-machine-learning/) that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. It computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation with observed data. It predicts the continuous output variables based on the independent input variable.

For example if we want to predict house price we consider various factor such as house age, distance from the main road, location, area and number of room, linear regression uses all these parameter to predict house price as it consider a linear relation between all these features and price of house.

### Why Linear Regression is Important?

The interpretability of linear regression is one of its greatest strengths. The model’s equation offers clear coefficients that illustrate the influence of each independent variable on the dependent variable, enhancing our understanding of the underlying relationships. Its simplicity is a significant advantage; linear regression is transparent, easy to implement, and serves as a foundational concept for more advanced algorithms.

Now that we have discussed why linear regression is important now we will discuss its working based on best fit line in regression.

## What is the best Fit Line?

Our primary objective while using linear regression is to locate the best-fit line, which implies that the error between the predicted and actual values should be kept to a minimum. There will be the least error in the best-fit line.

The best Fit Line equation provides a straight line that represents the relationship between the dependent and independent variables. The slope of the line indicates how much the dependent variable changes for a unit change in the independent variable(s).

![Linear Regression in Machine learning](https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png)

Linear Regression

Here Y is called a dependent or target variable and X is called an independent variable also known as the predictor of Y. There are many types of functions or modules that can be used for regression. A linear function is the simplest type of function. Here, X may be a single feature or multiple features representing the problem.

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x)). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best-fit line for our model.

In linear regression some hypothesis are made to ensure reliability of the model’s results.

### **Hypothesis function in Linear Regression**

Assumptions are:

- **Linearity:** It assumes that there is a linear relationship between the independent and dependent variables. This means that changes in the independent variable lead to proportional changes in the dependent variable.
- **Independence**: The observations should be independent from each other that is the errors from one observation should not influence other.

As we have discussed that our independent feature is the experience i.e X and the respective salary Y is the dependent variable. Let’s assume there is a linear relationship between X and Y then the salary can be predicted using:

Y^=θ1+θ2X\\hat{Y} = \\theta\_1 + \\theta\_2X Y^=θ1​+θ2​X

OR

y^i=θ1+θ2xi\\hat{y}\_i = \\theta\_1 + \\theta\_2x\_iy^​i​=θ1​+θ2​xi​

Here,

- yiϵY(i=1,2,⋯,n)y\_i \\epsilon Y \\;\\; (i= 1,2, \\cdots , n)
























yi​ϵY(i=1,2,⋯,n) are labels to data (Supervised learning)
- xiϵX(i=1,2,⋯,n)x\_i \\epsilon X \\;\\; (i= 1,2, \\cdots , n)
























xi​ϵX(i=1,2,⋯,n) are the input independent training data (univariate – one input variable(parameter))
- yi^ϵY^(i=1,2,⋯,n)\\hat{y\_i} \\epsilon \\hat{Y} \\;\\; (i= 1,2, \\cdots , n)
























yi​^​ϵY^(i=1,2,⋯,n) are the predicted values.

The model gets the best regression fit line by finding the best θ1 and θ2 values.

- **θ** ****1**** **:** intercept
- **θ** ****2**** **:** coefficient of x

Once we find the best θ1 and θ2 values, we get the best-fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.

### **How to update θ** ****1**** **and θ** ****2**** **values to get the best-fit line?**

To achieve the best-fit regression line, the model aims to predict the target value Y^\\hat{Y}     Y^ such that the error difference between the predicted value Y^\\hat{Y}     Y^ and the true value Y is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimizes the error between the predicted y value (pred) and the true y value (y).

minimize1n∑i=1n(yi^−yi)2minimize\\frac{1}{n}\\sum\_{i=1}^{n}(\\hat{y\_i}-y\_i)^2minimizen1​∑i=1n​(yi​^​−yi​)2

## Types of Linear Regression

When there is only one independent feature it is known as **Simple Linear Regression** or [Univariate Linear Regression](https://www.geeksforgeeks.org/univariate-linear-regression-in-python/) and when there are more than one feature it is known as **Multiple Linear Regression** or [Multivariate Regression](https://www.geeksforgeeks.org/multivariate-regression/).

### **1\. Simple Linear Regression**

[Simple linear regression](https://www.geeksforgeeks.org/simple-linear-regression-in-python/) is the simplest form of linear regression and it involves only one independent variable and one dependent variable. The equation for simple linear regression is:

y=β0+β1Xy=\\beta\_{0}+\\beta\_{1}Xy=β0​+β1​X

where:

- Y is the dependent variable
- X is the independent variable
- β0 is the intercept
- β1 is the slope

### Assumptions of Simple Linear Regression

Linear regression is a powerful tool for understanding and predicting the behavior of a variable, however, it needs to meet a few conditions in order to be accurate and dependable solutions.

1. **Linearity**: The independent and dependent variables have a linear relationship with one another. This implies that changes in the dependent variable follow those in the independent variable(s) in a linear fashion. This means that there should be a straight line that can be drawn through the data points. If the relationship is not linear, then linear regression will not be an accurate model.

![](https://media.geeksforgeeks.org/wp-content/uploads/20231123113044/python-linear-regression-4.png)
2. **Independence**: The observations in the dataset are independent of each other. This means that the value of the dependent variable for one observation does not depend on the value of the dependent variable for another observation. If the observations are not independent, then linear regression will not be an accurate model.
3. **Homoscedasticity**: Across all levels of the independent variable(s), the variance of the errors is constant. This indicates that the amount of the independent variable(s) has no impact on the variance of the errors. If the variance of the residuals is not constant, then linear regression will not be an accurate model.


![](https://media.geeksforgeeks.org/wp-content/uploads/20231123113103/python-linear-regression-5.png)

Homoscedasticity in Linear Regression

4. **Normality**: The residuals should be normally distributed. This means that the residuals should follow a bell-shaped curve. If the residuals are not normally distributed, then linear regression will not be an accurate model.

### Use Case of Simple Linear Regression

- In a case study evaluating student performance analysts use simple linear regression to examine the relationship between study hours and exam scores. By collecting data on the number of hours students studied and their corresponding exam results the analysts developed a model that reveal correlation, for each additional hour spent studying, students exam scores increased by an average of 5 points. This case highlights the utility of simple linear regression in understanding and improving academic performance.
- Another case study focus on marketing and sales where businesses uses simple linear regression to forecast sales based on historical data particularly examining how factors like advertising expenditure influence revenue. By collecting data on past advertising spending and corresponding sales figures analysts develop a regression model that tells the relationship between these variables. For instance if the analysis reveals that for every additional dollar spent on advertising sales increase by $10. This predictive capability enables companies to optimize their advertising strategies and allocate resources effectively.

### **2\. Multiple Linear Regression**

[Multiple linear regression](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/) involves more than one independent variable and one dependent variable. The equation for multiple linear regression is:

y=β0+β1X1+β2X2+………βnXny=\\beta\_{0}+\\beta\_{1}X1+\\beta\_{2}X2+………\\beta\_{n}Xny=β0​+β1​X1+β2​X2+………βn​Xn

where:

- Y is the dependent variable
- X1, X2, …, Xn are the independent variables
- β0 is the intercept
- β1, β2, …, βn are the slopes

#### The goal of the algorithm is to find the **best Fit Line** equation that can predict the values based on the independent variables.

In regression set of records are present with X and Y values and these values are used to learn a function so if you want to predict Y from an unknown X this learned function can be used. In regression we have to find the value of Y, So, a function is required that predicts continuous Y in the case of regression given X as independent features.

### Assumptions of Multiple Linear Regression

For Multiple Linear Regression, all four of the assumptions from Simple Linear Regression apply. In addition to this, below are few more:

1. **No multicollinearity**: There is no high correlation between the independent variables. This indicates that there is little or no correlation between the independent variables. Multicollinearity occurs when two or more independent variables are highly correlated with each other, which can make it difficult to determine the individual effect of each variable on the dependent variable. If there is multicollinearity, then multiple linear regression will not be an accurate model.
2. **Additivity:** The model assumes that the effect of changes in a predictor variable on the response variable is consistent regardless of the values of the other variables. This assumption implies that there is no interaction between variables in their effects on the dependent variable.
3. **Feature Selection:** In multiple linear regression, it is essential to carefully select the independent variables that will be included in the model. Including irrelevant or redundant variables may lead to overfitting and complicate the interpretation of the model.
4. **Overfitting:** Overfitting occurs when the model fits the training data too closely, capturing noise or random fluctuations that do not represent the true underlying relationship between variables. This can lead to poor generalization performance on new, unseen data.

Multiple linear regression sometimes faces issues like multicollinearity.

### **Multicollinearity**

[Multicollinearity](https://www.geeksforgeeks.org/multicollinearity-in-data/) is a statistical phenomenon where two or more independent variables in a multiple regression model are highly correlated, making it difficult to assess the individual effects of each variable on the dependent variable.

**Detecting Multicollinearity includes two techniques:**

- **Correlation Matrix:** Examining the correlation matrix among the independent variables is a common way to detect multicollinearity. High correlations (close to 1 or -1) indicate potential multicollinearity.
- **VIF (Variance Inflation Factor):** VIF is a measure that quantifies how much the variance of an estimated regression coefficient increases if your predictors are correlated. A high VIF (typically above 10) suggests multicollinearity.

### Use Case of Multiple Linear Regression

Multiple linear regression allows us to analyze relationship between multiple independent variables and a single dependent variable. Here are some use cases:

- **Real Estate Pricing:** In real estate MLR is used to predict property prices based on multiple factors such as location, size, number of bedrooms, etc. This helps buyers and sellers understand market trends and set competitive prices.
- **Financial Forecasting:** Financial analysts use MLR to predict stock prices or economic indicators based on multiple influencing factors such as interest rates, inflation rates and market trends. This enables better investment strategies and risk management24.
- **Agricultural Yield Prediction:** Farmers can use MLR to estimate crop yields based on several variables like rainfall, temperature, soil quality and fertilizer usage. This information helps in planning agricultural practices for optimal productivity
- **E-commerce Sales Analysis:** An e-commerce company can utilize MLR to assess how various factors such as product price, marketing promotions and seasonal trends impact sales.

Now that we have understood about linear regression, its assumption and its type now we will learn how to make a linear regression model.

## Cost function for Linear Regression

As we have discussed earlier about best fit line in linear regression, its not easy to get it easily in real life cases so we need to calculate errors that affects it. These errors need to be calculated to mitigate them. The difference between the predicted value Y^\\hat{Y}     Y^and the true value Y and it is called [cost function](https://www.geeksforgeeks.org/what-is-cost-function/) or the [loss function](https://www.geeksforgeeks.org/ml-common-loss-functions/).

In Linear Regression, the **Mean Squared Error (MSE)** cost function is employed, which calculates the average of the squared errors between the predicted values y^i\\hat{y}\_iy^​i​ and the actual values yi{y}\_iyi​. The purpose is to determine the optimal values for the intercept θ1\\theta\_1θ1​ and the coefficient of the input feature θ2\\theta\_2θ2​ providing the best-fit line for the given data points. The linear equation expressing this relationship is y^i=θ1+θ2xi\\hat{y}\_i = \\theta\_1 + \\theta\_2x\_iy^​i​=θ1​+θ2​xi​.

MSE function can be calculated as:

Cost function(J)=1n∑ni(yi^−yi)2\\text{Cost function}(J) = \\frac{1}{n}\\sum\_{n}^{i}(\\hat{y\_i}-y\_i)^2Cost function(J)=n1​∑ni​(yi​^​−yi​)2

Utilizing the MSE function, the iterative process of gradient descent is applied to update the values of \θ1&θ2\\theta\_1 \\& \\theta\_2 θ1​&θ2​. This ensures that the MSE value converges to the global minima, signifying the most accurate fit of the linear regression line to the dataset.

This process involves continuously adjusting the parameters \\(\\theta\_1\\) and \\(\\theta\_2\\) based on the gradients calculated from the MSE. The final result is a linear regression line that minimizes the overall squared differences between the predicted and actual values, providing an optimal representation of the underlying relationship in the data.

Now we have calculated loss function we need to optimize model to mtigate this error and it is done through gradient descent.

### **Gradient Descent for Linear Regression**

A linear regression model can be trained using the optimization algorithm [gradient descent](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/) by iteratively modifying the model’s parameters to reduce the [mean squared error (MSE)](https://www.geeksforgeeks.org/python-mean-squared-error/) of the model on a training dataset. To update θ1 and θ2 values in order to reduce the Cost function (minimizing RMSE value) and achieve the best-fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively update the values, reaching minimum cost.

A gradient is nothing but a derivative that defines the effects on outputs of the function with a little bit of variation in inputs.

Let’s differentiate the cost function(J) with respect to θ1\\theta\_1     θ1​

J’θ1=∂J(θ1,θ2)∂θ1=∂∂θ1\[1n(∑i=1n(y^i−yi)2)\]=1n\[∑i=1n2(y^i−yi)(∂∂θ1(y^i−yi))\]=1n\[∑i=1n2(y^i−yi)(∂∂θ1(θ1+θ2xi−yi))\]=1n\[∑i=1n2(y^i−yi)(1+0−0)\]=1n\[∑i=1n(y^i−yi)(2)\]=2n∑i=1n(y^i−yi)\\begin {aligned} {J}’\_{\\theta\_1} &=\\frac{\\partial J(\\theta\_1,\\theta\_2)}{\\partial \\theta\_1} \\\ &= \\frac{\\partial}{\\partial \\theta\_1} \\left\[\\frac{1}{n} \\left(\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i)^2 \\right )\\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(\\frac{\\partial}{\\partial \\theta\_1}(\\hat{y}\_i-y\_i) \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(\\frac{\\partial}{\\partial \\theta\_1}( \\theta\_1 + \\theta\_2x\_i-y\_i) \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(1+0-0 \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i) \\left(2 \\right ) \\right\] \\\ &= \\frac{2}{n}\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i) \\end {aligned}J’θ1​​​=∂θ1​∂J(θ1​,θ2​)​=∂θ1​∂​\[n1​(i=1∑n​(y^​i​−yi​)2)\]=n1​\[i=1∑n​2(y^​i​−yi​)(∂θ1​∂​(y^​i​−yi​))\]=n1​\[i=1∑n​2(y^​i​−yi​)(∂θ1​∂​(θ1​+θ2​xi​−yi​))\]=n1​\[i=1∑n​2(y^​i​−yi​)(1+0−0)\]=n1​\[i=1∑n​(y^​i​−yi​)(2)\]=n2​i=1∑n​(y^​i​−yi​)​

Let’s differentiate the cost function(J) with respect to θ2\\theta\_2θ2​

J’θ2=∂J(θ1,θ2)∂θ2=∂∂θ2\[1n(∑i=1n(y^i−yi)2)\]=1n\[∑i=1n2(y^i−yi)(∂∂θ2(y^i−yi))\]=1n\[∑i=1n2(y^i−yi)(∂∂θ2(θ1+θ2xi−yi))\]=1n\[∑i=1n2(y^i−yi)(0+xi−0)\]=1n\[∑i=1n(y^i−yi)(2xi)\]=2n∑i=1n(y^i−yi)⋅xi\\begin {aligned} {J}’\_{\\theta\_2} &=\\frac{\\partial J(\\theta\_1,\\theta\_2)}{\\partial \\theta\_2} \\\ &= \\frac{\\partial}{\\partial \\theta\_2} \\left\[\\frac{1}{n} \\left(\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i)^2 \\right )\\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(\\frac{\\partial}{\\partial \\theta\_2}(\\hat{y}\_i-y\_i) \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(\\frac{\\partial}{\\partial \\theta\_2}( \\theta\_1 + \\theta\_2x\_i-y\_i) \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}2(\\hat{y}\_i-y\_i) \\left(0+x\_i-0 \\right ) \\right\] \\\ &= \\frac{1}{n}\\left\[\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i) \\left(2x\_i \\right ) \\right\] \\\ &= \\frac{2}{n}\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i)\\cdot x\_i \\end {aligned}J’θ2​​​=∂θ2​∂J(θ1​,θ2​)​=∂θ2​∂​\[n1​(i=1∑n​(y^​i​−yi​)2)\]=n1​\[i=1∑n​2(y^​i​−yi​)(∂θ2​∂​(y^​i​−yi​))\]=n1​\[i=1∑n​2(y^​i​−yi​)(∂θ2​∂​(θ1​+θ2​xi​−yi​))\]=n1​\[i=1∑n​2(y^​i​−yi​)(0+xi​−0)\]=n1​\[i=1∑n​(y^​i​−yi​)(2xi​)\]=n2​i=1∑n​(y^​i​−yi​)⋅xi​​

Finding the coefficients of a linear equation that best fits the training data is the objective of linear regression. By moving in the direction of the Mean Squared Error negative gradient with respect to the coefficients, the coefficients can be changed. And the respective intercept and coefficient of X will be if α\\alpha     α is the learning rate.

![Gradient Descent -Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230424151248/Gradient-Descent-for-ML-Linear-Regression-(1).webp)

Gradient Descent

θ1=θ1–α(J’θ1)=θ1−α(2n∑i=1n(y^i−yi))θ2=θ2–α(J’θ2)=θ2–α(2n∑i=1n(y^i−yi)⋅xi)\\begin{aligned} \\theta\_1 &= \\theta\_1 – \\alpha \\left( {J}’\_{\\theta\_1}\\right) \\\&=\\theta\_1 -\\alpha \\left( \\frac{2}{n}\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i)\\right) \\end{aligned} \\\ \\begin{aligned} \\theta\_2 &= \\theta\_2 – \\alpha \\left({J}’\_{\\theta\_2}\\right) \\\&=\\theta\_2 – \\alpha \\left(\\frac{2}{n}\\sum\_{i=1}^{n}(\\hat{y}\_i-y\_i)\\cdot x\_i\\right) \\end{aligned}θ1​​=θ1​–α(J’θ1​​)=θ1​−α(n2​i=1∑n​(y^​i​−yi​))​θ2​​=θ2​–α(J’θ2​​)=θ2​–α(n2​i=1∑n​(y^​i​−yi​)⋅xi​)​

After optimizing our model, we evaluate our models accuracy to see how well it will perform in real world scenario.

## Evaluation Metrics for Linear Regression

A variety of [evaluation measures](https://www.geeksforgeeks.org/metrics-for-machine-learning-model/) can be used to determine the strength of any linear regression model. These assessment metrics often give an indication of how well the model is producing the observed outputs.

The most common measurements are:

### Mean Square Error (MSE)

[Mean Squared Error (MSE)](https://www.geeksforgeeks.org/python-mean-squared-error/) is an evaluation metric that calculates the average of the squared differences between the actual and predicted values for all the data points. The difference is squared to ensure that negative and positive differences don’t cancel each other out.

MSE=1n∑i=1n(yi–yi^)2MSE = \\frac{1}{n}\\sum\_{i=1}^{n}\\left ( y\_i – \\widehat{y\_{i}} \\right )^2MSE=n1​∑i=1n​(yi​–yi​​)2

Here,

- n is the number of data points.
- yi is the actual or observed value for the ith data point.
- yi^\\widehat{y\_{i}}


yi​​ is the predicted value for the ith data point.

MSE is a way to quantify the accuracy of a model’s predictions. MSE is sensitive to outliers as large errors contribute significantly to the overall score.

### Mean Absolute Error (MAE)

[Mean Absolute Error](https://www.geeksforgeeks.org/how-to-calculate-mean-absolute-error-in-python/) is an evaluation metric used to calculate the accuracy of a regression model. MAE measures the average absolute difference between the predicted values and actual values.

Mathematically, MAE is expressed as:

MAE=1n∑i=1n∣Yi–Yi^∣MAE =\\frac{1}{n} \\sum\_{i=1}^{n}\|Y\_i – \\widehat{Y\_i}\|MAE=n1​∑i=1n​∣Yi​–Yi​​∣

Here,

- n is the number of observations
- Yi represents the actual values.
- Yi^\\widehat{Y\_i}

Yi​​ represents the predicted values

Lower MAE value indicates better model performance. It is not sensitive to the outliers as we consider absolute differences.

### **Root Mean Squared Error (RMSE)**

The square root of the residuals’ variance is the [Root Mean Squared Error](https://www.geeksforgeeks.org/root-mean-square-error-in-r-programming/). It describes how well the observed data points match the expected values, or the model’s absolute fit to the data.

In mathematical notation, it can be expressed as:

RMSE=RSSn=∑i=2n(yiactual−yipredicted)2nRMSE=\\sqrt{\\frac{RSS}{n}}=\\sqrt\\frac{{{\\sum\_{i=2}^{n}(y^{actual}\_{i}}- y\_{i}^{predicted})^2}}{n}RMSE=nRSS​​=n∑i=2n​(yiactual​−yipredicted​)2​​

Rather than dividing the entire number of data points in the model by the number of degrees of freedom, one must divide the sum of the squared residuals to obtain an unbiased estimate. Then, this figure is referred to as the Residual Standard Error (RSE).

In mathematical notation, it can be expressed as:

RMSE=RSSn=∑i=2n(yiactual−yipredicted)2(n−2)RMSE=\\sqrt{\\frac{RSS}{n}}=\\sqrt\\frac{{{\\sum\_{i=2}^{n}(y^{actual}\_{i}}- y\_{i}^{predicted})^2}}{(n-2)}RMSE=nRSS​​=(n−2)∑i=2n​(yiactual​−yipredicted​)2​​

RSME is not as good of a metric as R-squared. Root Mean Squared Error can fluctuate when the units of the variables vary since its value is dependent on the variables’ units (it is not a normalized measure).

### Coefficient of Determination (R-squared)

[R-Squared](https://www.geeksforgeeks.org/r-squared/) is a statistic that indicates how much variation the developed model can explain or capture. It is always in the range of 0 to 1. In general, the better the model matches the data, the greater the R-squared number.

In mathematical notation, it can be expressed as:

R2=1−(RSSTSS)R^{2}=1-(^{\\frac{RSS}{TSS}})R2=1−(TSSRSS​)

- [**Residual sum of Squares**](https://www.geeksforgeeks.org/residual-sum-of-squares/#:~:text=Residual%20sum%20of%20squares%20is%20used%20to%20calculate%20the%20variance,squares%2C%20the%20better%20the%20model.) **(RSS): The** sum of squares of the residual for each data point in the plot or data is known as the residual sum of squares, or RSS. It is a measurement of the difference between the output that was observed and what was anticipated.

RSS=∑i=1n(yi−b0−b1xi)2RSS=\\sum\_{i=1}^{n}(y\_{i}-b\_{0}-b\_{1}x\_{i})^{2}














RSS=∑i=1n​(yi​−b0​−b1​xi​)2
- **Total Sum of Squares (TSS):** The sum of the data points’ errors from the answer variable’s mean is known as the total sum of squares, or TSS.

TSS=∑i=1n(y−yi‾)2TSS=\\sum\_{i=1}^{n}(y-\\overline{y\_{i}})^2














TSS=∑i=1n​(y−yi​​)2

R squared metric is a measure of the proportion of variance in the dependent variable that is explained the independent variables in the model.

### Adjusted R-Squared Error

Adjusted R2 measures the proportion of variance in the dependent variable that is explained by independent variables in a regression model. [Adjusted R-square](https://www.geeksforgeeks.org/ml-adjusted-r-square-in-regression-analysis/) accounts the number of predictors in the model and penalizes the model for including irrelevant predictors that don’t contribute significantly to explain the variance in the dependent variables.

Mathematically, adjusted R2 is expressed as:

AdjustedR2=1–((1−R2).(n−1)n−k−1)Adjusted \\, R^2 = 1 – (\\frac{(1-R^2).(n-1)}{n-k-1})AdjustedR2=1–(n−k−1(1−R2).(n−1)​)

Here,

- n is the number of observations
- k is the number of predictors in the model
- R2  is coeeficient of determination

Adjusted R-square helps to prevent overfitting. It penalizes the model with additional predictors that do not contribute significantly to explain the variance in the dependent variable.

While evaluation metrics help us measure the performance of a model, regularization helps in improving that performance by addressing overfitting and enhancing generalization.

## Regularization Techniques for Linear Models

### Lasso Regression (L1 Regularization)

[Lasso Regression](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/) is a technique used for regularizing a linear regression model, it adds a penalty term to the linear regression objective function to prevent [overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/).

The objective function after applying lasso regression is:

J(θ)=12m∑i=1m(yi^–yi)2+λ∑j=1n∣θj∣J(\\theta) = \\frac{1}{2m} \\sum\_{i=1}^{m}(\\widehat{y\_i} – y\_i) ^2+ \\lambda \\sum\_{j=1}^{n}\|\\theta\_j\|J(θ)=2m1​∑i=1m​(yi​​–yi​)2+λ∑j=1n​∣θj​∣

- the first term is the least squares loss, representing the squared difference between predicted and actual values.
- the second term is the L1 regularization term, it penalizes the sum of absolute values of the regression coefficient θj.

### Ridge Regression (L2 Regularization)

[Ridge regression](https://www.geeksforgeeks.org/implementation-of-ridge-regression-from-scratch-using-python/) is a linear regression technique that adds a regularization term to the standard linear objective. Again, the goal is to prevent overfitting by penalizing large coefficient in linear regression equation. It useful when the dataset has [multicollinearity](https://www.geeksforgeeks.org/test-of-multicollinearity/) where predictor variables are highly correlated.

The objective function after applying ridge regression is:

J(θ)=12m∑i=1m(yi^–yi)2+λ∑j=1nθj2J(\\theta) = \\frac{1}{2m} \\sum\_{i=1}^{m}(\\widehat{y\_i} – y\_i)^2 + \\lambda \\sum\_{j=1}^{n}\\theta\_{j}^{2}J(θ)=2m1​∑i=1m​(yi​​–yi​)2+λ∑j=1n​θj2​

- the first term is the least squares loss, representing the squared difference between predicted and actual values.
- the second term is the L1 regularization term, it penalizes the sum of square of values of the regression coefficient θj.

### Elastic Net Regression

[Elastic Net Regression](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/) is a hybrid regularization technique that combines the power of both L1 and L2 regularization in linear regression objective.

J(θ)=12m∑i=1m(yi^–yi)2+αλ∑j=1n∣θj∣+12(1−α)λ∑j=1nθj2J(\\theta) = \\frac{1}{2m} \\sum\_{i=1}^{m}(\\widehat{y\_i} – y\_i)^2 + \\alpha \\lambda \\sum\_{j=1}^{n}{\|\\theta\_j\|} + \\frac{1}{2}(1- \\alpha) \\lambda \\sum\_{j=1}{n} \\theta\_{j}^{2}J(θ)=2m1​∑i=1m​(yi​​–yi​)2+αλ∑j=1n​∣θj​∣+21​(1−α)λ∑j=1​nθj2​

- the first term is least square loss.
- the second term is L1 regularization and third is ridge regression.
- λ is the overall regularization strength.
- α controls the mix between L1 and L2 regularization.

Now that we have learned how to make a linear regression model, now we will implement it.

## Python Implementation of Linear Regression

#### Import the necessary libraries:

Python`
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation
`

#### Load the dataset and separate input and Target variables

Here is the link for dataset: [Dataset Link](https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv)

Python`
url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
data = pd.read_csv(url)
data
# Drop the missing values
data = data.dropna()
# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500, 1)
train_output = np.array(data.y[0:500]).reshape(500, 1)
# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)
`

#### Build the Linear Regression Model and Plot the regression line

#### Steps:

- In forward propagation Linear regression function Y=mx+c is applied by initially assigning random value of parameter (m & c).
- The we have written the function to finding the cost function i.e the mean

Python`
class LinearRegression:
    def __init__(self):
        self.parameters = {}
    def forward_propagation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c
        return predictions
    def cost_function(self, predictions, train_output):
        cost = np.mean((train_output - predictions) ** 2)
        return cost
    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = (predictions-train_output)
        # dm= 2/n * mean of (predictions-actual) * input
        dm = 2 * np.mean(np.multiply(train_input, df))
        # dc = 2/n * mean of (predictions-actual)
        dc = 2 * np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']
    def train(self, train_input, train_output, learning_rate, iters):
        # Initialize random parameters
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1
        # Initialize loss
        self.loss = []
        # Initialize figure and axis for animation
        fig, ax = plt.subplots()
        x_vals = np.linspace(min(train_input), max(train_input), 100)
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals +
                        self.parameters['c'], color='red', label='Regression Line')
        ax.scatter(train_input, train_output, marker='o',
                color='green', label='Training Data')
        # Set y-axis limits to exclude negative values
        ax.set_ylim(0, max(train_output) + 1)
        def update(frame):
            # Forward propagation
            predictions = self.forward_propagation(train_input)
            # Cost function
            cost = self.cost_function(predictions, train_output)
            # Back propagation
            derivatives = self.backward_propagation(
                train_input, train_output, predictions)
            # Update parameters
            self.update_parameters(derivatives, learning_rate)
            # Update the regression line
            line.set_ydata(self.parameters['m']
                        * x_vals + self.parameters['c'])
            # Append loss and print
            self.loss.append(cost)
            print("Iteration = {}, Loss = {}".format(frame + 1, cost))
            return line,
        # Create animation
        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
        # Save the animation as a video file (e.g., MP4)
        ani.save('linear_regression_A.gif', writer='ffmpeg')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
        return self.parameters, self.loss
`

#### Trained the model and Final Prediction

Python`
#Example usage
linear_reg = LinearRegression()
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)
`

**Output**:

```
Iteration = 1, Loss = 9130.407560462196
Iteration = 1, Loss = 1107.1996742908998
Iteration = 1, Loss = 140.31580932842422
Iteration = 1, Loss = 23.795780526084116
Iteration = 2, Loss = 9.753848205147605
Iteration = 3, Loss = 8.061641745006835
Iteration = 4, Loss = 7.8577116490914864
Iteration = 5, Loss = 7.8331350515579015
Iteration = 6, Loss = 7.830172502503967
Iteration = 7, Loss = 7.829814681591015
Iteration = 8, Loss = 7.829770758846183
Iteration = 9, Loss = 7.829764664327399
Iteration = 10, Loss = 7.829763128602258
Iteration = 11, Loss = 7.829762142342088
Iteration = 12, Loss = 7.829761222379141
Iteration = 13, Loss = 7.829760310486438
Iteration = 14, Loss = 7.829759399646989
Iteration = 15, Loss = 7.829758489015161
Iteration = 16, Loss = 7.829757578489033
Iteration = 17, Loss = 7.829756668056319
Iteration = 18, Loss = 7.829755757715535
Iteration = 19, Loss = 7.829754847466484
Iteration = 20, Loss = 7.829753937309139

```

![ezgif-1-ba0c9540c5](https://media.geeksforgeeks.org/wp-content/uploads/20231123123151/ezgif-1-ba0c9540c5.gif)

The linear regression line provides valuable insights into the relationship between the two variables. It represents the best-fitting line that captures the overall trend of how a dependent variable (Y) changes in response to variations in an independent variable (X).

- **Positive Linear Regression Line**: A positive linear regression line indicates a direct relationship between the independent variable (X) and the dependent variable (Y). This means that as the value of X increases, the value of Y also increases. The slope of a positive linear regression line is positive, meaning that the line slants upward from left to right.
- **Negative Linear Regression Line**: A negative linear regression line indicates an inverse relationship between the independent variable (X) and the dependent variable (Y). This means that as the value of X increases, the value of Y decreases. The slope of a negative linear regression line is negative, meaning that the line slants downward from left to right.

## Applications of Linear Regression

Linear regression is used in many different fields including finance, economics and psychology to understand and predict the behavior of a particular variable.

For example linear regression is widely used in finance to analyze relationships and make predictions. It can model how a company’s earnings per share (EPS) influence its stock price. If the model shows that a $1 increase in EPS results in a $15 rise in stock price, investors gain insights into the company’s valuation. Similarly, linear regression can forecast currency values by analyzing historical exchange rates and economic indicators, helping financial professionals make informed decisions and manage risks effectively.

> Also read – [Linear Regression – In Simple Words, with real-life Examples](https://www.geeksforgeeks.org/linear-regression-real-life-examples/)

## Advantages and Disadvantages of Linear Regression

### Advantages of Linear Regression

- Linear regression is a relatively simple algorithm, making it easy to understand and implement. The coefficients of the linear regression model can be interpreted as the change in the dependent variable for a one-unit change in the independent variable, providing insights into the relationships between variables.
- Linear regression is computationally efficient and can handle large datasets effectively. It can be trained quickly on large datasets, making it suitable for real-time applications.
- Linear regression is relatively robust to outliers compared to other machine learning algorithms. Outliers may have a smaller impact on the overall model performance.
- Linear regression often serves as a good baseline model for comparison with more complex machine learning algorithms.
- Linear regression is a well-established algorithm with a rich history and is widely available in various machine learning libraries and software packages.

### Disadvantages of Linear Regression

- Linear regression assumes a linear relationship between the dependent and independent variables. If the relationship is not linear, the model may not perform well.
- Linear regression is sensitive to multicollinearity, which occurs when there is a high correlation between independent variables. Multicollinearity can inflate the variance of the coefficients and lead to unstable model predictions.
- Linear regression assumes that the features are already in a suitable form for the model. Feature engineering may be required to transform features into a format that can be effectively used by the model.
- Linear regression is susceptible to both overfitting and underfitting. Overfitting occurs when the model learns the training data too well and fails to generalize to unseen data. Underfitting occurs when the model is too simple to capture the underlying relationships in the data.
- Linear regression provides limited explanatory power for complex relationships between variables. More advanced machine learning techniques may be necessary for deeper insights.

## Conclusion

Linear regression is a fundamental machine learning algorithm that has been widely used for many years due to its simplicity, interpretability, and efficiency. It is a valuable tool for understanding relationships between variables and making predictions in a variety of applications.

However, it is important to be aware of its limitations, such as its assumption of linearity and sensitivity to multicollinearity. When these limitations are carefully considered, linear regression can be a powerful tool for data analysis and prediction.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)

[Ordinary Least Squares (OLS) using statsmodels](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-Regression](https://www.geeksforgeeks.org/tag/ml-regression/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Maths for Machine Learning\\
\\
\\
Mathematics is the foundation of machine learning. Math concepts plays a crucial role in understanding how models learn from data and optimizing their performance. Before diving into machine learning algorithms, it's important to familiarize yourself with foundational topics, like Statistics, Probab\\
\\
5 min read](https://www.geeksforgeeks.org/machine-learning-mathematics/)

## Linear Algebra and Matrix

- [Matrices\\
\\
\\
Matrices are key concepts in mathematics, widely used in solving equations and problems in fields like physics and computer science. A matrix is simply a grid of numbers, and a determinant is a value calculated from a square matrix. Example: \[Tex\]\\begin{bmatrix} 6 & 9 \\\ 5 & -4 \\\ \\end{bmatr\\
\\
3 min read](https://www.geeksforgeeks.org/matrices/)

* * *

- [Scalar and Vector\\
\\
\\
Scalar and Vector Quantities are used to describe the motion of an object. Scalar Quantities are defined as physical quantities that have magnitude or size only. For example, distance, speed, mass, density, etc. However, vector quantities are those physical quantities that have both magnitude and di\\
\\
8 min read](https://www.geeksforgeeks.org/scalars-and-vectors/)

* * *

- [Python Program to Add Two Matrices\\
\\
\\
The task of adding two matrices in Python involves combining corresponding elements from two given matrices to produce a new matrix. Each element in the resulting matrix is obtained by adding the values at the same position in the input matrices. For example, if two 2x2 matrices are given as: The su\\
\\
3 min read](https://www.geeksforgeeks.org/python-program-add-two-matrices/)

* * *

- [Python Program to Multiply Two Matrices\\
\\
\\
Given two matrices, we will have to create a program to multiply two matrices in Python. Example: Python Matrix Multiplication of Two-Dimension \[GFGTABS\] Python matrix\_a = \[\[1, 2\], \[3, 4\]\] matrix\_b = \[\[5, 6\], \[7, 8\]\] result = \[\[0, 0\], \[0, 0\]\] for i in range(2): for j in range(2): result\[i\]\[j\] = (mat\\
\\
5 min read](https://www.geeksforgeeks.org/python-program-multiply-two-matrices/)

* * *

- [Vector Operations\\
\\
\\
Vectors are fundamental quantities in physics and mathematics, that have both magnitude and direction. So performing mathematical operations on them directly is not possible. So we have special operations that work only with vector quantities and hence the name, vector operations. Thus, It is essent\\
\\
8 min read](https://www.geeksforgeeks.org/vector-operations/)

* * *

- [Product of Vectors\\
\\
\\
Vector operations are used almost everywhere in the field of physics. Many times these operations include addition, subtraction, and multiplication. Addition and subtraction can be performed using the triangle law of vector addition. In the case of products, vector multiplication can be done in two\\
\\
6 min read](https://www.geeksforgeeks.org/product-of-vectors/)

* * *

- [Scalar Product of Vectors\\
\\
\\
Two vectors or a vector and a scalar can be multiplied. There are mainly two kinds of products of vectors in physics, scalar multiplication of vectors and Vector Product (Cross Product) of two vectors. The result of the scalar product of two vectors is a number (a scalar). The common use of the scal\\
\\
9 min read](https://www.geeksforgeeks.org/scalar-product-of-vectors/)

* * *

- [Dot and Cross Products on Vectors\\
\\
\\
A quantity that is characterized not only by magnitude but also by its direction, is called a vector. Velocity, force, acceleration, momentum, etc, are vectors. Vectors can be multiplied in two ways: Scalar product or Dot productVector Product or Cross productTable of Content Scalar Product/Dot Prod\\
\\
9 min read](https://www.geeksforgeeks.org/dot-and-cross-products-on-vectors/)

* * *

- [Transpose a matrix in Single line in Python\\
\\
\\
Transpose of a matrix is a task we all can perform very easily in Python (Using a nested loop). But there are some interesting ways to do the same in a single line. In Python, we can implement a matrix as a nested list (a list inside a list). Each element is treated as a row of the matrix. For examp\\
\\
4 min read](https://www.geeksforgeeks.org/transpose-matrix-single-line-python/)

* * *

- [Transpose of a Matrix\\
\\
\\
A Matrix is a rectangular arrangement of numbers (or elements) in rows and columns. It is often used in mathematics to represent data, solve systems of equations, or perform transformations. A matrix is written as: \[Tex\]A = \\begin{bmatrix} 1 & 2 & 3\\\ 4 & 5 & 6 \\\ 7 & 8 & 9\\e\\
\\
12 min read](https://www.geeksforgeeks.org/transpose-of-a-matrix/)

* * *

- [Adjoint and Inverse of a Matrix\\
\\
\\
Given a square matrix, find the adjoint and inverse of the matrix. We strongly recommend you to refer determinant of matrix as a prerequisite for this. Adjoint (or Adjugate) of a matrix is the matrix obtained by taking the transpose of the cofactor matrix of a given square matrix is called its Adjoi\\
\\
15+ min read](https://www.geeksforgeeks.org/adjoint-inverse-matrix/)

* * *

- [How to inverse a matrix using NumPy\\
\\
\\
In this article, we will see NumPy Inverse Matrix in Python before that we will try to understand the concept of it. The inverse of a matrix is just a reciprocal of the matrix as we do in normal arithmetic for a single number which is used to solve the equations to find the value of unknown variable\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-inverse-a-matrix-using-numpy/)

* * *

- [Program to find Determinant of a Matrix\\
\\
\\
The determinant of a Matrix is defined as a special number that is defined only for square matrices (matrices that have the same number of rows and columns). A determinant is used in many places in calculus and other matrices related to algebra, it actually represents the matrix in terms of a real n\\
\\
15+ min read](https://www.geeksforgeeks.org/determinant-of-a-matrix/)

* * *

- [Program to find Normal and Trace of a matrix\\
\\
\\
Given a 2D matrix, the task is to find Trace and Normal of matrix.Normal of a matrix is defined as square root of sum of squares of matrix elements.Trace of a n x n square matrix is sum of diagonal elements. Examples : Input : mat\[\]\[\] = {{7, 8, 9}, {6, 1, 2}, {5, 4, 3}}; Output : Normal = 16 Trace =\\
\\
6 min read](https://www.geeksforgeeks.org/program-find-normal-trace-matrix/)

* * *

- [Data Science \| Solving Linear Equations\\
\\
\\
Linear Algebra is a very fundamental part of Data Science. When one talks about Data Science, data representation becomes an important aspect of Data Science. Data is represented usually in a matrix form. The second important thing in the perspective of Data Science is if this data contains several\\
\\
9 min read](https://www.geeksforgeeks.org/data-science-solving-linear-equations/)

* * *

- [Data Science - Solving Linear Equations with Python\\
\\
\\
A collection of equations with linear relationships between the variables is known as a system of linear equations. The objective is to identify the values of the variables that concurrently satisfy each equation, each of which is a linear constraint. By figuring out the system, we can learn how the\\
\\
4 min read](https://www.geeksforgeeks.org/data-science-solving-linear-equations-2/)

* * *

- [System of Linear Equations\\
\\
\\
In mathematics, a system of linear equations consists of two or more linear equations that share the same variables. These systems often arise in real-world applications, such as engineering, physics, economics, and more, where relationships between variables need to be analyzed. Understanding how t\\
\\
8 min read](https://www.geeksforgeeks.org/system-linear-equations/)

* * *

- [System of Linear Equations in three variables using Cramer's Rule\\
\\
\\
Cramer's rule: In linear algebra, Cramer's rule is an explicit formula for the solution of a system of linear equations with as many equations as unknown variables. It expresses the solution in terms of the determinants of the coefficient matrix and of matrices obtained from it by replacing one colu\\
\\
12 min read](https://www.geeksforgeeks.org/system-linear-equations-three-variables-using-cramers-rule/)

* * *

- [Eigenvalues and Eigenvectors\\
\\
\\
Eigenvectors are the directions that remain unchanged during a transformation, even if they get longer or shorter. Eigenvalues are the numbers that indicate how much something stretches or shrinks during that transformation. These ideas are important in many areas of math and engineering, including\\
\\
15+ min read](https://www.geeksforgeeks.org/eigen-values/)

* * *

- [Applications of Eigenvalues and Eigenvectors\\
\\
\\
Eigenvalues and eigenvectors play a crucial role in a wide range of applications across engineering and science. Fields like control theory, vibration analysis, electric circuits, advanced dynamics, and quantum mechanics frequently rely on these concepts. One key application involves transforming ma\\
\\
7 min read](https://www.geeksforgeeks.org/applications-of-eigenvalues-and-eigenvectors/)

* * *

- [How to compute the eigenvalues and right eigenvectors of a given square array using NumPY?\\
\\
\\
In this article, we will discuss how to compute the eigenvalues and right eigenvectors of a given square array using NumPy library. Example: Suppose we have a matrix as: \[\[1,2\], \[2,3\]\] Eigenvalue we get from this matrix or square array is: \[-0.23606798 4.23606798\] Eigenvectors of this matrix are: \[\[\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-compute-the-eigenvalues-and-right-eigenvectors-of-a-given-square-array-using-numpy/)\
\
* * *\
\
\
## Statistics for Machine Learning\
\
- [Descriptive Statistic\\
\\
\\
Statistics serves as the backbone of data science providing tools and methodologies to extract meaningful insights from raw data. Data scientists rely on statistics for every crucial task - from cleaning messy datasets and creating powerful visualizations to building predictive models that glimpse i\\
\\
5 min read](https://www.geeksforgeeks.org/descriptive-statistic/)\
\
* * *\
\
- [Measures of Central Tendency\\
\\
\\
Usually, frequency distribution and graphical representation are used to depict a set of raw data to attain meaningful conclusions from them. However, sometimes, these methods fail to convey a proper and clear picture of the data as expected. Therefore, some measures, also known as Measures of Centr\\
\\
5 min read](https://www.geeksforgeeks.org/measures-of-central-tendency-2/)\
\
* * *\
\
- [Measures of Dispersion \| Types, Formula and Examples\\
\\
\\
Measures of Dispersion are used to represent the scattering of data. These are the numbers that show the various aspects of the data spread across various parameters. Let's learn about the measure of dispersion in statistics, its types, formulas, and examples in detail. Dispersion in StatisticsDispe\\
\\
10 min read](https://www.geeksforgeeks.org/measures-of-dispersion/)\
\
* * *\
\
- [Mean, Variance and Standard Deviation\\
\\
\\
Mean, Variance and Standard Deviation are fundamental concepts in statistics and engineering mathematics, essential for analyzing and interpreting data. These measures provide insights into data's central tendency, dispersion, and spread, which are crucial for making informed decisions in various en\\
\\
10 min read](https://www.geeksforgeeks.org/mathematics-mean-variance-and-standard-deviation/)\
\
* * *\
\
- [Calculate the average, variance and standard deviation in Python using NumPy\\
\\
\\
Numpy in Python is a general-purpose array-processing package. It provides a high-performance multidimensional array object and tools for working with these arrays. It is the fundamental package for scientific computing with Python. Numpy provides very easy methods to calculate the average, variance\\
\\
5 min read](https://www.geeksforgeeks.org/calculate-the-average-variance-and-standard-deviation-in-python-using-numpy/)\
\
* * *\
\
- [Random Variable\\
\\
\\
Random variable is a fundamental concept in statistics that bridges the gap between theoretical probability and real-world data. A Random variable in statistics is a function that assigns a real value to an outcome in the sample space of a random experiment. For example: if you roll a die, you can a\\
\\
11 min read](https://www.geeksforgeeks.org/random-variable/)\
\
* * *\
\
- [Difference between Parametric and Non-Parametric Methods\\
\\
\\
Statistical analysis plays a crucial role in understanding and interpreting data across various disciplines. Two prominent approaches in statistical analysis are Parametric and Non-Parametric Methods. While both aim to draw inferences from data, they differ in their assumptions and underlying princi\\
\\
8 min read](https://www.geeksforgeeks.org/difference-between-parametric-and-non-parametric-methods/)\
\
* * *\
\
- [Probability Distribution - Function, Formula, Table\\
\\
\\
A probability distribution describes how the probabilities of different outcomes are assigned to the possible values of a random variable. It provides a way of modeling the likelihood of each outcome in a random experiment. While a frequency distribution shows how often outcomes occur in a sample or\\
\\
15+ min read](https://www.geeksforgeeks.org/probability-distribution/)\
\
* * *\
\
- [Confidence Interval\\
\\
\\
Confidence Interval (CI) is a range of values that estimates where the true population value is likely to fall. Instead of just saying The average height of students is 165 cm a confidence interval allow us to say We are 95% confident that the true average height is between 160 cm and 170 cm. Before\\
\\
9 min read](https://www.geeksforgeeks.org/confidence-interval/)\
\
* * *\
\
- [Covariance and Correlation\\
\\
\\
Covariance and correlation are the two key concepts in Statistics that help us analyze the relationship between two variables. Covariance measures how two variables change together, indicating whether they move in the same or opposite directions. In this article, we will learn about the differences\\
\\
5 min read](https://www.geeksforgeeks.org/mathematics-covariance-and-correlation/)\
\
* * *\
\
- [Program to Find Correlation Coefficient\\
\\
\\
The correlation coefficient is a statistical measure that helps determine the strength and direction of the relationship between two variables. It quantifies how changes in one variable correspond to changes in another. This coefficient, sometimes referred to as the cross-correlation coefficient, al\\
\\
9 min read](https://www.geeksforgeeks.org/program-find-correlation-coefficient/)\
\
* * *\
\
- [Robust Correlation\\
\\
\\
Correlation is a statistical tool that is used to analyze and measure the degree of relationship or degree of association between two or more variables. There are generally three types of correlation: Positive correlation: When we increase the value of one variable, the value of another variable inc\\
\\
8 min read](https://www.geeksforgeeks.org/robust-correlation/)\
\
* * *\
\
- [Normal Probability Plot\\
\\
\\
The probability plot is a way of visually comparing the data coming from different distributions. These data can be of empirical dataset or theoretical dataset. The probability plot can be of two types: P-P plot: The (Probability-to-Probability) p-p plot is the way to visualize the comparing of cumu\\
\\
3 min read](https://www.geeksforgeeks.org/normal-probability-plot/)\
\
* * *\
\
- [Quantile Quantile plots\\
\\
\\
The quantile-quantile( q-q plot) plot is a graphical method for determining if a dataset follows a certain probability distribution or whether two samples of data came from the same population or not. Q-Q plots are particularly useful for assessing whether a dataset is normally distributed or if it\\
\\
8 min read](https://www.geeksforgeeks.org/quantile-quantile-plots/)\
\
* * *\
\
- [True Error vs Sample Error\\
\\
\\
True Error The true error can be said as the probability that the hypothesis will misclassify a single randomly drawn sample from the population. Here the population represents all the data in the world. Let's consider a hypothesis h(x) and the true/target function is f(x) of population P. The proba\\
\\
3 min read](https://www.geeksforgeeks.org/true-error-vs-sample-error/)\
\
* * *\
\
- [Bias-Variance Trade Off - Machine Learning\\
\\
\\
It is important to understand prediction errors (bias and variance) when it comes to accuracy in any machine-learning algorithm. There is a tradeoff between a modelâ€™s ability to minimize bias and variance which is referred to as the best solution for selecting a value of Regularization constant. A p\\
\\
3 min read](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/)\
\
* * *\
\
- [Understanding Hypothesis Testing\\
\\
\\
Hypothesis method compares two opposite statements about a population and uses sample data to decide which one is more likely to be correct.To test this assumption we first take a sample from the population and analyze it and use the results of the analysis to decide if the claim is valid or not. Su\\
\\
14 min read](https://www.geeksforgeeks.org/understanding-hypothesis-testing/)\
\
* * *\
\
- [T-test\\
\\
\\
After learning about the Z-test we now move on to another important statistical test called the t-test. While the Z-test is useful when we know the population variance. The t-test is used to compare the averages of two groups to see if they are significantly different from each other. Suppose You wa\\
\\
11 min read](https://www.geeksforgeeks.org/t-test/)\
\
* * *\
\
- [Paired T-Test - A Detailed Overview\\
\\
\\
Studentâ€™s t-test or t-test is the statistical method used to determine if there is a difference between the means of two samples. The test is often performed to find out if there is any sampling error or unlikeliness in the experiment. This t-test is further divided into 3 types based on your data a\\
\\
5 min read](https://www.geeksforgeeks.org/paired-t-test-a-detailed-overview/)\
\
* * *\
\
- [P-value in Machine Learning\\
\\
\\
P-value helps us determine how likely it is to get a particular result when the null hypothesis is assumed to be true. It is the probability of getting a sample like ours or more extreme than ours if the null hypothesis is correct. Therefore, if the null hypothesis is assumed to be true, the p-value\\
\\
6 min read](https://www.geeksforgeeks.org/p-value-in-machine-learning/)\
\
* * *\
\
- [F-Test in Statistics\\
\\
\\
F test is a statistical test that is used in hypothesis testing that determines whether the variances of two samples are equal or not. The article will provide detailed information on f test, f statistic, its calculation, critical value and how to use it to test hypotheses. To understand F test firs\\
\\
6 min read](https://www.geeksforgeeks.org/f-test/)\
\
* * *\
\
- [Z-test : Formula, Types, Examples\\
\\
\\
After learning about inferential statistics we now move on to a more specific technique used for making decisions based on sample data â€“ the Z-test. Studying entire populations can be time-consuming, costly and sometimes impossible. so instead you take a sample from that population. This is where th\\
\\
9 min read](https://www.geeksforgeeks.org/z-test/)\
\
* * *\
\
- [Residual Leverage Plot (Regression Diagnostic)\\
\\
\\
In linear or multiple regression, it is not enough to just fit the model into the dataset. But, it may not give the desired result. To apply the linear or multiple regression efficiently to the dataset. There are some assumptions that we need to check on the dataset that made linear/multiple regress\\
\\
5 min read](https://www.geeksforgeeks.org/residual-leverage-plot-regression-diagnostic/)\
\
* * *\
\
- [Difference between Null and Alternate Hypothesis\\
\\
\\
Hypothesis is a statement or an assumption that may be true or false. There are six types of hypotheses mainly the Simple hypothesis, Complex hypothesis, Directional hypothesis, Associative hypothesis, and Null hypothesis. Usually, the hypothesis is the start point of any scientific investigation, I\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-null-and-alternate-hypothesis/)\
\
* * *\
\
- [Mann and Whitney U test\\
\\
\\
Mann and Whitney's U-test or Wilcoxon rank-sum testis the non-parametric statistic hypothesis test that is used to analyze the difference between two independent samples of ordinal data. In this test, we have provided two randomly drawn samples and we have to verify whether these two samples is from\\
\\
5 min read](https://www.geeksforgeeks.org/mann-and-whitney-u-test/)\
\
* * *\
\
- [Wilcoxon Signed Rank Test\\
\\
\\
The Wilcoxon Signed Rank Test is a non-parametric statistical test used to compare two related groups. It is often applied when the assumptions for the paired t-test (such as normality) are not met. This test evaluates whether there is a significant difference between two paired observations, making\\
\\
5 min read](https://www.geeksforgeeks.org/wilcoxon-signed-rank-test/)\
\
* * *\
\
- [Kruskal Wallis Test\\
\\
\\
The Kruskal-Wallis test (H test) is a nonparametric statistical test used to compare three or more independent groups to determine if there are statistically significant differences between them. It is an extension of the Mann-Whitney U test, which is used for comparing two groups. Unlike the one-wa\\
\\
5 min read](https://www.geeksforgeeks.org/kruskal-wallis-test/)\
\
* * *\
\
- [Friedman Test\\
\\
\\
The Friedman Test is a non-parametric statistical test used to detect differences in treatments across multiple test attempts. It is often used when the data is in the form of rankings or ordinal data, and when you have more than two related groups or repeated measures. The Friedman test is the non-\\
\\
6 min read](https://www.geeksforgeeks.org/friedman-test/)\
\
* * *\
\
- [Probability Class 10 Important Questions\\
\\
\\
Probability is a fundamental concept in mathematics for measuring of chances of an event happening By assigning numerical values to the chances of different outcomes, probability allows us to model, analyze, and predict complex systems and processes. Probability Formulas for Class 10 It says the pos\\
\\
4 min read](https://www.geeksforgeeks.org/probability-class-10-important-questions/)\
\
* * *\
\
\
## Probability and Probability Distributions\
\
- [Mathematics - Law of Total Probability\\
\\
\\
Probability theory is the branch of mathematics concerned with the analysis of random events. It provides a framework for quantifying uncertainty, predicting outcomes, and understanding random phenomena. In probability theory, an event is any outcome or set of outcomes from a random experiment, and\\
\\
13 min read](https://www.geeksforgeeks.org/mathematics-law-of-total-probability/)\
\
* * *\
\
- [Bayes's Theorem for Conditional Probability\\
\\
\\
Bayes's Theorem for Conditional Probability: Bayes's Theorem is a fundamental result in probability theory that describes how to update the probabilities of hypotheses when given evidence. Named after the Reverend Thomas Bayes, this theorem is crucial in various fields, including engineering, statis\\
\\
9 min read](https://www.geeksforgeeks.org/bayess-theorem-for-conditional-probability/)\
\
* * *\
\
- [Mathematics \| Probability Distributions Set 1 (Uniform Distribution)\\
\\
\\
Prerequisite - Random Variable In probability theory and statistics, a probability distribution is a mathematical function that can be thought of as providing the probabilities of occurrence of different possible outcomes in an experiment. For instance, if the random variable X is used to denote the\\
\\
4 min read](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-1/)\
\
* * *\
\
- [Mathematics \| Probability Distributions Set 4 (Binomial Distribution)\\
\\
\\
The previous articles talked about some of the Continuous Probability Distributions. This article covers one of the distributions which are not continuous but discrete, namely the Binomial Distribution. Introduction - To understand the Binomial distribution, we must first understand what a Bernoulli\\
\\
5 min read](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-4-binomial-distribution/)\
\
* * *\
\
- [Mathematics \| Probability Distributions Set 5 (Poisson Distribution)\\
\\
\\
The previous article covered the Binomial Distribution. This article talks about another Discrete Probability Distribution, the Poisson Distribution. Introduction -Suppose an event can occur several times within a given unit of time. When the total number of occurrences of the event is unknown, we c\\
\\
4 min read](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-5-poisson-distribution/)\
\
* * *\
\
- [Uniform Distribution \| Formula, Definition and Examples\\
\\
\\
A Uniform Distribution is a type of probability distribution in which every outcome in a given range is equally likely to occur. That means there is no biasâ€”no outcome is more likely than another within the specified set. It is also known as rectangular distribution (continuous uniform distribution)\\
\\
12 min read](https://www.geeksforgeeks.org/uniform-distribution-formula/)\
\
* * *\
\
- [Mathematics \| Probability Distributions Set 2 (Exponential Distribution)\\
\\
\\
The previous article covered the basics of Probability Distributions and talked about the Uniform Probability Distribution. This article covers the Exponential Probability Distribution which is also a Continuous distribution just like Uniform Distribution. Introduction - Suppose we are posed with th\\
\\
5 min read](https://www.geeksforgeeks.org/probability-distributions-exponential-distribution/)\
\
* * *\
\
- [Mathematics \| Probability Distributions Set 3 (Normal Distribution)\\
\\
\\
The previous two articles introduced two Continuous Distributions: Uniform and Exponential. This article covers the Normal Probability Distribution, also a Continuous distribution, which is by far the most widely used model for continuous measurement. Introduction - Whenever a random experiment is r\\
\\
5 min read](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-3-normal-distribution/)\
\
* * *\
\
- [Mathematics \| Beta Distribution Model\\
\\
\\
The Beta Distribution is a continuous probability distribution defined on the interval \[0, 1\], widely used in statistics and various fields for modeling random variables that represent proportions or probabilities. It is particularly useful when dealing with scenarios where the outcomes are bounded\\
\\
12 min read](https://www.geeksforgeeks.org/mathematics-beta-distribution-model/)\
\
* * *\
\
- [Gamma Distribution Model in Mathematics\\
\\
\\
Introduction : Suppose an event can occur several times within a given unit of time. When the total number of occurrences of the event is unknown, we can think of it as a random variable. Now, if this random variable X has gamma distribution, then its probability density function is given as follows\\
\\
2 min read](https://www.geeksforgeeks.org/gamma-distribution-model-in-mathematics/)\
\
* * *\
\
- [Chi-Square Test for Feature Selection - Mathematical Explanation\\
\\
\\
One of the primary tasks involved in any supervised Machine Learning venture is to select the best features from the given dataset to obtain the best results. One way to select these features is the Chi-Square Test. Mathematically, a Chi-Square test is done on two distributions two determine the lev\\
\\
4 min read](https://www.geeksforgeeks.org/chi-square-test-for-feature-selection-mathematical-explanation/)\
\
* * *\
\
- [Student's t-distribution in Statistics\\
\\
\\
As we know normal distribution assumes two important characteristics about the dataset: a large sample size and knowledge of the population standard deviation. However, if we do not meet these two criteria, and we have a small sample size or an unknown population standard deviation, then we use the\\
\\
10 min read](https://www.geeksforgeeks.org/students-t-distribution-in-statistics/)\
\
* * *\
\
- [Python - Central Limit Theorem\\
\\
\\
Central Limit Theorem (CLT) is a foundational principle in statistics, and implementing it using Python can significantly enhance data analysis capabilities. Statistics is an important part of data science projects. We use statistical tools whenever we want to make any inference about the population\\
\\
7 min read](https://www.geeksforgeeks.org/python-central-limit-theorem/)\
\
* * *\
\
- [Limits, Continuity and Differentiability\\
\\
\\
Limits, Continuity, and Differentiation are fundamental concepts in calculus. They are essential for analyzing and understanding function behavior and are crucial for solving real-world problems in physics, engineering, and economics. Table of Content LimitsKey Characteristics of LimitsExample of Li\\
\\
10 min read](https://www.geeksforgeeks.org/limits-continuity-differentiability/)\
\
* * *\
\
- [Implicit Differentiation\\
\\
\\
Implicit Differentiation is the process of differentiation in which we differentiate the implicit function without converting it into an explicit function. For example, we need to find the slope of a circle with an origin at 0 and a radius r. Its equation is given as x2 + y2 = r2. Now, to find the s\\
\\
6 min read](https://www.geeksforgeeks.org/implicit-differentiation/)\
\
* * *\
\
\
## Calculus for Machine Learning\
\
- [Partial Derivatives in Engineering Mathematics\\
\\
\\
Partial derivatives are a basic concept in multivariable calculus. They convey how a function would change when one of its input variables changes, while keeping all the others constant. This turns out to be particularly useful in fields such as physics, engineering, economics, and computer science,\\
\\
10 min read](https://www.geeksforgeeks.org/engineering-mathematics-partial-derivatives/)\
\
* * *\
\
- [Advanced Differentiation\\
\\
\\
Derivatives are used to measure the rate of change of any quantity. This process is called differentiation. It can be considered as a building block of the theory of calculus. Geometrically speaking, the derivative of any function at a particular point gives the slope of the tangent at that point of\\
\\
8 min read](https://www.geeksforgeeks.org/advanced-differentiation/)\
\
* * *\
\
- [How to find Gradient of a Function using Python?\\
\\
\\
The gradient of a function simply means the rate of change of a function. We will use numdifftools to find Gradient of a function. Examples: Input : x^4+x+1 Output :Gradient of x^4+x+1 at x=1 is 4.99 Input :(1-x)^2+(y-x^2)^2 Output :Gradient of (1-x^2)+(y-x^2)^2 at (1, 2) is \[-4. 2.\] Approach: For S\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-find-gradient-of-a-function-using-python/)\
\
* * *\
\
- [Optimization techniques for Gradient Descent\\
\\
\\
Gradient Descent is a widely used optimization algorithm for machine learning models. However, there are several optimization techniques that can be used to improve the performance of Gradient Descent. Here are some of the most popular optimization techniques for Gradient Descent: Learning Rate Sche\\
\\
4 min read](https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/)\
\
* * *\
\
- [Higher Order Derivatives\\
\\
\\
Higher order derivatives refer to the derivatives of a function that are obtained by repeatedly differentiating the original function. The first derivative of a function, fâ€²(x), represents the rate of change or slope of the function at a point.The second derivative, fâ€²â€²(x), is the derivative of the\\
\\
6 min read](https://www.geeksforgeeks.org/higher-order-derivatives/)\
\
* * *\
\
- [Taylor Series\\
\\
\\
A Taylor series represents a function as an infinite sum of terms, calculated from the values of its derivatives at a single point. Taylor series is a powerful mathematical tool used to approximate complex functions with an infinite sum of terms derived from the function's derivatives at a single po\\
\\
8 min read](https://www.geeksforgeeks.org/taylor-series/)\
\
* * *\
\
- [Application of Derivative - Maxima and Minima\\
\\
\\
Derivatives have many applications, like finding rate of change, approximation, maxima/minima and tangent. In this section, we focus on their use in finding maxima and minima. Note: If f(x) is a continuous function, then for every continuous function on a closed interval has a maximum and a minimum\\
\\
6 min read](https://www.geeksforgeeks.org/application-of-derivative-maxima-and-minima-mathematics/)\
\
* * *\
\
- [Absolute Minima and Maxima\\
\\
\\
Absolute Maxima and Minima are the maximum and minimum values of the function defined on a fixed interval. A function in general can have high values or low values as we move along the function. The maximum value of the function in any interval is called the maxima and the minimum value of the funct\\
\\
12 min read](https://www.geeksforgeeks.org/absolute-minima-and-maxima/)\
\
* * *\
\
- [Optimization for Data Science\\
\\
\\
From a mathematical foundation viewpoint, it can be said that the three pillars for data science that we need to understand quite well are Linear Algebra , Statistics and the third pillar is Optimization which is used pretty much in all data science algorithms. And to understand the optimization con\\
\\
5 min read](https://www.geeksforgeeks.org/optimization-for-data-science/)\
\
* * *\
\
- [Unconstrained Multivariate Optimization\\
\\
\\
Wikipedia defines optimization as a problem where you maximize or minimize a real function by systematically choosing input values from an allowed set and computing the value of the function. That means when we talk about optimization we are always interested in finding the best solution. So, let sa\\
\\
4 min read](https://www.geeksforgeeks.org/unconstrained-multivariate-optimization/)\
\
* * *\
\
- [Lagrange Multipliers \| Definition and Examples\\
\\
\\
In mathematics, a Lagrange multiplier is a potent tool for optimization problems and is applied especially in the cases of constraints. Named after the Italian-French mathematician Joseph-Louis Lagrange, the method provides a strategy to find maximum or minimum values of a function along one or more\\
\\
8 min read](https://www.geeksforgeeks.org/lagrange-multipliers/)\
\
* * *\
\
- [Lagrange's Interpolation\\
\\
\\
What is Interpolation? Interpolation is a method of finding new data points within the range of a discrete set of known data points (Source Wiki). In other words interpolation is the technique to estimate the value of a mathematical function, for any intermediate value of the independent variable. F\\
\\
7 min read](https://www.geeksforgeeks.org/lagranges-interpolation/)\
\
* * *\
\
- [Linear Regression in Machine learning\\
\\
\\
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It provides valuable insights for prediction and data analysis. This article will explore its types, assumptions, implementation, advantages and evaluation met\\
\\
15+ min read](https://www.geeksforgeeks.org/ml-linear-regression/)\
\
* * *\
\
- [Ordinary Least Squares (OLS) using statsmodels\\
\\
\\
Ordinary Least Squares (OLS) is a widely used statistical method for estimating the parameters of a linear regression model. It minimizes the sum of squared residuals between observed and predicted values. In this article we will learn how to implement Ordinary Least Squares (OLS) regression using P\\
\\
3 min read](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)\
\
* * *\
\
\
## Regression in Machine Learning\
\
Like\
\
We use cookies to ensure you have the best browsing experience on our website. By using our site, you\
acknowledge that you have read and understood our\
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &\
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)\
Got It !\
\
\
![Lightbox](https://www.geeksforgeeks.org/ml-linear-regression/)\
\
Improvement\
\
Suggest changes\
\
Suggest Changes\
\
Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.\
\
![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)\
\
Create Improvement\
\
Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.\
\
![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)\
\
Suggest Changes\
\
min 4 words, max Words Limit:1000\
\
## Thank You!\
\
Your suggestions are valuable to us.\
\
## What kind of Experience do you want to share?\
\
[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)\
\
[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1858087446.1745055573&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1192449619)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055574159&cv=11&fst=1745055574159&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102015664~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&ptag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fml-linear-regression%2F&hn=www.googleadservices.com&frm=0&tiba=Linear%20Regression%20in%20Machine%20learning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1809666265.1745055574&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)\
\
[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)