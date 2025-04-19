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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/?type%3Darticle%26id%3D924948&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Microsoft Stock Price Prediction with Machine Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/microsoft-stock-price-prediction-with-machine-learning/)

# Random Forest Hyperparameter Tuning in Python

Last Updated : 30 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

[Random Forest](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/) is one of the most popular and powerful machine learning algorithms used for both classification and regression tasks. It _**works by building multiple decision trees and combining their outputs to improve accuracy and control overfitting**_. While Random Forest is already a robust model **fine-tuning its hyperparameters such as the number of trees, maximum depth and feature selection can improve its prediction.**

This optimization process tailoring the model to better fit specific datasets result in improved performance and more accurate and reliable predictions. Therefore fine tuning plays a crucial role in maximizing the effectiveness of the Random Forest model and in this article we will learn how we can do it.

## Random Forest Hyperparameters

Since we are talking about Random Forest Hyperparameters let us see what different Hyperparameters can be Tuned.

**1\. n\_estimators:** Defines the number of trees in the forest. More trees typically improve model performance but increase computational cost.

```
By default: n_estimators=100
```

- n\_estimators=100 means it takes 100 decision trees for prediction

**2\. max\_features:** Limits the number of features to consider when splitting a node. This helps control overfitting.

```
By default: max_features="sqrt" [available: ["sqrt", "log2", None}]\
```\
\
- **`sqrt: `** Selects the square root of the total features. This is a common setting to reduce overfitting and speed up the model.\
- **`log2`**: This option selects the base-2 logarithm of the total number of features. It provide more randomness and reduce overfitting more than the square root option.\
- **`None`**: If `None` is chosen the model uses all available features for splitting each node. This increases the model’s complexity and may cause overfitting, especially with many features.\
\
**3\. max\_depth**: Controls the maximum depth of each tree. A shallow tree may underfit while a deep tree may overfit. So choosing right value of it is important.\
\
```\
By default: max_depth=None\
```\
\
**4\. max\_leaf\_nodes:** Limits the number of leaf nodes in the tree hence controlling its size and complexity.\
\
```\
By default: max_leaf_nodes = None\
```\
\
- max\_leaf\_nodes = None means it takes an unlimited number of nodes\
\
**5\. max\_sample**: Apart from the features, we have a large set of training datasets. max\_sample determines how much of the dataset is given to each individual tree.\
\
```\
By default: max_sample = None\
```\
\
- max\_sample = None means data.shape\[0\] is taken\
\
**6\. min\_sample\_split:** Specifies the minimum number of samples required to split an internal node.\
\
```\
By default: min_sample_split = 2\
```\
\
- min\_sample\_split = 2 this means every node has 2 subnodes\
\
> For a more detailed article, you can check this:  [Hyperparameters of Random Forest Classifier](https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/)\
\
## Random Forest Hyperparameter Tuning in Python using Sklearn\
\
Scikit-learn offers tools for hyperparameter tuning which can help improve the performance of machine learning models. Hyperparameter tuning involves selecting the best set of parameters for a given model to maximize its efficiency and accuracy. We will explore two commonly used techniques for hyperparameter tuning: **GridSearchCV** and **RandomizedSearchCV**.\
\
Both methods are essential for automating the process of fine-tuning machine learning models and we will examine how each works and when to use them.\
\
**Below is the code with random forest working on heart disease prediction.**\
\
Python`\
from sklearn.metrics import classification_report\
from sklearn.model_selection import train_test_split\
import pandas as pd\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV\
data = pd.read_csv(\
    "https://raw.githubusercontent.com/lucifertrj/100DaysOfML/main/Day14%3A%20Logistic_Regression"\
    "_Metric_and_practice/heart_disease.csv")\
data.head(7)\
X = data.drop("target", axis=1)\
y = data['target']\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\
model = RandomForestClassifier(\
    n_estimators=100,\
    max_features="sqrt",\
    max_depth=6,\
    max_leaf_nodes=6\
)\
model.fit(X_train, y_train)\
y_pred = model.predict(X_test)\
print(classification_report(y_pred, y_test))\
`\
\
**Output:**\
\
> precision recall f1-score support\
>\
> 0 0.77 0.87 0.82 31\
>\
> 1 0.90 0.82 0.86 45\
>\
> accuracy 0.84 76\
>\
> macro avg 0.84 0.85 0.84 76\
>\
> weighted avg 0.85 0.84 0.84 76\
\
The classification report shows that the model has an accuracy of 84% with good precision for class 1 (0.90) but slightly lower precision for class 0 (0.77) and a recall of 0.87 for class 0. This suggests that fine-tuning hyperparameters such as `n_estimators` and `max_depth` could help improve the performance especially for class 0.\
\
### 1\. Hyperparameter Tuning- GridSearchCV\
\
First let’s use [GridSearchCV](https://www.geeksforgeeks.org/hyperparameter-tuning-using-gridsearchcv-and-kerasclassifier/) to obtain the best parameters for the model. It is a hyperparameter tuning method in Scikit-learn that exhaustively searches through all possible combinations of parameters provided in the **`param_grid`**. For that we will pass **RandomForestClassifier**() instance to the model and then fit the GridSearchCV using the training data to find the best parameters.\
\
Python`\
grid_search = GridSearchCV(RandomForestClassifier(),\
                           param_grid=param_grid)\
grid_search.fit(X_train, y_train)\
print(grid_search.best_estimator_)\
`\
\
- **param\_grid**: A dictionary containing hyperparameters and their possible values. GridSearchCV will try every combination of these values to find the best-performing set of hyperparameters.\
- **grid\_search.fit(X\_train, y\_train)**: This trains the model on the training data ( `X_train`, `y_train`) for every combination of hyperparameters defined in `param_grid`.\
- **grid\_search.best\_estimator\_**: After completing the grid search, this will print the RandomForest model that has the best combination of hyperparameters from the search.\
\
**Output:**\
\
> RandomForestClassifier(max\_depth=3, max\_features=”log2″, max\_leaf\_nodes=3, n\_estimators=50)\
\
#### Update the Model\
\
Now we will update the parameters of the model by those which are obtained by using GridSearchCV.\
\
Python`\
model_grid = RandomForestClassifier(max_depth=3,\
                                    max_features="log2",\
                                    max_leaf_nodes=3,\
                                    n_estimators=50)\
model_grid.fit(X_train, y_train)\
y_pred_grid = model.predict(X_test)\
print(classification_report(y_pred_grid, y_test))\
`\
\
**Output:**\
\
> precision recall f1-score support\
>\
> 0 0.77 0.84 0.81 32\
>\
> 1 0.88 0.82 0.85 44\
>\
> accuracy 0.83 76\
>\
> macro avg 0.82 0.83 0.83 76\
>\
> weighted avg 0.83 0.83 0.83 76\
\
### Hyperparameter Tuning- RandomizedSearchCV\
\
**RandomizedSearchCV** performs a random search over a specified parameter grid. Unlike **GridSearchCV**, which tries every possible combination of hyperparameters. **RandomizedSearchCV** randomly selects combinations and evaluates the model often leading to faster results especially when there are many hyperparameters.\
\
Now let’s use [RandomizedSearchCV](https://www.geeksforgeeks.org/comparing-randomized-search-and-grid-search-for-hyperparameter-estimation-in-scikit-learn/) to obtain the best parameters for the model. For that we will pass RandomFoestClassifier() instance to the model and then fit the RandomizedSearchCV using the training data to find the best parameters.\
\
Python`\
random_search = RandomizedSearchCV(RandomForestClassifier(),\
                                   param_grid)\
random_search.fit(X_train, y_train)\
print(random_search.best_estimator_)\
`\
\
- **param\_grid** specifies the hyperparameters that you want to tune similar to the grid in **GridSearchCV**.\
- **fit(X\_train, y\_train)** trains the model using the training data.\
- **best\_estimator\_** shows the model with the best combination of hyperparameters found by the search process.\
\
**Output:**\
\
> RandomForestClassifier(max\_depth=3, max\_features=’log2′, max\_leaf\_nodes=6)\
\
#### Update the model\
\
Now we will update the parameters of the model by those which are obtained by using RandomizedSearchCV.\
\
Python`\
model_random = RandomForestClassifier(max_depth=3,\
                                      max_features='log2',\
                                      max_leaf_nodes=6,\
                                      n_estimators=100)\
model_random.fit(X_train, y_train)\
y_pred_rand = model.predict(X_test)\
print(classification_report(y_pred_rand, y_test))\
`\
\
**Output:**\
\
> precision recall f1-score support\
>\
> 0 0.77 0.84 0.81 32\
>\
> 1 0.88 0.82 0.85 44\
>\
> accuracy 0.83 76\
>\
> macro avg 0.82 0.83 0.83 76\
>\
> weighted avg 0.83 0.83 0.83 76\
\
Both **GridSearchCV** and **RandomizedSearchCV** significantly improved the model’s performance by optimizing hyperparameters like max\_depth, max\_features and n\_estimators. These methods help identify the best combination of hyperparameters leading to improved model accuracy and more balanced precision, recall and F1-scores for both classes. The accuracy was enhanced to 83% indicating that hyperparameter tuning helped the model generalize better and perform more reliably.\
\
Comment\
\
\
More info\
\
[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)\
\
[Next Article](https://www.geeksforgeeks.org/microsoft-stock-price-prediction-with-machine-learning/)\
\
[Microsoft Stock Price Prediction with Machine Learning](https://www.geeksforgeeks.org/microsoft-stock-price-prediction-with-machine-learning/)\
\
![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)\
\
GeeksforGeeks\
\
Improve\
\
Article Tags :\
\
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)\
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)\
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)\
\
Practice Tags :\
\
- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)\
\
### Similar Reads\
\
[Hyperparameter tuning with Ray Tune in PyTorch\\
\\
\\
Hyperparameter tuning is a crucial step in the machine learning pipeline that can significantly impact the performance of a model. Choosing the right set of hyperparameters can be the difference between an average model and a highly accurate one. Ray Tune is an industry-standard tool for distributed\\
\\
8 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-with-ray-tune-in-pytorch/?ref=ml_lbp)\
[How to tune a Decision Tree in Hyperparameter tuning\\
\\
\\
Decision trees are powerful models extensively used in machine learning for classification and regression tasks. The structure of decision trees resembles the flowchart of decisions helps us to interpret and explain easily. However, the performance of decision trees highly relies on the hyperparamet\\
\\
14 min read](https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/?ref=ml_lbp)\
[Hyperparameters of Random Forest Classifier\\
\\
\\
In this article, we are going to learn about different hyperparameters that exist in a Random Forest Classifier. We have already learnt about the implementation of Random Forest Classifier using scikit-learn library in the article https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-l\\
\\
4 min read](https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/?ref=ml_lbp)\
[Hyperparameter Tuning in Linear Regression\\
\\
\\
Linear regression is one of the simplest and most widely used algorithms in machine learning. Despite its simplicity, it can be quite powerful, especially when combined with proper hyperparameter tuning. Hyperparameter tuning is the process of tuning a machine learning model's parameters to achieve\\
\\
7 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-in-linear-regression/?ref=ml_lbp)\
[Hyperparameter tuning with Optuna in PyTorch\\
\\
\\
Hyperparameter tuning is a critical step in the machine learning pipeline, often determining the success of a model. Optuna is a powerful and flexible framework for hyperparameter optimization, designed to automate the search for optimal hyperparameters. When combined with PyTorch, a popular deep le\\
\\
5 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-with-optuna-in-pytorch/?ref=ml_lbp)\
[Random Forest Regression in Python\\
\\
\\
A random forest is an ensemble learning method that combines the predictions from multiple decision trees to produce a more accurate and stable prediction. It is a type of supervised learning algorithm that can be used for both classification and regression tasks. In regression task we can use Rando\\
\\
9 min read](https://www.geeksforgeeks.org/random-forest-regression-in-python/?ref=ml_lbp)\
[Hyperparameter tuning\\
\\
\\
Machine Learning model is defined as a mathematical model with several parameters that need to be learned from the data. By training a model with existing data we can fit the model parameters. However there is another kind of parameter known as hyperparameters which cannot be directly learned from t\\
\\
8 min read](https://www.geeksforgeeks.org/hyperparameter-tuning/?ref=ml_lbp)\
[Hyperparameter Tuning with R\\
\\
\\
In R Language several techniques and packages can be used to optimize these hyperparameters, leading to better, more reliable models. in this article, we will discuss all the techniques and packages for Hyperparameter Tuning with R. What are Hyperparameters?Hyperparameters are the settings that cont\\
\\
5 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-with-r/?ref=ml_lbp)\
[HyperParameter Tuning: Fixing Overfitting in Neural Networks\\
\\
\\
Overfitting is a pervasive problem in neural networks, where the model becomes too specialized to the training data and fails to generalize well to new, unseen data. This issue can be addressed through hyperparameter tuning, which involves adjusting various parameters to optimize the performance of\\
\\
6 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-fixing-overfitting-in-neural-networks/?ref=ml_lbp)\
[Sklearn \| Model Hyper-parameters Tuning\\
\\
\\
Hyperparameter tuning is the process of finding the optimal values for the hyperparameters of a machine-learning model. Hyperparameters are parameters that control the behaviour of the model but are not learned during training. Hyperparameter tuning is an important step in developing machine learnin\\
\\
12 min read](https://www.geeksforgeeks.org/sklearn-model-hyper-parameters-tuning/?ref=ml_lbp)\
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
![Lightbox](https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/)\
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
[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1271700236.1745056430&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=1873950872)\
\
Login Modal \| GeeksforGeeks\
\
# Log in\
\
New user ?Register Now\
\
Continue with Google\
\
or\
\
Username or Email\
\
Password\
\
Remember me\
\
Forgot Password\
\
Sign In\
\
By creating this account, you agree to our [Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/) & [Cookie Policy.](https://www.geeksforgeeks.org/legal/privacy-policy/#:~:text=the%20appropriate%20measures.-,COOKIE%20POLICY,-A%20cookie%20is)\
\
# Create Account\
\
Already have an account ?Log in\
\
Continue with Google\
\
or\
\
Username or Email\
\
Password\
\
Institution / Organization\
\
```\
\
```\
\
Sign Up\
\
\*Please enter your email address or userHandle.\
\
Back to Login\
\
Reset Password\
\
[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)