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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/?type%3Darticle%26id%3D319985&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Recurrent Neural Networks Explanation\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/recurrent-neural-networks-explanation/)

# SVM Hyperparameter Tuning using GridSearchCV \| ML

Last Updated : 11 Jan, 2023

Comments

Improve

Suggest changes

30 Likes

Like

Report

A Machine Learning model is defined as a mathematical model with a number of parameters that need to be learned from the data. However, there are some parameters, known as **Hyperparameters** and those cannot be directly learned. They are commonly chosen by humans based on some intuition or hit and trial before the actual training begins. These parameters exhibit their importance by improving the performance of the model such as its complexity or its learning rate. Models can have many hyper-parameters and finding the best combination of parameters can be treated as a search problem.

**SVM** also has some hyper-parameters (like what C or gamma values to use) and finding optimal hyper-parameter is a very hard task to solve. But it can be found by just trying all combinations and see what parameters work best. The main idea behind it is to create a grid of hyper-parameters and just try all of their combinations (hence, this method is called **Gridsearch**, But don’t worry! we don’t have to do it manually because Scikit-learn has this functionality built-in with GridSearchCV.

**GridSearchCV** takes a dictionary that describes the parameters that could be tried on a model to train it. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.

This article demonstrates how to use the **GridSearchCV** searching method to find optimal hyper-parameters and hence improve the accuracy/prediction results

### Import necessary libraries and get the Data:

We’ll use the built-in breast cancer dataset from Scikit Learn. We can get with the load function:

- Python3

## Python3

|     |
| --- |
| `import` `pandas as pd `<br>`import` `numpy as np `<br>`from` `sklearn.metrics ` `import` `classification_report, confusion_matrix `<br>`from` `sklearn.datasets ` `import` `load_breast_cancer `<br>`from` `sklearn.svm ` `import` `SVC `<br>` `<br>`cancer ` `=` `load_breast_cancer() `<br>` `<br>`# The data set is presented in a dictionary form: `<br>`print` `(cancer.keys()) ` |

```

```

```

```

```
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

Now we will extract all features into the new data frame and our target features into separate data frames.

- Python3

## Python3

|     |
| --- |
| `df_feat ` `=` `pd.DataFrame(cancer[` `'data'` `], `<br>`                       ``columns ` `=` `cancer[` `'feature_names'` `]) `<br>` `<br>`# cancer column is our target `<br>`df_target ` `=` `pd.DataFrame(cancer[` `'target'` `],  `<br>`                     ``columns ` `=` `[` `'Cancer'` `]) `<br>` `<br>`print` `(` `"Feature Variables: "` `) `<br>`print` `(df_feat.info()) ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20190705194128/svm_tuning1.png)

- Python3

## Python3

|     |
| --- |
| `print` `(` `"Dataframe looks like : "` `) `<br>`print` `(df_feat.head()) ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20190705194145/svm_tuning2.png)

### Train Test Split

Now we will split our data into train and test set with a 70: 30 ratio

- Python3

## Python3

|     |
| --- |
| `from` `sklearn.model_selection ` `import` `train_test_split `<br>` `<br>`X_train, X_test, y_train, y_test ` `=` `train_test_split( `<br>`                        ``df_feat, np.ravel(df_target), `<br>`                ``test_size ` `=` `0.30` `, random_state ` `=` `101` `) ` |

```

```

```

```

### Train the Support Vector Classifier without Hyper-parameter Tuning –

First, we will train our model by calling the standard SVC() function without doing Hyperparameter Tuning and see its classification and confusion matrix.

- Python3

## Python3

|     |
| --- |
| `# train the model on train set `<br>`model ` `=` `SVC() `<br>`model.fit(X_train, y_train) `<br>` `<br>`# print prediction results `<br>`predictions ` `=` `model.predict(X_test) `<br>`print` `(classification_report(y_test, predictions)) ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20190705194239/svm_tuning3.png)

**We got 61 % accuracy but did you notice something strange?**

Notice that recall and precision for class 0 are always 0. It means that the classifier is always classifying everything into a single class i.e class 1! This means our model needs to have its parameters tuned.

Here is when the usefulness of GridSearch comes into the picture. We can search for parameters using GridSearch!

### Use GridsearchCV

One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC and creates a new estimator, that behaves exactly the same – in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want, the higher the number, the more verbose (verbose just means the text output describing the process).

- Python3

## Python3

|     |
| --- |
| `from` `sklearn.model_selection ` `import` `GridSearchCV `<br>` `<br>`# defining parameter range `<br>`param_grid ` `=` `{` `'C'` `: [` `0.1` `, ` `1` `, ` `10` `, ` `100` `, ` `1000` `],  `<br>`              ``'gamma'` `: [` `1` `, ` `0.1` `, ` `0.01` `, ` `0.001` `, ` `0.0001` `], `<br>`              ``'kernel'` `: [` `'rbf'` `]}  `<br>` `<br>`grid ` `=` `GridSearchCV(SVC(), param_grid, refit ` `=` `True` `, verbose ` `=` `3` `) `<br>` `<br>`# fitting the model for grid search `<br>`grid.fit(X_train, y_train) ` |

```

```

```

```

What **fit** does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.

You can inspect the best parameters found by GridSearchCV in the best\_params\_ attribute, and the best estimator in the best\_estimator\_ attribute:

- Python3

## Python3

|     |
| --- |
| `# print best parameter after tuning `<br>`print` `(grid.best_params_) `<br>` `<br>`# print how our model looks after hyper-parameter tuning `<br>`print` `(grid.best_estimator_) ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20190705194417/svm_tuning4.png)

Then you can re-run predictions and see a classification report on this grid object just like you would with a normal model.

- Python3

## Python3

|     |
| --- |
| `grid_predictions ` `=` `grid.predict(X_test) `<br>` `<br>`# print classification report `<br>`print` `(classification_report(y_test, grid_predictions)) ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20190705194435/svm_tuning5.png)

We have got almost **95 % prediction** result.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/recurrent-neural-networks-explanation/)

[Recurrent Neural Networks Explanation](https://www.geeksforgeeks.org/recurrent-neural-networks-explanation/)

[T](https://www.geeksforgeeks.org/user/tyagikartik4282/)

[tyagikartik4282](https://www.geeksforgeeks.org/user/tyagikartik4282/)

Follow

30

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Hyperparameter tuning using GridSearchCV and KerasClassifier\\
\\
\\
Hyperparameter tuning is done to increase the efficiency of a model by tuning the parameters of the neural network. Some scikit-learn APIs like GridSearchCV and RandomizedSearchCV are used to perform hyper parameter tuning. In this article, you'll learn how to use GridSearchCV to tune Keras Neural N\\
\\
2 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-using-gridsearchcv-and-kerasclassifier/?ref=ml_lbp)
[Hyperparameter tuning SVM parameters using Genetic Algorithm\\
\\
\\
The performance support Vector Machines (SVMs) are heavily dependent on hyperparameters such as the regularization parameter (C) and the kernel parameters (gamma for RBF kernel). Genetic Algorithms (GAs) leverage evolutionary principles to search for optimal hyperparameter values. This article explo\\
\\
9 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-svm-parameters-using-genetic-algorithm/?ref=ml_lbp)
[Hyperparameter Tuning with R\\
\\
\\
In R Language several techniques and packages can be used to optimize these hyperparameters, leading to better, more reliable models. in this article, we will discuss all the techniques and packages for Hyperparameter Tuning with R. What are Hyperparameters?Hyperparameters are the settings that cont\\
\\
5 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-with-r/?ref=ml_lbp)
[Hyperparameter tuning\\
\\
\\
Machine Learning model is defined as a mathematical model with several parameters that need to be learned from the data. By training a model with existing data we can fit the model parameters. However there is another kind of parameter known as hyperparameters which cannot be directly learned from t\\
\\
8 min read](https://www.geeksforgeeks.org/hyperparameter-tuning/?ref=ml_lbp)
[How to tune a Decision Tree in Hyperparameter tuning\\
\\
\\
Decision trees are powerful models extensively used in machine learning for classification and regression tasks. The structure of decision trees resembles the flowchart of decisions helps us to interpret and explain easily. However, the performance of decision trees highly relies on the hyperparamet\\
\\
14 min read](https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/?ref=ml_lbp)
[Random Forest Hyperparameter Tuning in Python\\
\\
\\
Random Forest is one of the most popular and powerful machine learning algorithms used for both classification and regression tasks. It works by building multiple decision trees and combining their outputs to improve accuracy and control overfitting. While Random Forest is already a robust model fin\\
\\
6 min read](https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/?ref=ml_lbp)
[Cross-validation and Hyperparameter tuning of LightGBM Model\\
\\
\\
In a variety of industries, including finance, healthcare, and marketing, machine learning models have become essential for resolving challenging real-world issues. Gradient boosting techniques have become incredibly popular among the myriad of machine learning algorithms due to their remarkable pre\\
\\
14 min read](https://www.geeksforgeeks.org/cross-validation-and-hyperparameter-tuning-of-lightgbm-model/?ref=ml_lbp)
[CatBoost Cross-Validation and Hyperparameter Tuning\\
\\
\\
CatBoost is a powerful gradient-boosting algorithm of machine learning that is very popular for its effective capability to handle categorial features of both classification and regression tasks. To maximize the potential of CatBoost, it's essential to fine-tune its hyperparameters which can be done\\
\\
11 min read](https://www.geeksforgeeks.org/catboost-cross-validation-and-hyperparameter-tuning/?ref=ml_lbp)
[Hyperparameter Tuning in Linear Regression\\
\\
\\
Linear regression is one of the simplest and most widely used algorithms in machine learning. Despite its simplicity, it can be quite powerful, especially when combined with proper hyperparameter tuning. Hyperparameter tuning is the process of tuning a machine learning model's parameters to achieve\\
\\
7 min read](https://www.geeksforgeeks.org/hyperparameter-tuning-in-linear-regression/?ref=ml_lbp)
[Sklearn \| Model Hyper-parameters Tuning\\
\\
\\
Hyperparameter tuning is the process of finding the optimal values for the hyperparameters of a machine-learning model. Hyperparameters are parameters that control the behaviour of the model but are not learned during training. Hyperparameter tuning is an important step in developing machine learnin\\
\\
12 min read](https://www.geeksforgeeks.org/sklearn-model-hyper-parameters-tuning/?ref=ml_lbp)

Like30

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2077283035.1745055899&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=674072211)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

Sign In

By creating this account, you agree to our [Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/) & [Cookie Policy.](https://www.geeksforgeeks.org/legal/privacy-policy/#:~:text=the%20appropriate%20measures.-,COOKIE%20POLICY,-A%20cookie%20is)

# Create Account

Already have an account ?Log in

Continue with Google

or

Username or Email

Password

Institution / Organization

```

```

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)