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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-feature-scaling-part-2/?type%3Darticle%26id%3D209160&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Pandas Read CSV in Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/)

# Feature Engineering: Scaling, Normalization, and Standardization

Last Updated : 09 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Feature Scaling is a technique to standardize the independent features present in the data. It is performed during the data pre-processing to handle highly varying values. If [feature scaling](https://www.geeksforgeeks.org/python-how-and-where-to-apply-feature-scaling/) is not done then **machine learning algorithm tends to use greater values as higher and consider smaller values as lower regardless of the unit of the values**. For example it will take 10 m and 10 cm both as same regardless of their unit. In this article we will learn about different techniques which are used to perform feature scaling.

## 1\. Absolute Maximum Scaling

This method of scaling requires two-step:

1. We should first select the maximum absolute value out of all the entries of a particular measure.
2. Then after this we divide each entry of the column by this maximum value.

Xscaled=Xi−max(∣X∣)max(∣X∣)X\_{\\rm {scaled }}=\\frac{X\_{i}-\\rm{max}\\left(\|X\|\\right)}{\\rm{max}\\left(\|X\|\\right)}Xscaled​=max(∣X∣)Xi​−max(∣X∣)​

After performing the above-mentioned two steps we will observe that each entry of the column lies in the range of -1 to 1. But this method is not used that often the reason behind this is that it is too sensitive to the outliers. And while dealing with the real-world data presence of outliers is a very common thing.

For the demonstration purpose we will use the dataset which you can download from [here](https://media.geeksforgeeks.org/wp-content/uploads/20250114174407596134/SampleFile.csv). This dataset is a simpler version of the original house price prediction dataset having only two columns from the original dataset. The first five rows of the original data are shown below:

Python`
import pandas as pd
df = pd.read_csv('SampleFile.csv')
print(df.head())
`

**Output:**

```
   LotArea  MSSubClass
0     8450          60
1     9600          20
2    11250          60
3     9550          70
4    14260          60

```

Now let’s apply the first method which is of the absolute maximum scaling. For this first, we are supposed to evaluate the absolute maximum values of the columns.

Python`
import numpy as np
max_vals = np.max(np.abs(df))
max_vals
`

**Output:**

```
LotArea       215245
MSSubClass       190
dtype: int64

```

Now we are supposed to subtract these values from the data and then divide the results from the maximum values as well.

Python`
print((df - max_vals) / max_vals)
`

**Output:**

```
       LotArea  MSSubClass
0    -0.960742   -0.999721
1    -0.955400   -0.999907
2    -0.947734   -0.999721
3    -0.955632   -0.999675
4    -0.933750   -0.999721
...        ...         ...
1455 -0.963219   -0.999721
1456 -0.938791   -0.999907
1457 -0.957992   -0.999675
1458 -0.954856   -0.999907
1459 -0.953834   -0.999907
[1460 rows x 2 columns]

```

## 2\. Min-Max Scaling

This method of scaling requires below two-step:

1. First we are supposed to find the minimum and the maximum value of the column.
2. Then we will subtract the minimum value from the entry and divide the result by the difference between the maximum and the minimum value.

Xscaled=Xi−XminXmax–XminX\_{\\rm {scaled }}=\\frac{X\_{i}-X\_{\\text {min}}}{X\_{\\rm{max}} – X\_{\\rm{min}}}Xscaled​=Xmax​–Xmin​Xi​−Xmin​​

As we are using the maximum and the minimum value this method is also prone to [outliers](https://www.geeksforgeeks.org/machine-learning-outlier/) but the range in which the data will range after performing the above two steps is between 0 to 1.

Python`
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
scaled_df.head()
`

**Output:**

```
    LotArea  MSSubClass
0  0.033420    0.235294
1  0.038795    0.000000
2  0.046507    0.235294
3  0.038561    0.294118
4  0.060576    0.235294

```

## 3\. Normalization

This method is more or less the same as the previous method but here instead of the minimum value we subtract each entry by the mean value of the whole data and then divide the results by the difference between the minimum and the maximum value.

Xscaled=Xi−XmeanXmax–XminX\_{\\rm {scaled }}=\\frac{X\_{i}-X\_{\\text {mean}}}{X\_{\\rm{max}} – X\_{\\rm{min}}}Xscaled​=Xmax​–Xmin​Xi​−Xmean​​

Python`
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
print(scaled_df.head())
`

**Output:**

```
    LotArea  MSSubClass
0  0.999975    0.007100
1  0.999998    0.002083
2  0.999986    0.005333
3  0.999973    0.007330
4  0.999991    0.004208

```

## 4\. Standardization

This method of scaling is basically based on the central tendencies and variance of the data.

1. First we should calculate the [mean and standard deviation](https://www.geeksforgeeks.org/mathematics-mean-variance-and-standard-deviation/) of the data we would like to normalize it.
2. Then we are supposed to subtract the mean value from each entry and then divide the result by the standard deviation.

This helps us achieve a [normal distribution](https://www.geeksforgeeks.org/mathematics-probability-distributions-set-3-normal-distribution/) of the data with a mean equal to zero and a standard deviation equal to 1.

Xscaled=Xi−Xmean σX\_{\\rm {scaled }}=\\frac{X\_{i}-X\_{\\text {mean }}}{\\sigma}Xscaled​=σXi​−Xmean ​​

Python`
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
print(scaled_df.head())
`

**Output:**

```
    LotArea  MSSubClass
0 -0.207142    0.073375
1 -0.091886   -0.872563
2  0.073480    0.073375
3 -0.096897    0.309859
4  0.375148    0.073375

```

## 5\. Robust Scaling

In this method of scaling, we use two main statistical measures of the data.

- [Median](https://www.geeksforgeeks.org/median/)
- [Inter-Quartile Range](https://www.geeksforgeeks.org/interquartile-range-formula/)

After calculating these two values we are supposed to subtract the median from each entry and then divide the result by the interquartile range.

Xscaled=Xi−Xmedian IQRX\_{\\rm {scaled }}=\\frac{X\_{i}-X\_{\\text {median }}}{IQR}Xscaled​=IQRXi​−Xmedian ​​

Python`
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
print(scaled_df.head())
`

**Output:**

```
    LotArea  MSSubClass
0 -0.254076         0.2
1  0.030015        -0.6
2  0.437624         0.2
3  0.017663         0.4
4  1.181201         0.2
```

In conclusion **scaling, normalization and standardization** are essential feature engineering techniques that ensure data is well-prepared for machine learning models. They help improve model performance, enhance convergence and reduce biases. Choosing the right method depends on your data and algorithm.

## Why use Feature Scaling?

In machine learning feature scaling is used for number of purposes:

- **Range:** Scaling guarantees that all features are on a comparable scale and have comparable ranges. This process is known as feature normalisation. This is significant because the magnitude of the features has an impact on many machine learning techniques. Larger scale features may dominate the learning process and have an excessive impact on the outcomes.
- **Algorithm performance improvement:** When the features are scaled several machine learning methods including gradient descent-based algorithms, distance-based algorithms (such k-nearest neighbours) and support vector machines perform better or converge more quickly. The algorithm’s performance can be enhanced by scaling the features which prevent the convergence of the algorithm to the ideal outcome.
- **Preventing numerical instability:** Numerical instability can be prevented by avoiding significant scale disparities between features. For examples include distance calculations where having features with differing scales can result in numerical overflow or underflow problems. Stable computations are required to mitigate this issue by scaling the features.
- **Equal importance:** Scaling features makes sure that each characteristic is given the same consideration during the learning process. Without scaling bigger scale features could dominate the learning producing skewed outcomes. This bias is removed through scaling and each feature contributes fairly to model predictions.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/)

[Pandas Read CSV in Python](https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/)

[![author](https://media.geeksforgeeks.org/auth/profile/71uvm7pynx4cnk0rtuk6)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

[mohit gupta\_omg :)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Normalization vs Standardization\\
\\
\\
Feature scaling is one of the most important data preprocessing step in machine learning. Algorithms that compute the distance between the features are biased towards numerically larger values if the data is not scaled. Tree-based algorithms are fairly insensitive to the scale of the features. Also,\\
\\
3 min read](https://www.geeksforgeeks.org/normalization-vs-standardization/)
[Z-Score Normalization: Definition and Examples\\
\\
\\
Z-score normalization, also known as standardization, is a crucial data preprocessing technique in machine learning and statistics. It is used to transform data into a standard normal distribution, ensuring that all features are on the same scale. This process helps to avoid the dominance of certain\\
\\
6 min read](https://www.geeksforgeeks.org/z-score-normalization-definition-and-examples/)
[Feature Engineering in R Programming\\
\\
\\
Feature engineering is the process of transforming raw data into features that can be used in a machine-learning model. In R programming, feature engineering can be done using a variety of built-in functions and packages. One common approach to feature engineering is to use the dplyr package to mani\\
\\
7 min read](https://www.geeksforgeeks.org/feature-engineering-in-r-programming/)
[What is Standardization in Machine Learning\\
\\
\\
In Machine Learning we train our data to predict or classify things in such a manner that isn't hardcoded in the machine. So for the first, we have the Dataset or the input data to be pre-processed and manipulated for our desired outcomes. Any ML Model to be built follows the following procedure: Co\\
\\
6 min read](https://www.geeksforgeeks.org/what-is-standardization-in-machine-learning/)
[Logistic Regression and the Feature Scaling Ensemble\\
\\
\\
Logistic Regression is a widely used classification algorithm in machine learning. However, to enhance its performance further specially when dealing with features of different scales, employing feature scaling ensemble techniques becomes imperative. In this guide, we will dive depth into logistic r\\
\\
9 min read](https://www.geeksforgeeks.org/logistic-regression-and-the-feature-scaling-ensemble/)
[Normalization and Scaling\\
\\
\\
Normalization and Scaling are two fundamental preprocessing techniques when you perform data analysis and machine learning. They are useful when you want to rescale, standardize or normalize the features (values) through distribution and scaling of existing data that make your machine learning model\\
\\
9 min read](https://www.geeksforgeeks.org/normalization-and-scaling/)
[Difference Between StandardScaler and Normalizer in sklearn.preprocessing\\
\\
\\
Preprocessing step in machine learning task that helps improve the performance of models. Two commonly used techniques in the sklearn.preprocessing module are StandardScaler and Normalizer. Although both are used to transform features, they serve different purposes and apply different methods. In th\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-standardscaler-and-normalizer-in-sklearn-preprocessing/)
[What is Feature Engineering?\\
\\
\\
Feature Engineering is the process of creating new features or transforming existing features to improve the performance of a machine-learning model. It involves selecting relevant information from raw data and transforming it into a format that can be easily understood by a model. The goal is to im\\
\\
14 min read](https://www.geeksforgeeks.org/what-is-feature-engineering/)
[Feature Engineering for Time-Series Data: Methods and Applications\\
\\
\\
Time-series data, which consists of sequential measurements taken over time, is ubiquitous in many fields such as finance, healthcare, and social media. Extracting useful features from this type of data can significantly improve the performance of predictive models and help uncover underlying patter\\
\\
9 min read](https://www.geeksforgeeks.org/feature-engineering-for-time-series-data-methods-and-applications/)
[What is Zero Mean and Unit Variance Normalization\\
\\
\\
Answer: Zero Mean and Unit Variance normalization rescale data to have a mean of zero and a standard deviation of one.Explanation:Mean Centering: The first step of Zero Mean normalization involves subtracting the mean value of each feature from all data points. This centers the data around zero, mea\\
\\
2 min read](https://www.geeksforgeeks.org/what-is-zero-mean-and-unit-variance-normalization/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-feature-scaling-part-2/)

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

[iframe](https://www.googletagmanager.com/static/service_worker/54a0/sw_iframe.html?origin=https%3A%2F%2Fwww.geeksforgeeks.org)