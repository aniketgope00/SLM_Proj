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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/data-cleansing-introduction/?type%3Darticle%26id%3D198887&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Machine Learning Models\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/machine-learning-models/)

# ML \| Overview of Data Cleaning

Last Updated : 20 Jan, 2025

Comments

Improve

Suggest changes

70 Likes

Like

Report

Data cleaning is a important step in the [machine learning (ML)](https://www.geeksforgeeks.org/machine-learning/) pipeline as it involves **identifying and removing any missing duplicate or irrelevant data**. The goal of data cleaning is to **ensure that the data is accurate, consistent and free of errors as raw data is often noisy, incomplete and inconsistent which can negatively impact the accuracy of model and its reliability of insights derived from it.** Professional data scientists usually invest a large portion of their time in this step because of the belief that

> **“Better data beats fancier algorithms”**

Clean datasets also helps in [EDA](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/) that enhance the interpretability of data so that right actions can be taken based on insights.

## How to Perform Data Cleanliness?

The process begins by thorough understanding data and its structure to identify issues like missing values, duplicates and outliers. Performing data cleaning involves a systematic process to identify and remove errors in a dataset. The following are essential steps to perform data cleaning.

![Data Cleaning - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/datacleaning.jpg)

Data Cleaning

- **Removal of Unwanted Observations**: Identify and remove irrelevant or redundant (unwanted) observations from the dataset. This step involves analyzing data entries for duplicate records, irrelevant information or data points that do not contribute to analysis and prediction. Removing them from dataset helps reducing noise and improving the overall quality of dataset.
- **Fixing Structure errors:** Address structural issues in the dataset such as inconsistencies in data formats or variable types. Standardize formats ensure uniformity in data structure and hence data consistency.
- **Managing outliers:** Outliers are those points that deviate significantly from dataset mean. Identifying and managing outliers significantly improve model accuracy as these extreme values influence analysis. Depending on the context decide whether to remove outliers or transform them to minimize their impact on analysis.
- **Handling Missing Data:** To handle missing data effectively we need to impute missing values based on statistical methods, removing records with missing values or employing advanced imputation techniques. Handling missing data helps preventing biases and maintaining the integrity of data.

Throughout the process documentation of changes is crucial for transparency and future reference. Iterative validation is done to test effectiveness of the data cleaning resulting in a refined dataset and can be used for meaningful analysis and insights.

## Python Implementation for Database Cleaning

Let’s understand each step for Database Cleaning, using [titanic dataset](https://media.geeksforgeeks.org/wp-content/uploads/20250114171103408125/Titanic-Dataset.csv). Below are the necessary steps:

- Import the necessary libraries
- Load the dataset
- Check the data information using df.info()

Python`
import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv('titanic.csv')
df.head()
`

**Output**:

```
PassengerId    Survived    Pclass    Name    Sex    Age    SibSp    Parch    Ticket    Fare    Cabin    Embarked
0    1    0    3    Braund, Mr. Owen Harris    male    22.0    1    0    A/5 21171    7.2500    NaN    S
1    2    1    1    Cumings, Mrs. John Bradley (Florence Briggs Th...    female    38.0    1    0    PC 17599    71.2833    C85    C
2    3    1    3    Heikkinen, Miss. Laina    female    26.0    0    0    STON/O2. 3101282    7.9250    NaN    S
3    4    1    1    Futrelle, Mrs. Jacques Heath (Lily May Peel)    female    35.0    1    0    113803    53.1000    C123    S
4    5    0    3    Allen, Mr. William Henry    male    35.0    0    0    373450    8.0500    NaN    S
```

### Data Inspection and Exploration

Let’s first understand the data by inspecting its structure and identifying missing values, outliers and inconsistencies and check the duplicate rows with below python code:

Python`
df.duplicated()
`

**Output**:

```
0      False
1      False
2      False
3      False
4      False
       ...
886    False
887    False
888    False
889    False
890    False
Length: 891, dtype: bool
```

**Check the data information using df.info()**

Python`
df.info()
`

**Output**:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

From the above data info we can see that Age and Cabin have an **unequal number of counts**. And some of the columns are categorical and have data type objects and some are integer and float values.

**Check the Categorical and Numerical Columns.**

Python`
# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)
`

**Output**:

```
Categorical columns : ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
Numerical columns : ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
```

#### Check the total number of Unique Values in the Categorical Columns

Python`
df[cat_col].nunique()
`

**Output**:

```
Name        891
Sex           2
Ticket      681
Cabin       147
Embarked      3
dtype: int64
```

### **Removal of all Above Unwanted Observations**

Duplicate observations most frequently arise during data collection and Irrelevant observations are those that don’t actually fit with the specific problem that we’re trying to solve.

- Redundant observations alter the efficiency to a great extent as the data repeats and may add towards the correct side or towards the incorrect side, therefore producing useless results.
- Irrelevant observations are any type of data that is of no use to us and can be removed directly.

**Now we have to make a decision according to the subject of analysis which factor is important for our discussion.**

As we know our machines don’t understand the text data. So we have to either drop or convert the categorical column values into numerical types. Here we are dropping the Name columns because the Name will be always unique and it hasn’t a great influence on target variables. For the ticket, Let’s first print the 50 unique tickets.

Python`
df['Ticket'].unique()[:50]
`

**Output**:

```
array(['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450',\
       '330877', '17463', '349909', '347742', '237736', 'PP 9549',\
       '113783', 'A/5. 2151', '347082', '350406', '248706', '382652',\
       '244373', '345763', '2649', '239865', '248698', '330923', '113788',\
       '347077', '2631', '19950', '330959', '349216', 'PC 17601',\
       'PC 17569', '335677', 'C.A. 24579', 'PC 17604', '113789', '2677',\
       'A./5. 2152', '345764', '2651', '7546', '11668', '349253',\
       'SC/Paris 2123', '330958', 'S.C./A.4. 23567', '370371', '14311',\
       '2662', '349237', '3101295'], dtype=object)
```

From the above tickets, we can observe that it is made of two like first values ‘A/5 21171’ is joint from of ‘A/5’ and  ‘21171’ this may influence our target variables. It will the case of **Feature Engineering**. where we derived new features from a column or a group of columns. In the current case, we are dropping the “Name” and “Ticket” columns.

**Drop Name and Ticket Columns**

Python`
df1 = df.drop(columns=['Name','Ticket'])
df1.shape
`

**Output**:

```
(891, 10)
```

### **Handling Missing Data**

Missing data is a common issue in real-world datasets and it can occur due to various reasons such as human errors, system failures or data collection issues. Various techniques can be used to handle missing data, such as imputation, deletion or substitution.

Let’s check the missing values columns-wise for each row using df.isnull() it checks whether the values are null or not and gives returns boolean values and sum() will sum the total number of null values rows and we divide it by the total number of rows present in the dataset then we multiply to get values in i.e per 100 values how much values are null.

Python`
round((df1.isnull().sum()/df1.shape[0])*100,2)
`

**Output**:

```
PassengerId     0.00
Survived        0.00
Pclass          0.00
Sex             0.00
Age            19.87
SibSp           0.00
Parch           0.00
Fare            0.00
Cabin          77.10
Embarked        0.22
dtype: float64
```

We cannot just ignore or remove the missing observation. They must be handled carefully as they can be an indication of something important.

- The fact that the value was missing may be informative in itself.
- In the real world we often need to make predictions on new data even if some of the features are missing!

As we can see from the above result that Cabin has 77% null values and Age has 19.87% and Embarked has 0.22% of null values.

So, it’s not a good idea to fill 77% of null values. So we will drop the Cabin column. Embarked column has only 0.22% of null values so, we drop the null values rows of Embarked column.

Python`
df2 = df1.drop(columns='Cabin')
df2.dropna(subset=['Embarked'], axis=0, inplace=True)
df2.shape
`

**Output**:

```
(889, 9)

```

Imputing the missing values from past observations.

- Again “missingness” is almost informative in itself and we should tell our algorithm if a value was missing.
- Even if we build a model to impute our values we’re not adding any real information. we’re just reinforcing the patterns already provided by other features. We can use **Mean imputation** or **Median imputations** for the case.

**Note:**

- Mean imputation is suitable when the data is normally distributed and has no extreme outliers.
- Median imputation is preferable when the data contains outliers or is skewed.

Python`
# Mean imputation
df3 = df2.fillna(df2.Age.mean())
# Let's check the null values again
df3.isnull().sum()
`

**Output**:

```
PassengerId    0
Survived       0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
Embarked       0
dtype: int64
```

### Handling Outliers

Outliers are extreme values that deviate significantly from the majority of the data. They can negatively impact the analysis and model performance. Techniques such as clustering, interpolation or transformation can be used to handle outliers.

To check the outliers we generally use a [box plo](https://www.geeksforgeeks.org/box-plot/) t. A box plot is a graphical representation of a dataset’s distribution. It shows a variable’s median, quartiles and potential outliers. The line inside the box denotes the median while the box itself denotes the [interquartile range (IQR)](https://www.geeksforgeeks.org/interquartile-range-iqr/). The box plot extend to the most extreme non-outlier values within 1.5 times the IQR. Individual points beyond the box are considered potential outliers. A box plot offers an easy-to-understand overview of the range of the data and makes it possible to identify outliers or skewness in the distribution.

**Let’s plot the box plot for Age column data.**

Python`
import matplotlib.pyplot as plt
plt.boxplot(df3['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()
`

**Output**:

![Box Plot - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230516164913/download-(3).png)

Box Plot

As we can see from the above Box and whisker plot, Our age dataset has outliers values. The values less than 5 and more than 55 are outliers.

Python`
# calculate summary statistics
mean = df3['Age'].mean()
std  = df3['Age'].std()
# Calculate the lower and upper bounds
lower_bound = mean - std*2
upper_bound = mean + std*2
print('Lower Bound :',lower_bound)
print('Upper Bound :',upper_bound)
# Drop the outliers
df4 = df3[(df3['Age'] >= lower_bound)\
                & (df3['Age'] <= upper_bound)]
`

**Output**:

```
Lower Bound : 3.705400107925648
Upper Bound : 55.578785285332785
```

Similarly, we can remove the outliers of the remaining columns.

### **Data Transformation**

Data transformation involves converting the data from one form to another to make it more suitable for analysis. Techniques such as normalization, scaling or encoding can be used to transform the data.

### **Data validation and verification**

Data validation and verification involve ensuring that the data is accurate and consistent by comparing it with external sources or expert knowledge.

For the machine learning prediction we separate independent and target features. Here we will consider only **‘Sex’ ‘Age’ ‘SibSp’, ‘Parch’ ‘Fare’ ‘Embarked’** only as the independent features and **Survived** as target variables because PassengerId will not affect the survival rate.

Python`
X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']
`

### **Data formatting**

Data formatting involves converting the data into a standard format or structure that can be easily processed by the algorithms or models used for analysis. Here we will discuss commonly used data formatting techniques i.e. Scaling and Normalization.

**Scaling**

- Scaling involves transforming the values of features to a specific range. It maintains the shape of the original distribution while changing the scale.
- Particularly useful when features have different scales, and certain algorithms are sensitive to the magnitude of the features.
- Common scaling methods include Min-Max scaling and Standardization (Z-score scaling).

**Min-Max Scaling**: Min-Max scaling rescales the values to a specified range, typically between 0 and 1. It preserves the original distribution and ensures that the minimum value maps to 0 and the maximum value maps to 1.

Python`
from sklearn.preprocessing import MinMaxScaler
# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Numerical columns
num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()
`

**Output**:

```
Pclass    Sex    Age    SibSp    Parch    Fare    Embarked
0    1.0    male    0.271174    0.125    0.0    0.014151    S
1    0.0    female    0.472229    0.125    0.0    0.139136    C
2    1.0    female    0.321438    0.000    0.0    0.015469    S
3    0.0    female    0.434531    0.125    0.0    0.103644    S
4    1.0    male    0.434531    0.000    0.0    0.015713    S
```

**Standardization (Z-score scaling):** Standardization transforms the values to have a mean of 0 and a standard deviation of 1. It centers the data around the mean and scales it based on the standard deviation. Standardization makes the data more suitable for algorithms that assume a Gaussian distribution or require features to have zero mean and unit variance.

```
Z = (X - μ) / σ
```

Where,

- X = Data
- μ = Mean value of X
- σ = Standard deviation of X

## **Data Cleansing Tools**

Some data cleansing tools **:**

- **OpenRefine**: A powerful open-source tool for cleaning and transforming messy data. It supports tasks like removing duplicate and data enrichment with easy-to-use interface.
- **Trifacta Wrangler:** A user-friendly tool designed for cleaning, transforming and preparing data for analysis. It uses AI to suggest transformations to streamline workflows.
- **TIBCO Clarity:** A tool that helps in profiling, standardizing and enriching data. It’s ideal to make high quality data and consistency across datasets.
- **Cloudingo:** A cloud-based tool focusing on de-duplication, data cleansing and record management to maintain accuracy of data.
- **IBM Infosphere Quality Stage:** It’s highly suitable for large-scale and complex data.

## Advantages and Disadvantages of Data Cleaning in Machine Learning

**Advantages:**

- **Improved model performance**: Removal of errors, inconsistencies and irrelevant data helps the model to better learn from the data.
- **Increased accuracy**: Helps ensure that the data is accurate, consistent and free of errors.
- **Better representation of the data**: Data cleaning allows the data to be transformed into a format that better represents the underlying relationships and patterns in the data.
- **Improved data quality:** Improve the quality of the data, making it more reliable and accurate.
- **Improved data security:** Helps to identify and remove sensitive or confidential information that could compromise data security.

**Disadvantages:**

- **Time-consuming**: It is very time consuming task specially for large and complex datasets.
- **Error-prone:** It can result in loss of important information.
- **Cost and resource-intensive:** It is resource-intensive process that requires significant time, effort and expertise. It can also require the use of specialized software tools.
- **Overfitting:** Data cleaning can contribute to overfitting by removing too much data.

So we have discussed four different steps in data cleaning to make the data more reliable and to produce good results. After properly completing the Data Cleaning steps, we’ll have a robust dataset that avoids any error and inconsistency. In summary, data cleaning is a crucial step in the data science pipeline that involves identifying and correcting errors, inconsistencies and inaccuracies in the data to improve its quality and usability.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Data Cleaning with NumPy

[Visit Course![explore course icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/Vector11.svg)](https://www.geeksforgeeks.org/courses/data-science-live)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/machine-learning-models/)

[Machine Learning Models](https://www.geeksforgeeks.org/machine-learning-models/)

[U](https://www.geeksforgeeks.org/user/utsavgoel/)

[utsavgoel](https://www.geeksforgeeks.org/user/utsavgoel/)

Follow

70

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Analysis](https://www.geeksforgeeks.org/category/ai-ml-ds/r-data-analysis/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

### Similar Reads

[Machine Learning Tutorial\\
\\
\\
Machine learning is a subset of Artificial Intelligence (AI) that enables computers to learn from data and make predictions without being explicitly programmed. If you're new to this field, this tutorial will provide a comprehensive understanding of machine learning, its types, algorithms, tools, an\\
\\
8 min read](https://www.geeksforgeeks.org/machine-learning/)

## Prerequisites for Machine Learning

[Python for Machine Learning\\
\\
\\
Welcome to "Python for Machine Learning," a comprehensive guide to mastering one of the most powerful tools in the data science toolkit. Python is widely recognized for its simplicity, versatility, and extensive ecosystem of libraries, making it the go-to programming language for machine learning. I\\
\\
6 min read](https://www.geeksforgeeks.org/python-for-machine-learning/)
[SQL for Machine Learning\\
\\
\\
Integrating SQL with machine learning can provide a powerful framework for managing and analyzing data, especially in scenarios where large datasets are involved. By combining the structured querying capabilities of SQL with the analytical and predictive capabilities of machine learning algorithms,\\
\\
6 min read](https://www.geeksforgeeks.org/sql-for-machine-learning/)

## Getting Started with Machine Learning

[Advantages and Disadvantages of Machine Learning\\
\\
\\
Machine learning (ML) has revolutionized industries, reshaped decision-making processes, and transformed how we interact with technology. As a subset of artificial intelligence ML enables systems to learn from data, identify patterns, and make decisions with minimal human intervention. While its pot\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-machine-learning/)
[Why ML is Important ?\\
\\
\\
Machine learning (ML) has become a cornerstone of modern technology, revolutionizing industries and reshaping the way we interact with the world. As a subset of artificial intelligence (AI), ML enables systems to learn and improve from experience without being explicitly programmed. Its importance s\\
\\
4 min read](https://www.geeksforgeeks.org/why-ml-is-important/)
[Real- Life Examples of Machine Learning\\
\\
\\
Machine learning plays an important role in real life, as it provides us with countless possibilities and solutions to problems. It is used in various fields, such as health care, financial services, regulation, and more. Importance of Machine Learning in Real-Life ScenariosThe importance of machine\\
\\
13 min read](https://www.geeksforgeeks.org/real-life-applications-of-machine-learning/)
[What is the Role of Machine Learning in Data Science\\
\\
\\
In today's world, the collaboration between machine learning and data science plays an important role in maximizing the potential of large datasets. Despite the complexity, these concepts are integral in unraveling insights from vast data pools. Let's delve into the role of machine learning in data\\
\\
9 min read](https://www.geeksforgeeks.org/role-of-machine-learning-in-data-science/)
[Top Machine Learning Careers/Jobs\\
\\
\\
Machine Learning (ML) is one of the fastest-growing fields in technology, driving innovations across healthcare, finance, e-commerce, and more. As companies increasingly adopt AI-based solutions, the demand for skilled ML professionals is Soaring. This article delves into the Type of Machine Learnin\\
\\
10 min read](https://www.geeksforgeeks.org/top-career-paths-in-machine-learning/)

Like70

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/data-cleansing-introduction/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1136484406.1745055229&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=798725463)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055228990&cv=11&fst=1745055228990&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&ptag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fdata-cleansing-introduction%2F%3Fref%3Dlbp&hn=www.googleadservices.com&frm=0&tiba=ML%20%7C%20Overview%20of%20Data%20Cleaning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=618530034.1745055229&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=cpl0gj5y173q)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=u4q4rq4idpvx)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=invisible&cb=riv677nwmpuq)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)