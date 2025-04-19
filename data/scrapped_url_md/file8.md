- [Number System and Arithmetic](https://www.geeksforgeeks.org/number-theory/)
- [Algebra](https://www.geeksforgeeks.org/algebra/)
- [Set Theory](https://www.geeksforgeeks.org/set-theory/)
- [Probability](https://www.geeksforgeeks.org/probability-in-maths/)
- [Statistics](https://www.geeksforgeeks.org/statistics/)
- [Geometry](https://www.geeksforgeeks.org/geometry/)
- [Calculus](https://www.geeksforgeeks.org/math-calculus/)
- [Logarithms](https://www.geeksforgeeks.org/logarithms/)
- [Mensuration](https://www.geeksforgeeks.org/mensuration/)
- [Matrices](https://www.geeksforgeeks.org/matrices/)
- [Trigonometry](https://www.geeksforgeeks.org/math-trigonometry/)
- [Mathematics](https://www.geeksforgeeks.org/maths/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/statistics-for-machine-learning/?type%3Darticle%26id%3D1219808&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Maths for Machine Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/machine-learning-mathematics/)

# Statistics For Machine Learning

Last Updated : 06 Aug, 2024

Comments

Improve

Suggest changes

4 Likes

Like

Report

**Machine Learning Statistics:** In the field of machine learning (ML), statistics plays a pivotal role in extracting meaningful insights from data to make informed decisions. Statistics provides the foundation upon which various ML algorithms are built, enabling the analysis, interpretation, and prediction of complex patterns within datasets.

This article delves into the significance of statistics in machine learning and explores its applications across different domains.

![Statistics-For-Machine-Learning](https://media.geeksforgeeks.org/wp-content/uploads/20240417130843/Statistics-For-Machine-Learning.webp)Machine Learning Statistics

Table of Content

- [What is Statistics?](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#what-is-statistics)
- [What is Machine Learning?](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#what-is-machine-learning)
- [Applications of Statistics in Machine Learning](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#applications-of-statistics-in-machine-learning)
- [Types of Statistics](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#types-of-statistics)
- [Descriptive Statistics](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#descriptive-statistics)

  - [Measures of Dispersion](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#measures-of-dispersion)
  - [Measures of Shape](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#measures-of-shape)

- [Covariance and Correlation](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#covariance-and-correlation)

  - [Visualization Techniques](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#visualization-techniques)

- [Probability Theory](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#probability-theory)
- [Inferential Statistics](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#inferential-statistics)

  - [Population and Sample](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#population-and-sample)
  - [Estimation](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#estimation)
  - [Hypothesis Testing](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#hypothesis-testing)
  - [ANOVA (Analysis of Variance):](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#anova-analysis-of-variance)
  - [Chi-Square Tests:](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#chisquare-tests)
  - [Correlation and Regression:](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#correlation-and-regression)
  - [Bayesian Statistics](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp#bayesian-statistics)

## **What is Statistics?**

[Statistics](https://www.geeksforgeeks.org/statistics/) is the science of collecting, organizing, analyzing, interpreting, and presenting data. It encompasses a wide range of techniques for summarizing data, making inferences, and drawing conclusions.

Statistical methods help quantify uncertainty and variability in data, allowing researchers and analysts to make data-driven decisions with confidence.

## **What is Machine Learning?**

[Machine learning](https://www.geeksforgeeks.org/machine-learning/) is a branch of artificial intelligence (AI) that focuses on developing algorithms and models capable of learning from data without being explicitly programmed.

ML algorithms learn patterns and relationships from data, which they use to make predictions or decisions. Machine learning encompasses various techniques, including supervised learning, unsupervised learning, and reinforcement learning.

## Applications of Statistics in Machine Learning

Statistics is a key component of machine learning, with broad applicability in various fields.

- Feature engineering relies heavily on statistics to convert geometric features into meaningful predictors for [machine learning algorithms.](https://www.geeksforgeeks.org/machine-learning-algorithms/)

- In image processing tasks like object recognition and segmentation, statistics accurately reflect the shape and structure of objects in images.

- Anomaly detection and quality control benefit from statistics by identifying deviations from norms, aiding in the detection of defects in manufacturing processes.

- Environmental observation and geospatial mapping leverage statistical analysis to monitor land cover patterns and ecological trends effectively.

Overall, statistics plays a crucial role in machine learning, driving insights and advancements across diverse industries and applications.

## Types of Statistics

There are commonly two types of statistics, which are discussed below:

- **Descriptive Statistics:** " [De­scriptive Statistics](https://www.geeksforgeeks.org/descriptive-statistic/)" helps us simplify and organize big chunks of data. This makes large amounts of data easier to understand.
- **Inferential Statistics:** " [Inferential Statistics](https://www.geeksforgeeks.org/difference-between-descriptive-and-inferential-statistics/)" is a little different. It uses smaller data to draw conclusions about a larger group. It helps us predict and draw conclusions about a population.

## Descriptive Statistics

Descriptive statistics summarize and describe the features of a dataset, providing a foundation for further statistical analysis.

| Mean | Median | Mode |
| --- | --- | --- |
| [Mean](https://www.geeksforgeeks.org/mean/) is calculated by summing all values present in the sample divided by total number of values present in the sample. <br>Mean(μ)=SumofValuesNumberofValuesMean (\\mu) = \\frac{Sum \\, of \\, Values}{Number \\, of \\, Values}  <br>Mean(μ)=NumberofValuesSumofValues​ | [Median](https://www.geeksforgeeks.org/median/) is the middle of a sample when arranged from lowest to highest or highest to lowest. in order to find the median, the data must be sorted.<br>For odd number of data points: <br>Median=(n+12)thMedian = (\\frac{n+1}{2})^{th}<br>Median=(2n+1​)th<br>For even number of data points: <br>Median=Averageof(n2)thvalueanditsnextvalueMedian = Average \\, of \\, (\\frac{n}{2})^{th} value \\, and \\, its \\, next \\, value<br>Median=Averageof(2n​)thvalueanditsnextvalue | [Mode](https://www.geeksforgeeks.org/mode/) is the most frequently occurring value in the dataset. |

### **Measures of Dispersion**

- **Range:** The difference between the maximum and minimum values.
- [**Variance**](https://www.geeksforgeeks.org/variance/) **:** The average squared deviation from the mean, representing data spread.
- [**Standard Deviation**](https://www.geeksforgeeks.org/variance-and-standard-deviation/) **:** The square root of variance, indicating data spread relative to the mean.
- [**Interquartile Range**](https://www.geeksforgeeks.org/interquartile-range/) **:** The range between the first and third quartiles, measuring data spread around the median.

### **Measures of Shape**

- [**Skewness**](https://www.geeksforgeeks.org/skewness-measures-and-interpretation/) **:** Indicates data asymmetry.

![](https://media.geeksforgeeks.org/wp-content/uploads/20240130115716/Skewness.png)Types of Skewed data

- [**Kurtosis**](https://www.geeksforgeeks.org/difference-between-skewness-and-kurtosis/) **:** Measures the peakedness of the data distribution.

![](https://media.geeksforgeeks.org/wp-content/uploads/20240130115446/Kurtosis.png)Types of Skewed data

## Covariance and Correlation

| Covariance | Correlation |
| --- | --- |
| [Covariance](https://www.geeksforgeeks.org/covariance-and-correlation-in-r-programming/) measures the degree to which two variables change together.<br>Cov(x,y)=∑(Xi−X‾)(Yi−Y‾)nCov(x,y) = \\frac{\\sum(X\_i-\\overline{X})(Y\_i - \\overline{Y})}{n}<br>Cov(x,y)=n∑(Xi​−X)(Yi​−Y)​ | [Correlation](https://www.geeksforgeeks.org/how-to-find-correlation-coefficient-in-excel/) measures the strength and direction of the linear relationship between two variables. It is represented by correlation coefficient which ranges from -1 to 1. A positive correlation indicates a direct relationship, while a negative correlation implies an inverse relationship. <br>Pearson's correlation coefficient is given by:<br>ρ(X,Y)=cov(X,Y)σXσY\\rho(X, Y) = \\frac{cov(X,Y)}{\\sigma\_X \\sigma\_Y}<br>ρ(X,Y)=σX​σY​cov(X,Y)​ |

### **Visualization Techniques**

- **Histograms:** Show data distribution.
- **Box Plots:** Highlight data spread and potential outliers.
- **Scatter Plots:** Illustrate relationships between variables.

## Probability Theory

Probability theory forms the backbone of [statistical inference](https://www.geeksforgeeks.org/statistical-inference/#:~:text=Statistical%20inference%20is%20the%20process,on%20data%20from%20a%20sample.), aiding in quantifying uncertainty and making predictions based on data.

**Basic Concepts**

- [**Random Variables**](https://www.geeksforgeeks.org/random-variable/) **:** Variables with random outcomes.
- [**Probability Distributions**](https://www.geeksforgeeks.org/probability-distribution/) **:** Describe the likelihood of different outcomes.

**Common Probability Distributions**

- [**Binomial Distribution**](https://www.geeksforgeeks.org/what-is-binomial-probability-distribution-with-example/) **:** Represents the number of successes in a fixed number of trials.
- [**Poisson Distribution**](https://www.geeksforgeeks.org/poisson-distribution/) **:** Describes the number of events occurring within a fixed interval.
- [**Normal Distribution**](https://www.geeksforgeeks.org/normal-distribution/) **:** Characterizes continuous data symmetrically distributed around the mean.

[**Law of Large Numbers**:](https://www.geeksforgeeks.org/law-of-large-numbers/)

States that as the sample size increases, the sample mean approaches the population mean.

[**Central Limit Theorem**:](https://www.geeksforgeeks.org/central-limit-theorem-in-machine-learning/)

Indicates that the distribution of sample means approximates a normal distribution as the sample size grows, regardless of the population's distribution.

## Inferential Statistics

Inferential statistics involve making predictions or inferences about a population based on a sample of data.

### **Population and Sample**

- [**Population**](https://www.geeksforgeeks.org/population-and-sample-statistic/) **:** The entire group being studied.
- **Sample:** A subset of the population used for analysis.

### **Estimation**

- [**Point Estimation**](https://www.geeksforgeeks.org/point-estimation/) **:** Provides a single value estimate of a population parameter.
- [**Interval Estimation**](https://www.geeksforgeeks.org/confidence-interval/) **:** Offers a range of values (confidence interval) within which the parameter likely lies.
- [**Confidence Intervals**](https://www.geeksforgeeks.org/confidence-interval/) **:** Indicate the reliability of an estimate.

### [**Hypothesis Testing**](https://www.geeksforgeeks.org/understanding-hypothesis-testing/)

- **Null and Alternative Hypotheses:** The null hypothesis assumes no effect or relationship, while the alternative suggests otherwise.
- **Type I and Type II Errors:** Type I error is rejecting a true null hypothesis, while Type II is failing to reject a false null hypothesis.
- **p-Values:** Measure the probability of obtaining the observed results under the null hypothesis.
- **t-Tests and z-Tests:** Compare means to assess statistical significance.

### [**ANOVA (Analysis of Variance)**:](https://www.geeksforgeeks.org/anova-formula/)

Compares means across multiple groups to determine if they differ significantly.

### [**Chi-Square Tests**:](https://www.geeksforgeeks.org/chi-square-test/)

Assess the association between categorical variables.

### Correlation and Regression:

Understanding relationships between variables is critical in machine learning.

**Correlation**

- [**Pearson Correlation Coefficient**](https://www.geeksforgeeks.org/pearson-correlation-coefficient/) **:** Measures linear relationship strength between two variables.
- [**Spearman Rank Correlation**](https://www.geeksforgeeks.org/spearmans-rank-correlation/) **:** Assesses the strength and direction of the monotonic relationship between variables.

**Regression Analysis**

- [**Simple Linear Regression**](https://www.geeksforgeeks.org/ml-linear-regression/) **:** Models the relationship between two variables.
- [**Multiple Linear Regression**](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/) **:** Extends to multiple predictors.
- [**Assumptions of Linear Regression**](https://www.geeksforgeeks.org/assumptions-of-linear-regression/) **:** Linearity, independence, homoscedasticity, normality.
- [**Interpretation of Regression Coefficients**](https://www.geeksforgeeks.org/regression-coefficients/) **:** Explains predictor influence on the response variable.
- [**Model Evaluation Metrics**](https://www.geeksforgeeks.org/metrics-for-machine-learning-model/) **:** R-squared, Adjusted R-squared, RMSE.

### Bayesian Statistics

Bayesian statistics incorporate prior knowledge with current evidence to update beliefs.

[Bayes' Theorem](https://www.geeksforgeeks.org/bayes-theorem/) is a fundamental concept in probability theory that relates conditional probabilities. It is named after the Reverend Thomas Bayes, who first introduced the theorem. Bayes' Theorem is a mathematical formula that provides a way to update probabilities based on new evidence. The formula is as follows:

_P_( _A_ ∣ _B_)= _P_( _B_) _P_( _B_ ∣ _A_)⋅ _P_( _A_)​, where

- _P_( _A_ ∣ _B_): The probability of event A given that event B has occurred (posterior probability).
- _P_( _B_ ∣ _A_): The probability of event B given that event A has occurred (likelihood).
- _P_( _A_): The probability of event A occurring (prior probability).
- _P_( _B_): The probability of event B occurring.

**Related article:**

> - [Difference between Statistical Model and Machine Learning](https://www.geeksforgeeks.org/difference-between-statistical-model-and-machine-learning/)
> - [Machine Learning Mathematics](https://www.geeksforgeeks.org/machine-learning-mathematics/)
> - [Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/)
> - [7 Basic Statistics Concepts For Data Science](https://www.geeksforgeeks.org/7-basic-statistics-concepts-for-data-science/)

## **Conclusion**

Statistics is the foundation of machine learning, allowing for the extraction of useful insights from data across multiple domains. Machine learning algorithms can use statistical techniques and methodologies to learn from data, generate predictions, and solve complicated problems successfully. Understanding the significance of statistics in machine learning is critical for practitioners and researchers who want to use the power of [data-driven decision](https://www.geeksforgeeks.org/what-is-data-driven-decision-making/)-making in their domains.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/machine-learning-mathematics/)

[Maths for Machine Learning](https://www.geeksforgeeks.org/machine-learning-mathematics/)

[D](https://www.geeksforgeeks.org/user/deepanshukq4p4/)

[deepanshukq4p4](https://www.geeksforgeeks.org/user/deepanshukq4p4/)

Follow

4

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Mathematics](https://www.geeksforgeeks.org/category/school-learning/maths/)
- [School Learning](https://www.geeksforgeeks.org/category/school-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Statistics](https://www.geeksforgeeks.org/tag/statistics/)
- [Real Life Application](https://www.geeksforgeeks.org/tag/real-life-application/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

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

Like4

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/statistics-for-machine-learning/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2125516370.1745054631&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025~103130495~103130497&z=1047569449)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=f20hjm9lful3)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=ki685z99vvvb)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=invisible&cb=9pd02960kod)