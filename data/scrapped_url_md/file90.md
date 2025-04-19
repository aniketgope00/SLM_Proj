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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/?type%3Darticle%26id%3D649079&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Measures of Central Tendency in Statistics\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/measures-of-central-tendency/)

# What is Exploratory Data Analysis?

Last Updated : 13 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Exploratory Data Analysis (EDA) is an important first step in data science projects. It involves looking at and visualizing data to understand its main features, find patterns, and discover how different parts of the data are connected.

EDA helps to spot any unusual data or outliers and is usually done before starting more detailed statistical analysis or building models. In this article, we will discuss what is **Exploratory Data Analysis (EDA)** and the [steps to perform EDA](https://www.geeksforgeeks.org/steps-for-mastering-exploratory-data-analysis-eda-steps/).

![What-is-Exploratory-Data-Analysis](https://media.geeksforgeeks.org/wp-content/uploads/20240507132146/What-is-Exploratory-Data-Analysis.webp)

## Why Exploratory Data Analysis is Important?

Exploratory Data Analysis (EDA) is important for several reasons, especially in the context of data science and statistical modeling. Here are some of the key reasons why EDA is a critical step in the data analysis process:

- Helps to understand the dataset, showing how many features there are, the type of data in each feature, and how the data is spread out, which helps in choosing the right methods for analysis.
- EDA helps to identify hidden patterns and relationships between different data points, which help us in and model building.
- Allows to spot errors or unusual data points (outliers) that could affect your results.
- Insights that you obtain from EDA help you decide which features are most important for building models and how to prepare them to improve performance.
- By understanding the data, EDA helps us in choosing the best modeling techniques and adjusting them for better results.

## Types of Exploratory Data Analysis

There are various sorts of EDA strategies based on nature of the records. Depending on the number of columns we are analyzing we can divide EDA into three types: [Univariate, bivariate and multivariate](https://www.geeksforgeeks.org/univariate-bivariate-and-multivariate-data-and-its-analysis/).

### 1\. Univariate Analysis

Univariate analysis focuses on studying one variable to understand its characteristics. It helps describe the data and find patterns within a single feature. Common methods include **histograms** to show data distribution, **box plots** to detect outliers and understand data spread, and **bar charts** for categorical data. **Summary statistics** like [**mean**,](https://www.geeksforgeeks.org/mean/) [**median**](https://www.geeksforgeeks.org/median/), **mode**, [**variance**,](https://www.geeksforgeeks.org/variance/) and [**standard deviation**](https://www.geeksforgeeks.org/standard-deviation-formula/) help describe the central tendency and spread of the data

### **2\. Bivariate Analysis**

Bivariate analysis focuses on exploring the relationship between two variables to find connections, correlations, and dependencies. It’s an important part of exploratory data analysis that helps understand how two variables interact. Some key techniques used in bivariate analysis include **scatter plots**, which visualize the relationship between two continuous variables; **correlation coefficient**, which measures how strongly two variables are related, commonly using [**Pearson’s correlation**](https://www.geeksforgeeks.org/pearson-correlation-coefficient/) for linear relationships; and **cross-tabulation**, or **contingency tables**, which show the frequency distribution of two categorical variables and help understand their relationship.

[**Line graphs**](https://www.geeksforgeeks.org/line-graph/) are useful for comparing two variables over time, especially in time series data, to identify trends or patterns. [**Covariance**](https://www.geeksforgeeks.org/mathematics-covariance-and-correlation/) measures how two variables change together, though it’s often supplemented by the correlation coefficient for a clearer, more standardized view of the relationship.

### 3\. Multivariate Analysis

Multivariate analysis examines the relationships between two or more variables in the dataset. It aims to understand how variables interact with one another, which is crucial for most statistical modeling techniques. It include Techniques like [**pair plots**,](https://www.geeksforgeeks.org/python-seaborn-pairplot-method/) which show the relationships between multiple variables at once, helping to see how they interact. Another technique is [**Principal Component Analysis (PCA**](https://www.geeksforgeeks.org/principal-component-analysis-pca/) **)**, which reduces the complexity of large datasets by simplifying them, while keeping the most important information.

In addition to univariate and multivariate analysis, there are specialized EDA techniques tailored for specific types of data or analysis needs:

- [**Spatial Analysis**](https://www.geeksforgeeks.org/what-is-spatial-analysis/): For geographical data, using maps and spatial plotting to understand the geographical distribution of variables.
- **Text Analysis**: Involves techniques like word clouds, frequency distributions, and sentiment analysis to explore text data.
- [**Time Series Analysis**](https://www.geeksforgeeks.org/time-series-data-visualization-in-python/) **:** This type of analysis is mainly applied to statistics sets that have a temporal component. Time collection evaluation entails inspecting and modeling styles, traits, and seasonality inside the statistics through the years. Techniques like line plots, autocorrelation analysis, transferring averages, and [ARIMA](https://www.geeksforgeeks.org/python-arima-model-for-time-series-forecasting/) (AutoRegressive Integrated Moving Average) fashions are generally utilized in time series analysis.

## Steps for Performing Exploratory Data Analysis

Performing Exploratory Data Analysis (EDA) involves a series of steps designed to help you understand the data you’re working with, uncover underlying patterns, identify anomalies, test hypotheses, and ensure the data is clean and suitable for further analysis.

![Steps-for-Performing-Exploratory-Data-Analysis](https://media.geeksforgeeks.org/wp-content/uploads/20240509161456/Steps-for-Performing-Exploratory-Data-Analysis.png)

### Step 1: Understand the Problem and the Data

The first step in any data analysis project is to clearly understand the problem you’re trying to solve and the data you have. This involves asking key questions such as:

- **What is the business goal or research question**?
- **What are the variables in the data** and what do they represent?
- **What types of data** (numerical, categorical, text, etc.) do you have?
- Are there any known **data quality issues** or **limitations**?
- Are there any **domain-specific concerns** or restrictions?

By thoroughly understanding the problem and the data, you can better plan your analysis, avoid wrong assumptions, and ensure accurate conclusions

### Step 2: Import and Inspect the Data

After clearly understanding the problem and the data, the next step is to import the data into your analysis environment (like **Python**, **R**, or a spreadsheet tool). At this stage, it’s crucial to examine the data to get an initial understanding of its structure, variable types, and potential issues.

Here’s what you can do:

- **Load the data** into your environment carefully to avoid errors or truncations.
- **Examine the size** of the data (number of rows and columns) to understand its complexity.
- **Check for missing values** and see how they are distributed across variables, since missing data can impact the quality of your analysis.
- **Identify data types** for each variable (like numerical, categorical, etc.), which will help in the next steps of data manipulation and analysis.
- **Look for errors** or inconsistencies, such as invalid values, mismatched units, or outliers, which could signal deeper issues with the data.

By completing these tasks, you’ll be prepared to clean and analyze the data more effectively.

### Step 3: Handle Missing Data

Missing data is common in many datasets and can significantly affect the quality of your analysis. During **Exploratory Data Analysis (EDA)**, it’s important to identify and handle missing data properly to avoid biased or misleading results.

Here’s how to handle it:

- **Understand the patterns** and possible reasons for missing data. Is it **missing completely at random** (MCAR), **missing at random** (MAR), or **missing not at random** (MNAR)? Knowing this helps decide how to handle the missing data.
- **Decide whether to remove** missing data (listwise deletion) or **impute** (fill in) the missing values. Removing data can lead to biased outcomes, especially if the missing data isn’t MCAR. Imputing values helps preserve data but should be done carefully.
- Use appropriate **imputation methods** like **mean/median imputation**, [**regression**](https://www.geeksforgeeks.org/regression-in-machine-learning/) **imputation**, or machine learning techniques like [**KNN**](https://www.geeksforgeeks.org/k-nearest-neighbours/) or [**decision trees**](https://www.geeksforgeeks.org/decision-tree/) based on the data’s characteristics.
- **Consider the impact** of missing data. Even after imputing, missing data can cause uncertainty and bias, so interpret the results with caution.

Properly handling missing data improves the accuracy of your analysis and prevents misleading conclusions.

### Step 4: Explore Data Characteristics

After addressing missing data, the next step in **EDA** is to explore the characteristics of your data by examining the **distribution**, [**central tendency**](https://www.geeksforgeeks.org/measures-of-central-tendency/), and **variability** of your variables, as well as identifying any **outliers** or anomalies. This helps in selecting appropriate analysis methods and spotting potential data issues. You should calculate **summary statistics** like **mean**, **median**, **mode**, [**standard deviation**](https://www.geeksforgeeks.org/standard-deviation-formula/), [**skewness**](https://www.geeksforgeeks.org/skewness-measures-and-interpretation/), and [**kurtosis**](https://www.geeksforgeeks.org/how-to-calculate-kurtosis-in-statistics/) for numerical variables. These provide an overview of the data’s distribution and help identify any irregular patterns or issues.

### Step 5: Perform Data Transformation

Data transformation is an essential step in **EDA** because it prepares your data for accurate analysis and modeling. Depending on your data’s characteristics and analysis needs, you may need to transform it to ensure it’s in the right format.

Common transformation techniques include:

- **Scaling or normalizing** numerical variables (e.g., [**min-max scaling**](https://www.geeksforgeeks.org/data-pre-processing-wit-sklearn-using-standard-and-minmax-scaler/) or **standardization**).
- **Encoding categorical variables** for machine learning (e.g., [**one-hot encoding**](https://www.geeksforgeeks.org/ml-one-hot-encoding/) or **label encoding**).
- Applying **mathematical transformations** (e.g., **logarithmic** or **square root**) to correct skewness or non-linearity.
- **Creating new variables** from existing ones (e.g., calculating ratios or combining variables).
- [**Aggregating or grouping**](https://www.geeksforgeeks.org/grouping-and-aggregating-with-pandas/) data based on specific variables or conditions

### Step 6: Visualize Data Relationship

Visualization is a powerful tool in the **EDA** process, helping to uncover relationships between variables and identify patterns or trends that may not be obvious from summary statistics alone.

- For categorical variables, create **frequency tables**, **bar plots**, and **pie charts** to understand the distribution of categories and identify imbalances or unusual patterns.
- For numerical variables, generate **histograms**, **box plots**, **violin plots**, and **density plots** to visualize distribution, shape, spread, and potential outliers.
- To explore relationships between variables, use **scatter plots**, **correlation matrices**, or statistical tests like **Pearson’s correlation coefficient** or [**Spearman’s rank correlation**](https://www.geeksforgeeks.org/spearmans-rank-correlation/)

### Step 7: Handling Outliers

Outliers are data points that significantly differ from the rest of the data, often caused by errors in measurement or data entry. Detecting and handling outliers is important because they can skew your analysis and affect model performance. You can identify outliers using methods like [**interquartile range (IQR)**](https://www.geeksforgeeks.org/interquartile-range-iqr/), [**Z-scores**](https://www.geeksforgeeks.org/z-score-in-statistics/), or **domain-specific rules**. Once identified, outliers can be removed or adjusted depending on the context. Properly managing outliers ensures your analysis is accurate and reliable.

### Step 8: Communicate Findings and Insights

The final step in **EDA** is to communicate your findings clearly. This involves summarizing your analysis, pointing out key discoveries, and presenting your results in a clear and engaging way.

- Clearly state the **goals** and **scope** of your analysis.
- Provide **context** and background to help others understand your approach.
- Use **visualizations** to support your findings and make them easier to understand.
- Highlight key **insights**, **patterns**, or **anomalies** discovered.
- Mention any **limitations** or challenges faced during the analysis.
- Suggest **next steps** or areas that need further investigation.

Effective conversation is critical for ensuring that your EDA efforts have a meaningful impact and that your insights are understood and acted upon with the aid of stakeholders.

Exploratory Data Analysis (EDA) can be performed using a variety of tools and software, each offering features that deal to different data and analysis needs.

In **Python**, libraries like [**Pandas**](https://www.geeksforgeeks.org/pandas-tutorial/) are essential for data manipulation, providing functions to clean, filter, and transform data. [**Matplotlib**](https://www.geeksforgeeks.org/python-introduction-matplotlib/) is used for creating basic static, interactive, and animated visualizations, while [**Seaborn**](https://www.geeksforgeeks.org/introduction-to-seaborn-python/), built on top of Matplotlib, allows for the creation of more attractive and informative statistical plots. For interactive and advanced visualizations, [**Plotly**](https://www.geeksforgeeks.org/python-plotly-tutorial/) is an excellent choice

In **R**, packages like **ggplot2** are powerful for creating complex and visually appealing plots from data frames. **dplyr** helps in data manipulation, making tasks like filtering and summarizing easier, and **tidyr** ensures your data is in a tidy format, making it easier to work with.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/measures-of-central-tendency/)

[Measures of Central Tendency in Statistics](https://www.geeksforgeeks.org/measures-of-central-tendency/)

[N](https://www.geeksforgeeks.org/user/nikhilaggarwal3/)

[nikhilaggarwal3](https://www.geeksforgeeks.org/user/nikhilaggarwal3/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Analysis](https://www.geeksforgeeks.org/category/ai-ml-ds/r-data-analysis/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-EDA](https://www.geeksforgeeks.org/tag/ml-eda/)

### Similar Reads

[What is Exploratory Data Analysis?\\
\\
\\
Exploratory Data Analysis (EDA) is an important first step in data science projects. It involves looking at and visualizing data to understand its main features, find patterns, and discover how different parts of the data are connected. EDA helps to spot any unusual data or outliers and is usually d\\
\\
9 min read](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/)

## Univariate Data EDA

[Measures of Central Tendency in Statistics\\
\\
\\
Central Tendencies in Statistics are the numerical values that are used to represent mid-value or central value a large collection of numerical data. These obtained numerical values are called central or average values in Statistics. A central or average value of any statistical data or series is th\\
\\
10 min read](https://www.geeksforgeeks.org/measures-of-central-tendency/)
[Measures of Spread - Range, Variance, and Standard Deviation\\
\\
\\
Collecting the data and representing it in form of tables, graphs, and other distributions is essential for us. But, it is also essential that we get a fair idea about how the data is distributed, how scattered it is, and what is the mean of the data. The measures of the mean are not enough to descr\\
\\
9 min read](https://www.geeksforgeeks.org/measures-of-spread-range-variance-and-standard-deviation/)
[Interquartile Range and Quartile Deviation using NumPy and SciPy\\
\\
\\
In statistical analysis, understanding the spread or variability of a dataset is crucial for gaining insights into its distribution and characteristics. Two common measures used for quantifying this variability are the interquartile range (IQR) and quartile deviation. Quartiles Quartiles are a kind\\
\\
5 min read](https://www.geeksforgeeks.org/interquartile-range-and-quartile-deviation-using-numpy-and-scipy/)
[Anova Formula\\
\\
\\
ANOVA Test, or Analysis of Variance, is a statistical method used to test the differences between the means of two or more groups. Developed by Ronald Fisher in the early 20th century, ANOVA helps determine whether there are any statistically significant differences between the means of three or mor\\
\\
7 min read](https://www.geeksforgeeks.org/anova-formula/)
[Skewness of Statistical Data\\
\\
\\
Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. In simpler terms, it indicates whether the data is concentrated more on one side of the mean compared to the other side. Why is skewness important?Understanding the skewness of dat\\
\\
5 min read](https://www.geeksforgeeks.org/program-find-skewness-statistical-data/)
[How to Calculate Skewness and Kurtosis in Python?\\
\\
\\
Skewness is a statistical term and it is a way to estimate or measure the shape of a distribution. It is an important statistical methodology that is used to estimate the asymmetrical behavior rather than computing frequency distribution. Skewness can be two types: Symmetrical: A distribution can be\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-calculate-skewness-and-kurtosis-in-python/)
[Difference Between Skewness and Kurtosis\\
\\
\\
What is Skewness? Skewness is an important statistical technique that helps to determine the asymmetrical behavior of the frequency distribution, or more precisely, the lack of symmetry of tails both left and right of the frequency curve. A distribution or dataset is symmetric if it looks the same t\\
\\
4 min read](https://www.geeksforgeeks.org/difference-between-skewness-and-kurtosis/)
[Histogram \| Meaning, Example, Types and Steps to Draw\\
\\
\\
What is Histogram?A histogram is a graphical representation of the frequency distribution of continuous series using rectangles. The x-axis of the graph represents the class interval, and the y-axis shows the various frequencies corresponding to different class intervals. A histogram is a two-dimens\\
\\
5 min read](https://www.geeksforgeeks.org/histogram-meaning-example-types-and-steps-to-draw/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/?utm_source=geeksforgeeks&utm_medium=gfgcontent_shm&utm_campaign=shm)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1935958822.1745056709&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=513985370)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056709209&cv=11&fst=1745056709209&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fwhat-is-exploratory-data-analysis%2F%3Futm_source%3Dgeeksforgeeks%26utm_medium%3Dgfgcontent_shm%26utm_campaign%3Dshm&hn=www.googleadservices.com&frm=0&tiba=What%20is%20Exploratory%20Data%20Analysis%3F%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1666325072.1745056709&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)