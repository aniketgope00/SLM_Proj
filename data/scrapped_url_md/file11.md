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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/7-best-r-packages-for-machine-learning/?type%3Darticle%26id%3D515527&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Best Python libraries for Machine Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/)

# Best R Packages for Machine Learning

Last Updated : 16 Apr, 2025

Comments

Improve

Suggest changes

1 Like

Like

Report

[Machine Learning](https://www.geeksforgeeks.org/machine-learning/) is a subset of artificial intelligence that focuses on the development of computer software or programs that access data to learn from them and make predictions.

![7-Best-R-Packages-for-Machine-Learning-1](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20201124113453/7-Best-R-Packages-for-Machine-Learning-1.png)

[R language](https://www.geeksforgeeks.org/introduction-to-r-programming-language/) is being used in building machine learning models due to its flexibility, efficient packages and the ability to perform deep learning models with integration to the cloud. Being an open-source language, it offers multiple packages. Following are some famous R packages widely used in industry.

## 1\. data.table

[data.table](https://www.geeksforgeeks.org/data-manipulation-in-r-with-data-table/) package is a enhanced version of `data.frame` package and is designed for high-performance. It is known for its memory efficiency and ability to perform complex data manipulations at high speed. Some key features of **data.table** are:

- Fast file reading and writing
- Scalable data aggregation with parallelism support
- Feature-rich data reshaping
- Simplified syntax for subsetting and merging data

R`
install.packages("data.table")
library(data.table)
iris_dt <- as.data.table(iris)
result <- iris_dt[Species == "setosa" & Sepal.Length > 5][1:5]
result
`

**Output:**

![data_table](https://media.geeksforgeeks.org/wp-content/uploads/20250414165718926364/data_table.png)

Data Table

## 2\. Dplyr

[Dplyr](https://www.geeksforgeeks.org/dplyr-package-in-r-programming/) package is one of the most widely used data manipulation tools in R. It provides easy to implement and consistent set of functions to perform data transformations. The key functions in **dplyr** are:

- **select()**: Choose columns by name
- **filter()**: Subset rows based on conditions
- **arrange()**: Sort rows by column values
- **mutate()**: Add new variables

**Select and Mutate Functions :**

R`
install.packages("dplyr")  # Run only once
library(dplyr)
data("mtcars")
cat("---- Select ----\n")
selected <- dplyr::select(mtcars, mpg, cyl)
head(selected)
cat("\n------------------\n")
cat("---- Mutate ----\n")
mutated <- dplyr::mutate(mtcars, power_to_weight = hp / wt)
head(mutated)
cat("\n------------------\n")
`

**Output:**

![select-and-filter](https://media.geeksforgeeks.org/wp-content/uploads/20250414165854372633/select-and-filter.png)

Select and Mutate

**Filter and Arrange Functions :**

Python`
cat("---- Filter ----\n")
filtered <- dplyr::filter(mtcars, cyl == 6)
head(filtered)
cat("\n------------------\n")
cat("---- Arrange ----\n")
arranged <- dplyr::arrange(mtcars, desc(mpg))
head(arranged)
cat("\n------------------\n")
`

**Output:**

![Filter-](https://media.geeksforgeeks.org/wp-content/uploads/20250414165937131244/Filter-.png)

Filter and Arrange

## 3\. ggplot2

[ggplot2](https://www.geeksforgeeks.org/using-ggplot2-package-in-r-programming/) is an open-source visualization package based on the Grammar of Graphics. It is widely regarded as one of the most famous and flexible visualization libraries in R. With **ggplot2** users can create a wide range of static and interactive visualizations including:

- Bar charts
- Scatter plots
- Line graphs
- Histograms
- Boxplots

The syntax is easy and visualizations are highly customizable making it go-to package for data visualization in R.

R`
install.packages("dplyr")
install.packages("ggplot2")
library(dplyr)
library(ggplot2)
ggplot(data = mtcars,
       aes(x = hp, y = mpg,
           col = disp)) + geom_point()
`

**Output:**

![Output](https://media.geeksforgeeks.org/wp-content/uploads/20201009005648/Aes3.png)

## 4\. caret

[caret](https://www.geeksforgeeks.org/how-to-use-different-algorithms-using-caret-package-in-r/) package (Classification and Regression Training) provides a comprehensive framework for building machine learning models in R. It includes tools for:

- Data splitting
- Preprocessing
- Feature selection
- Model training
- Model evaluation

**caret** supports numerous machine learning algorithms and is commonly used in industry due to its ease of use and flexibility.

R`
install.packages("e1071")
install.packages("caTools")
install.packages("caret")
library(e1071)
library(caTools)
library(caret)
data(iris)
split <- sample.split(iris, SplitRatio = 0.7)
train_cl <- subset(iris, split == TRUE)
test_cl <- subset(iris, split == FALSE)
train_scale <- scale(train_cl[, 1:4])
test_scale <- scale(test_cl[, 1:4])
set.seed(120)
classifier_cl <- naiveBayes(Species ~ ., data = train_cl)
y_pred <- predict(classifier_cl, newdata = test_cl)
cm <- table(test_cl$Species, y_pred)
print(classifier_cl)
print(cm)
`

**Output:**

- **Model classifier\_cl:**

![nb](https://media.geeksforgeeks.org/wp-content/uploads/20250414170629054511/nb.png)

Navie Bayers Model

- **Confusion Matrix**

![caret_cm](https://media.geeksforgeeks.org/wp-content/uploads/20250414171230268146/caret_cm.png)

caret\_cm

## 5\. e1071

[**e1071**](https://www.geeksforgeeks.org/package-e1071-in-r/) package is known for its implementation of various machine learning algorithms including support vector machines (SVM), clustering algorithms and K-Nearest Neighbors (KNN). It is widely used for classification, regression and clustering tasks.

R`
install.packages("e1071")
install.packages("caTools")
install.packages("class")
library(e1071)
library(caTools)
library(class)
data(iris)
split <- sample.split(iris, SplitRatio = 0.7)
train_cl <- subset(iris, split == TRUE)
test_cl <- subset(iris, split == FALSE)
train_scale <- scale(train_cl[, 1:4])
test_scale <- scale(test_cl[, 1:4])
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 1)
cm <- table(test_cl$Species, classifier_knn)
print(cm)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1 - misClassError))
`

**Outputs:**

![KNN](https://media.geeksforgeeks.org/wp-content/uploads/20250414170559063448/KNN.png)

KNN

## 6\. XGBoost

[XGBoost](https://www.geeksforgeeks.org/xgboost-in-r-programming/) is a implementation of gradient boosting algorithms and is useful for large datasets. It is widely used in machine learning due to its performance and scalability. **XGBoost** works by bagging and boosting techniques to improve model accuracy.

R`
install.packages("data.table")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("xgboost")
install.packages("e1071")
install.packages("cowplot")
library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)
library(e1071)
library(cowplot)
test[, Item_Outlet_Sales := NA]
combi = rbind(train, test)
missing_index = which(is.na(combi$Item_Weight))
for(i in missing_index){
item = combi$Item_Identifier[i]
combi$Item_Weight[i] = mean(combi$Item_Weight[combi$Item_Identifier == item], na.rm = T)
}
zero_index = which(combi$Item_Visibility == 0)
for(i in zero_index){
item = combi$Item_Identifier[i]
combi$Item_Visibility[i] = mean(combi$Item_Visibility[combi$Item_Identifier == item], na.rm = T)
}
combi[, Outlet_Size_num := ifelse(Outlet_Size == "Small", 0, ifelse(Outlet_Size == "Medium", 1, 2))]
combi[, Outlet_Location_Type_num := ifelse(Outlet_Location_Type == "Tier 3", 0, ifelse(Outlet_Location_Type == "Tier 2", 1, 2))]
combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]
ohe_1 = dummyVars("~.", data = combi[, -c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T)
ohe_df = data.table(predict(ohe_1, combi[, -c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")]))
combi = cbind(combi[, "Item_Identifier"], ohe_df)
skewness(combi$Item_Visibility)
skewness(combi$price_per_unit_wt)
combi[, Item_Visibility := log(Item_Visibility + 1)]
num_vars = which(sapply(combi, is.numeric))
num_vars_names = names(num_vars)
combi_numeric = combi[, setdiff(num_vars_names, "Item_Outlet_Sales"), with = F]
prep_num = preProcess(combi_numeric, method = c("center", "scale"))
combi_numeric_norm = predict(prep_num, combi_numeric)
combi[, setdiff(num_vars_names, "Item_Outlet_Sales") := NULL]
combi = cbind(combi, combi_numeric_norm)
train = combi[1:nrow(train)]
test = combi[(nrow(train) + 1):nrow(combi)]
test[, Item_Outlet_Sales := NULL]
param_list = list(
objective = "reg:linear",
eta = 0.01,
gamma = 1,
max_depth = 6,
subsample = 0.8,
colsample_bytree = 0.5
)
Dtrain = xgb.DMatrix(data = as.matrix(train[, -c("Item_Identifier", "Item_Outlet_Sales")]), label = train$Item_Outlet_Sales)
Dtest = xgb.DMatrix(data = as.matrix(test[, -c("Item_Identifier")]))
set.seed(112)
xgbcv = xgb.cv(params = param_list, data = Dtrain, nrounds = 1000, nfold = 5, print_every_n = 10, early_stopping_rounds = 30, maximize = F)
xgb_model = xgb.train(data = Dtrain, params = param_list, nrounds = 428)
xgb_model
`

**Output:**

![xgb](https://media.geeksforgeeks.org/wp-content/uploads/20250414170859826036/xgb.png)

XGBoost Model

## 7\. randomForest

[Random Forest in R](https://www.geeksforgeeks.org/random-forest-approach-in-r-programming/) Programming is an ensemble learning method that builds multiple decision trees and combines them to provide more accurate predictions. It is especially useful for classification and regression tasks. Each decision tree is trained on a subset of the data and predictions are made by aggregating the results of all trees.

R`
install.packages("caTools")
install.packages("randomForest")
library(caTools)
library(randomForest)
data(iris)
split <- sample.split(iris, SplitRatio = 0.7)
train <- subset(iris, split == "TRUE")
test <- subset(iris, split == "FALSE")
set.seed(120)
classifier_RF = randomForest(x = train[-5], y = train$Species, ntree = 500)
classifier_RF
y_pred = predict(classifier_RF, newdata = test[-5])
cm = table(test[, 5], y_pred)
cm
`

**Outputs:**

- **Model classifier\_RF:**

![Screenshot-2025-04-16-164956](https://media.geeksforgeeks.org/wp-content/uploads/20250416165048009539/Screenshot-2025-04-16-164956.png)

Random Forest

- **Confusion Matrix:**

![rfcom](https://media.geeksforgeeks.org/wp-content/uploads/20250414171209110851/rfcom.png)

Confusion Matrix

R provides many packages that are widely used for data manipulation, visualization and machine learning. These packages discussed in this article are just few packages available in R for data scientists and researchers.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/)

[Best Python libraries for Machine Learning](https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/)

[D](https://www.geeksforgeeks.org/user/dhruv5819/)

[dhruv5819](https://www.geeksforgeeks.org/user/dhruv5819/)

Follow

1

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [AI-ML-DS Blogs](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science-blogs/)
- [GBlog](https://www.geeksforgeeks.org/category/guestblogs/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [R Language](https://www.geeksforgeeks.org/category/programming-language/r-language/)
- [R Machine-Learning](https://www.geeksforgeeks.org/tag/r-machine-learning/)

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

Like1

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/7-best-r-packages-for-machine-learning/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=407386293.1745055216&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=19736484)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055216502&cv=11&fst=1745055216502&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2F7-best-r-packages-for-machine-learning%2F%3Fref%3Dlbp&hn=www.googleadservices.com&frm=0&tiba=Best%20R%20Packages%20for%20Machine%20Learning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1004085618.1745055217&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)