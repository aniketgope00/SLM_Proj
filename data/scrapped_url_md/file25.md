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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/python-data-visualization-tutorial/?type%3Darticle%26id%3D726850&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is Data Visualization and Why is It Important?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/data-visualization-and-its-importance/)

# Python – Data visualization tutorial

Last Updated : 26 Dec, 2024

Comments

Improve

Suggest changes

9 Likes

Like

Report

Data visualization is a crucial aspect of data analysis, helping to transform analyzed data into meaningful insights through graphical representations. This comprehensive tutorial will guide you through the fundamentals of data visualization using Python. We’ll explore various libraries, including **Matplotlib, Seaborn, Pandas, Plotly, Plotnine, Altair, Bokeh, Pygal, and Geoplotlib.** Each library offers unique features and advantages, catering to different visualization needs and preferences.

![Python--Data-Visualization-Tutorial](https://media.geeksforgeeks.org/wp-content/uploads/20240826131231/Python--Data-Visualization-Tutorial.webp)

Data visualization tutorial

## Introduction to Data Visualization

After analyzing data, it is important to visualize the data to uncover patterns, trends, outliers, and insights that may not be apparent in raw data using visual elements like charts, graphs, and maps. _**Choosing the right type of chart is crucial for effectively communicating your data. Different charts serve different purposes and can highlight various aspects of your data**_. For a deeper dive into selecting the best chart for your data, check out this comprehensive guide on:

- [What is Data Visualization and Why is It Important?](https://www.geeksforgeeks.org/data-visualization-and-its-importance/)
- [Types of Data Visualization Charts](https://www.geeksforgeeks.org/types-of-data-visualization/)
- [Choosing the Right Chart Type](https://www.geeksforgeeks.org/choosing-the-right-chart-type-a-technical-guide/)

Equally important is selecting the right colors for your visualizations. Proper color choices highlight key information, improve readability, and make visuals more engaging. For expert advice on choosing the best colors for your charts, visit [How to select Colors for Data Visualizations?](https://www.geeksforgeeks.org/how-to-select-colors-for-data-visualizations/)

## Python Libraries for Data Visualization

Python offers numerous [libraries for data visualization](https://www.geeksforgeeks.org/top-python-libraries-for-data-visualization/), each with unique features and advantages. Below are some of the most popular libraries:

Here are some of the most popular ones:

- **Matplotlib**
- **Seaborn**
- **Pandas**
- **Plotly**
- **Plotnine**
- **Altair**
- **Bokeh**
- **Pygal**
- **Geoplotlib**

## Getting Started – Data Visualization with Matplotlib

Matplotlib is a great way to begin visualizing data in Python, essential for data visualization in data science. It is a versatile library that designed to help users visualize data in a variety of formats. Well-suited for creating a wide range of static, animated, and interactive plots.

- [Introduction to Matplotlib](https://www.geeksforgeeks.org/python-introduction-matplotlib/)
- [Setting up Python Environment for installation](https://www.geeksforgeeks.org/install-matplotlib-python/)
- [Pyplot in Matplotlib](https://www.geeksforgeeks.org/pyplot-in-matplotlib/)
- [Matplotlib – Axes Class](https://www.geeksforgeeks.org/matplotlib-axes-class/)
- [Data Visualization With Matplotlib](https://www.geeksforgeeks.org/data-visualization-using-matplotlib/)

### Example: Plotting a Linear Relationship with Matplotlib

Python`
# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
# define data values
x = np.array([1, 2, 3, 4]) # X-axis points
y = x*2 # Y-axis points
plt.plot(x, y) # Plot the chart
plt.show() # display
`

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20201013170331/simpleplot-300x169.png)

## Effective Data Visualization With Seaborn

Seaborn is a Python library that simplifies **the creation of attractive and informative statistical graphics**. It integrates seamlessly with Pandas DataFrames and offers a range of functions tailored for visualizing statistical relationships and distributions. This chapter will guide you through using Seaborn to create effective data visualizations.

- [Data Visualization with Python Seaborn](https://www.geeksforgeeks.org/data-visualization-with-python-seaborn/)
- [Data visualization with Seaborn Pairplot](https://www.geeksforgeeks.org/data-visualization-with-pairplot-seaborn-and-pandas/)
- [Data Visualization with FacetGrid in Seaborn](https://www.geeksforgeeks.org/python-seaborn-facetgrid-method/)
- [Time Series Visualization with Seaborn : Line Plot](https://www.geeksforgeeks.org/data-visualization-with-seaborn-line-plot/)

### Example: Scatter Plot Analysis with Seaborn

Python`
import seaborn as sns
import matplotlib.pyplot as plt
# Load the 'tips' dataset
tips = sns.load_dataset('tips')
# Create a scatter plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time', style='time')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()
`

**Output:**

![datavisualizationwithseaborn](https://media.geeksforgeeks.org/wp-content/uploads/20240807185155/datavisualizationwithseaborn.png)

Data Visualization with Seaborn

## Data Visualization with Pandas

[Pandas](https://www.geeksforgeeks.org/introduction-to-pandas-in-python/) is a powerful data manipulation library in Python that also offers some basic data visualization capabilities. **While it may not be as feature-rich as dedicated visualization libraries like Matplotlib or Seaborn, Pandas’ built-in plotting is convenient for quick and simple visualizations.**

- [Data Visualization With Pandas](https://www.geeksforgeeks.org/pandas-built-in-data-visualization-ml/)
- [Visualizing Time Series Data with pandas](https://www.geeksforgeeks.org/how-to-plot-timeseries-based-charts-using-pandas/)
- [Plotting Geospatial Data using GeoPandas](https://www.geeksforgeeks.org/plotting-geospatial-data-using-geopandas/)

### Examples: Visualizing Spread and Outliers

[Box plots](https://www.geeksforgeeks.org/box-plot/) are useful for visualizing the spread and outliers in your data. They provide a graphical summary of the data distribution, highlighting the median, quartiles, and potential outliers. Let’s create box plot with Pandas:

Python`
# Sample data
data = {
    'Category': ['A']*10 + ['B']*10,
    'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
df = pd.DataFrame(data)
# Box plot
df.boxplot(by='Category')
plt.title('Box Plot Example')
plt.suptitle('')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
`

**Output:**

![boxplot](https://media.geeksforgeeks.org/wp-content/uploads/20240808113028/boxplot.png)

Box Plot

## Data Visualization with Plotly

Plotly is a versatile library for creating interactive and aesthetically pleasing visualizations. This chapter will introduce you to Plotly and guide you through creating basic visualizations.

- [Introduction to Plotly](https://www.geeksforgeeks.org/getting-started-with-plotly-python/)
- [Data Visualization with Plotly](https://www.geeksforgeeks.org/using-plotly-for-interactive-data-visualization-in-python/)

We’ll create a simple bar plot. For this example, we’ll use the same ‘tips’ dataset we used with Seaborn.

Python`
import plotly.express as px
import pandas as pd
tips = px.data.tips()
fig = px.bar(tips, x='day', y='total_bill', title='Average Total Bill per Day')
fig.show()
`

**Output:**

![barplot](https://media.geeksforgeeks.org/wp-content/uploads/20240808121232/barplot.webp)

Bar Plot Plotly

**Plotly allows for extensive customizations, including updating layouts, adding annotations, and incorporating dropdowns and sliders**.

## Data Visualization with Plotnine

Plotnine is a Python library that implements the **Grammar of Graphics, inspired by R’s ggplot2**. It provides a coherent and consistent way to create complex visualizations with minimal code.. This chapter will introduce you to Plotnine in Python, demonstrating how they can be used to create various types of plots.

- [Introduction to Concept of Grammar of Graphics](https://www.geeksforgeeks.org/an-introduction-to-grammar-of-graphics-for-python/)
- [Data Visualization using Plotnine](https://www.geeksforgeeks.org/data-visualization-using-plotnine-and-ggplot2-in-python/)

### Plotnine Example: Creating Line Plots

Python`
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_histogram, labs, theme_minimal
from plotnine.data import economics
# Load the 'economics' dataset available in Plotnine
# This dataset contains economic indicators including unemployment numbers
# Create a line plot to visualize the trend of unemployment rate over time
line_plot = (
    ggplot(economics, aes(x='date', y='unemploy'))
    + geom_line(color='blue')
    + labs(title='Unemployment Rate Over Time',
           x='Date', y='Number of Unemployed')
    + theme_minimal()
)
print(line_plot)
`

**Output:**

![Creating-Bar-Plots](https://media.geeksforgeeks.org/wp-content/uploads/20240808123539/Creating-Bar-Plots.webp)

Line Plots

## Data Visualizations with Altair

Altair is a declarative statistical visualization library for Python, designed to provide an intuitive way to create interactive and informative charts. Built on Vega and Vega-Lite, Altair allows users to build complex visualizations through simple and expressive syntax.

- [Data Visualization with Altair](https://www.geeksforgeeks.org/introduction-to-altair-in-python/)
- [Aggregating Data for Large Datasets](https://www.geeksforgeeks.org/using-altair-on-data-aggregated-from-large-datasets/)
- [Sharing and Publishing Visualizations with Altair](https://www.geeksforgeeks.org/sharing-and-publishing-visualizations-with-altair/)

### Altair Example: Creating Charts

Python`
# Import necessary libraries
import altair as alt
from vega_datasets import data
iris = data.iris()
# Create a scatter plot
scatter_plot = alt.Chart(iris).mark_point().encode(
    x='sepalLength',
    y='petalLength',
    color='species'
)
scatter_plot
`

**Output:**

![scatterplot](https://media.geeksforgeeks.org/wp-content/uploads/20240808125716/scatterplot.png)

Creating Charts

## Interactive Data Visualization with Bokeh

Bokeh is a powerful Python library for creating [interactive data visualization](https://www.geeksforgeeks.org/what-is-interactive-data-visualization/) and highly customizable visualizations. It is **designed for modern web browsers and allows for the creation of complex visualizations with ease**. Bokeh supports a wide range of plot types and interactivity features, making it a popular choice for interactive data visualization.

- [Introduction to Bokeh in Python](https://www.geeksforgeeks.org/introduction-to-bokeh-in-python/)
- [Interactive Data Visualization with Bokeh](https://www.geeksforgeeks.org/python-bokeh-tutorial-interactive-data-visualization-with-bokeh/)
- [Practical Examples for Mastering Data Visualization with Bokeh](https://www.geeksforgeeks.org/interactive-data-visualization-with-python-and-bokeh/)

### Example : Basic Plotting with Bokeh- Adding Hover Tool

Python`
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
p = figure(title="Scatter Plot with Hover Tool",
           x_axis_label='X-Axis', y_axis_label='Y-Axis')
p.scatter(x=[1, 2, 3, 4, 5], y=[6, 7, 2, 4, 5],
          size=10, color="green", alpha=0.5)
# Add HoverTool
hover = HoverTool()
hover.tooltips = [("X", "@x"), ("Y", "@y")]
p.add_tools(hover)
# Show the plot
show(p)
`

**Output:**

![Scatterplothovertool-ezgifcomoptimize](https://media.geeksforgeeks.org/wp-content/uploads/20240808150711/Scatterplothovertool-ezgifcomoptimize.gif)

Basic Plotting with Bokeh- Adding Hover Tool

## **Mastering Advanced Data Visualization with Pygal**

In this final chapter, we will delve into advanced techniques for data visualization using Pygal. It is known for its ease of use and ability to create beautiful, interactive charts that can be embedded in web applications.

- [Data Visualization with Pygal:](https://www.geeksforgeeks.org/data-visualization-with-pygal/?ref=next_article#:~:text=Pygal%20is%20a%20powerful%20tool,visualizing%20data%20in%20various%20ways.) With Pygal, you can create a wide range of charts including line charts, bar charts, pie charts, and more, all with interactive capabilities.

### Example: Creating Advanced Charts with Pygal

Firstly, you’ll need to install pygal, you can install it using pip:

```
pip install pygal
```

Python`
import pygal
from pygal.style import Style
# Create a custom style
custom_style = Style(
    background='transparent',
    plot_background='transparent',
    foreground='#000000',
    foreground_strong='#000000',
    foreground_subtle='#6e6e6e',
    opacity='.6',
    opacity_hover='.9',
    transition='400ms',
    colors=('#E80080', '#404040')
)
# Create a line chart
line_chart = pygal.Line(style=custom_style, show_legend=True,
                        x_title='Months', y_title='Values')
line_chart.title = 'Monthly Trends'
line_chart.add('Series 1', [1, 3, 5, 7, 9])
line_chart.add('Series 2', [2, 4, 6, 8, 10])
# Render the chart to a file
line_chart.render_to_file('line_chart.svg')
`

**Output:**

![line_chart](https://media.geeksforgeeks.org/wp-content/uploads/20240808152201/line_chart.webp)

Advanced Line Charts with Pygal

## Choosing the Right Data Visualization Library

| Library | Best For | Strengths | Limitations |
| --- | --- | --- | --- |
| Matplotlib | Static plots | Highly customizable | Steep learning curve |
| Seaborn | Statistical visualizations | Easy to use, visually appealing | Limited interactivity |
| Plotly | Interactive visualizations | Web integration, modern designs | Requires browser rendering |
| Bokeh | Web-based dashboards | Real-time interactivity | More complex setup |
| Altair | Declarative statistical plots | Concise syntax | Limited customization |
| Pygal | Scalable SVG charts | High-quality graphics | Less suited for complex datasets |

To create impactful and engaging data visualizations. **Start by selecting the appropriate chart type—bar charts for comparisons, line charts for trends, and pie charts for proportions.**

- Simplify your visualizations to focus on key insights.
- Use annotations to guide the viewer’s attention.
- Strategically use color to differentiate categories or highlight important data, but avoid overuse to prevent confusion.

For a more detailed exploration of these techniques consider below resources:

- [6 Tips for Creating Effective Data Visualizations](https://www.geeksforgeeks.org/6-tips-for-creating-effective-data-visualizations/)
- [Data Visualization in Infographics: Techniques and Examples](https://www.geeksforgeeks.org/data-visualization-in-infographics-techniques-and-examples/)
- [5 Best Practices for Effective and Good Data Visualizations](https://www.geeksforgeeks.org/5-best-practices-for-effective-and-good-data-visualizations/)
- [Bad Data Visualization Examples Explained](https://www.geeksforgeeks.org/bad-data-visualization-examples-explained/)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/data-visualization-and-its-importance/)

[What is Data Visualization and Why is It Important?](https://www.geeksforgeeks.org/data-visualization-and-its-importance/)

[![author](https://media.geeksforgeeks.org/auth/profile/8wh06px7rkj21jihprcr)](https://www.geeksforgeeks.org/user/abhishek1/)

[abhishek1](https://www.geeksforgeeks.org/user/abhishek1/)

Follow

9

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Visualization](https://www.geeksforgeeks.org/category/ai-ml-ds/r-data-visualization/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Python Data Visualization](https://www.geeksforgeeks.org/tag/python-data-visualization/)

### Similar Reads

[Python - Data visualization tutorial\\
\\
\\
Data visualization is a crucial aspect of data analysis, helping to transform analyzed data into meaningful insights through graphical representations. This comprehensive tutorial will guide you through the fundamentals of data visualization using Python. We'll explore various libraries, including M\\
\\
7 min read](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
[What is Data Visualization and Why is It Important?\\
\\
\\
Data visualization is the graphical representation of information. In this guide we will study what is Data visualization and its importance with use cases. Understanding Data VisualizationData visualizationÂ translates complexÂ data sets intoÂ visual formatsÂ that are easierÂ for the humanÂ brain to unde\\
\\
4 min read](https://www.geeksforgeeks.org/data-visualization-and-its-importance/)
[Data Visualization using Matplotlib in Python\\
\\
\\
Matplotlib is a powerful and widely-used Python library for creating static, animated and interactive data visualizations. In this article, we will provide a guide on Matplotlib and how to use it for data visualization with practical implementation. Matplotlib offers a wide variety of plots such as\\
\\
13 min read](https://www.geeksforgeeks.org/data-visualization-using-matplotlib/)
[Data Visualization with Seaborn - Python\\
\\
\\
Data visualization can be done by seaborn and it can transform complex datasets into clear visual representations making it easier to understand, identify trends and relationships within the data. This article will guide you through various plotting functions available in Seaborn. Getting Started wi\\
\\
13 min read](https://www.geeksforgeeks.org/data-visualization-with-python-seaborn/)
[Data Visualization with Pandas\\
\\
\\
Pandas allows to create various graphs directly from your data using built-in functions. This tutorial covers Pandas capabilities for visualizing data with line plots, area charts, bar plots, and more. Introducing Pandas for Data VisualizationPandas is a powerful open-source data analysis and manipu\\
\\
5 min read](https://www.geeksforgeeks.org/pandas-built-in-data-visualization-ml/)
[Plotly for Data Visualization in Python\\
\\
\\
Plotly is an open-source Python library for creating interactive visualizations like line charts, scatter plots, bar charts and more. In this article, we will explore plotting in Plotly and covers how to create basic charts and enhance them with interactive features. Introduction to Plotly in Python\\
\\
13 min read](https://www.geeksforgeeks.org/using-plotly-for-interactive-data-visualization-in-python/)
[Data Visualization using Plotnine and ggplot2 in Python\\
\\
\\
Plotnoine is a Python library that implements a grammar of graphics similar to ggplot2 in R. It allows users to build plots by defining data, aesthetics, and geometric objects. This approach provides a flexible and consistent method for creating a wide range of visualizations. It is built on the con\\
\\
7 min read](https://www.geeksforgeeks.org/data-visualization-using-plotnine-and-ggplot2-in-python/)
[Introduction to Altair in Python\\
\\
\\
Altair is a statistical visualization library in Python. It is a declarative in nature and is based on Vega and Vega-Lite visualization grammars. It is fast becoming the first choice of people looking for a quick and efficient way to visualize datasets. If you have used imperative visualization libr\\
\\
5 min read](https://www.geeksforgeeks.org/introduction-to-altair-in-python/)
[Python - Data visualization using Bokeh\\
\\
\\
Bokeh is a data visualization library in Python that provides high-performance interactive charts and plots. Bokeh output can be obtained in various mediums like notebook, html and server. It is possible to embed bokeh plots in Django and flask apps. Bokeh provides two visualization interfaces to us\\
\\
3 min read](https://www.geeksforgeeks.org/python-data-visualization-using-bokeh/)
[Pygal Introduction\\
\\
\\
Python has become one of the most popular programming languages for data science because of its vast collection of libraries. In data science, data visualization plays a crucial role that helps us to make it easier to identify trends, patterns, and outliers in large data sets. Pygal is best suited f\\
\\
5 min read](https://www.geeksforgeeks.org/pygal-introduction/)

Like9

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/python-data-visualization-tutorial/?utm_source=geeksforgeeks&utm_medium=gfgcontent_shm&utm_campaign=shm)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=739646818.1745055430&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=925397619)

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