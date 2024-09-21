

# Course Textbook: Week 03 Data Visualization and Populations and Sampling


**TUT/HW Topics**

1. [more precise data types (as opposed to object types)](week-03-Data-Visualization#continuous-discrete-nominal-and-ordinal-categorical-and-binary)... continuous, discrete, nominal and ordinal categorical, and binary
2. [bar plots](week-03-Data-Visualization#Bar-plots-and-modes) and the [mode](week-03-Data-Visualization#Bar-plots-and-modes)
3. [histograms](week-03-Data-Visualization#Histograms)
4. [box plots](week-03-Data-Visualization#Box-plots-and-spread), [range](week-03-Data-Visualization#Box-plots-and-spread), [IQR](week-03-Data-Visualization#Box-plots-and-spread) and [outliers](week-03-Data-Visualization#Box-plots-and-spread)
5. [skew and multimodality](week-03-Data-Visualization#skew-and-multimodality)
    1. [mean versus median](week-03-Data-Visualization#skew-and-multimodality)
    2. [normality and standard deviations](week-03-Data-Visualization#normal-distributions)
    
**LEC Extensions**

> Topic numbers below correspond to extensions of topic items above.

2\. [plotting... plotly, VS seaborn, VS matplotlib, VS pandas vs. ... ?](week-03-Data-Visualization#modern-plotting)\
___ i\. legends, annotations, figure panels, etc.\
3\. [kernel density estimation "violin plots"](week-03-Data-Visualization#smoothed-histograms)\
5\. [log transformations](week-03-Data-Visualization#log-transformations)

**LEC New Topics**

1. populations [_from scipy import stats_](week-03-Data-Visualization#Populations) (re: `stats.multinomial` and `np.random.choice()`) like `stats.norm`, `stats.gamma`, and `stats.poisson`
2. [samples](week-03-Data-Visualization#Sampling) versus populations (distributions)
3. [statistical inference](week-03-Data-Visualization#Statistics-Estimate-Parameters)

**Out of scope**
1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as expectation, moments, integration, heavy tailed distributions...
4. ...such as kernel functions for kernel density estimation
5. ...bokeh, shiny, d3, ...


## TUT/HW Topics

### Continuous, discrete, nominal and ordinal categorical, and binary

Not to be confused with `type()`, `.astype()`, and `.dtypes`; or, `list`, `tuple`, `dict`, `str`, and `float` and `int` (although the latter two correspond to **continuous** and **discrete** numerical variables...). 

![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhb3mixP0J2LWeQsK3VWxL8RM0tFg5-8KBLLMYSPfjzZrGw60PCyknoxmqJqyvn6yR2lS4J_8sblZs3blNsagoCq9JHxjSPhtv8PH3DnBZEmPl7zcNZaSjPwCoMQ1j6Cu8tFGULP8uZQmfV/s1600/horst+discrete+continuous.jpg)

![](https://pbs.twimg.com/media/Ehh6v4kVoAIbotc?format=jpg&name=4096x4096)


### Bar plots and modes

A **bar plot** is a chart that presents categorical data with rectangular bars with heights (or lengths) proportional to the values that they represent. 

```python
import pandas as pd
import plotly.express as px

# Sample data
df = pd.DataFrame({
    'Categories': ['Category1', 'Category2', 'Category3'],
    'Values': [10, 20, 30]
})

# Create a bar plot with Plotly
fig = px.bar(df, x='Categories', y='Values', text='Values') 
# The bars can be plotted horizontally instead of vertically by switching `x` and `y`

# Show the plot
fig.show()
```

A bar plot is essentially a visual representation of the `.value_counts()` method of the `pandas` library.

> Remember, the `.value_counts()` is a method that produces counts of unique values (sorted in descending order starting with the most frequently-occurring element). The the `.value_counts()` method excludes missing values by default, but these can included by instead using `.value_counts(dropna=False)` with the additional `parameter=argument` specification.

```python
# Example data
data = pd.Series(['a', 'b', 'a', 'c', 'b', 'a', 'd', 'a'], name="Variable")

# Use .value_counts()
counts = data.value_counts()

# Create bar plot with Plotly
fig = px.bar(counts, y=counts.values, x=counts.index, text=counts.values)
fig.update_layout(title_text='Value Counts', xaxis_title='Categories', yaxis_title='Count')
# This updates the layout to add a title and labels to the x and y axes to make the plot more immediately sensible
fig.show()
```

In this code, we first count the occurrences of each category in the series using `.value_counts()`. Then, we create a bar plot with `px.bar()`. The `y` argument is set to the counts (the number of occurrences of each category), and the `x` argument is set to the index of the counts (the categories themselves). The `text` argument is used to display the count numbers on top of the bars. For example, the height of the 'a' bar is the count of 'a' in the dataset. 

> A **bar plot** is a simple way to understand the distribution of categorical data that is a good alternative to just reading `df.value_counts()`.
>
> And with interactive `plotly` bar plots, you can hover over the bars to see the exact counts, and you can also zoom in and out, and save the plot as a png file.

The **mode** in statistics is the value (or values) that appears most frequently in a data set. In the bar plot above 'a' is the mode since it appears most frequently in the dataset. 

> The term "modes" is also used to refer to "peaks" in a distribution of data... this will be discussed later in the context of "multimodality"


### Histograms

A **histogram** is a graphical display of data using bars of different heights. It is similar to a **bar plot**, but a histogram instead counts groups of numbers within ranges. That is, the height of each bar shows how many data points fall into each range. For example if you measure the [heights of trees in an orchard](https://www.mathsisfun.com/data/histograms.html), you could put the heights data into 
and the heights vary from 100 cm to 350 cm in intervals 50 cm, so a tree that is 260 cm tall would be added to the "250-300" range.  Histograms are a great way to show results of numeric data, such as weight, height, time, etc.

- In a histogram, the **width of the bars (also known as bins)** represents the interval that is used to group the data points. The choice of bin size can greatly affect the resulting histogram and can change the way we interpret the data.

- If the **bin size is too large**, each bar might span a wide range of values, which could obscure important details about how the data is distributed. On the other hand, if the **bin size is too small**, the histogram could become cluttered with many bars, making it difficult to see the overall pattern.

- Choosing an appropriate bin size is a balance between accurately representing the data and maintaining readability. 

To illustrate this, the code below generates two histograms for the same dataset, but with different bin sizes. The first histogram uses a smaller bin size, and the second one uses a larger bin size. As you can see, the histogram with the smaller bin size has more bars, each representing a narrower range of values. In contrast, the histogram with the larger bin size has fewer bars, each representing a wider range of values.

```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Generate a random dataset
np.random.seed(0) # scipy random seeds can be set with numpy
n = 500
data = stats.norm().rvs(size=n)

# Create a histogram with smaller bin size
fig1 = go.Figure(data=[go.Histogram(x=data, nbinsx=50)])
fig1.update_layout(title_text='Histogram with Smaller Bin Size')
fig1.show()

# Create a histogram with larger bin size
fig2 = go.Figure(data=[go.Histogram(x=data, nbinsx=10)])
fig2.update_layout(title_text='Histogram with Larger Bin Size')
fig2.show()
```

> In `px.histogram` (from `import plotly.express as px`) the parameter specifying the number of bins is `nbins` (not `nbinsx`); whereas, in `seaborn` and `matplotlib` (and hence `pandas`) the parameter is just `bins`. Here's some nifty code that would let you think about "widths" instead of number of bins, which is probably more useful sometimes.
>
 ```python
 import math
 bin_width = 0.5  # Choose your desired bin width
 nbinsx = math.ceil((data.max() - data.min()) / bin_width) # or `nbins` or `bins` if you're using another
 ```
>
> - Also... please note that the actual bin width in the histogram might not be exactly the same as the desired bin width due to the way that `Plotly` [automatically calculates the bin edges](https://community.plotly.com/t/histogram-bin-size-with-plotly-express/38927)

### Box plots and spread

In statistics, the term **range** refers to the difference between the highest and lowest values in a dataset, so it provides a simple measure of the spread (or dispersion or variability) of the data. For example, in the set {4, 6, 9, 3, 7}, the lowest value is 3, and the highest is 9. So the range is 9 - 3 = 6. 
> The range can sometimes be misleading if the highest and lowest values are extremely exceptional relative to most of the values in the data set, so be careful when considering the range of a numeric variable. For example, the range of salaries is not very representative of most "working class" salaries. 

The **interquartile range (IQR)** is another statistical measure of the spread (or dispersion or variability) of the data. It is defined as [the difference between the 75th and 25th percentiles of the data](https://statisticsbyjim.com/basics/interquartile-range/). In other words, the IQR includes the 50% of data points that are above the first and third quartiles (Q1 and Q3). The IQR is used to assess the variability where most of your values lie. Larger values indicate that the central portion of your data spread out further, while smaller values show that the middle values cluster more tightly.

A **box plot** is a graphical representation for characterizing the relative statistical spread (or dispersion or variability) of a numerical data distribution around its median. A box plot serves a similar purpose to as **histogram**, but it is a simpler alternative that provides a higher-level summary of numerical data. 

The "box" of a box plot is constructed out of the first (Q1) and third quartiles (Q3) separated by the median (which is the second quartile Q2); so, a box plot shows the the 25th, 50th, and 75th percentile with the "box" containing the middle 50% of the data. 

Box plots additionally draw **whiskers** (lines extending from the box) which typically extend to the most extreme data points within 1.5 times the interquartile range (IQR) of the first and third quartiles. 

- The **lower whisker** extends to the smallest data point within **1.5 * IQR** below the first quartile.
- The **upper whisker** extends to the largest data point within **1.5 * IQR** above the third quartile.

Data points that fall outside this range of the whiskers are typically termed **outliers** and are plotted as individual points. 

> The term "outliers" is only meant to be suggestive as outliers can mean different things in different context. In the context of a box plot, referring to the points beyond the extent of the whiskers is simply a technical term related to the definitional construction of a box plot.

![](https://360digit.b-cdn.net/assets/img/Box-Plot.png)

### Skew and Multimodality

[**Skewness**](https://stats.libretexts.org/Courses/Penn_State_University_Greater_Allegheny/STAT_200%3A_Introductory_Statistics_%28OpenStax%29_GAYDOS/02%3A_Descriptive_Statistics/2.06%3A_Skewness_and_the_Mean_Median_and_Mode) is a measure of the asymmetry of a data distribution. If skewness is present, then it's either left "negative" or right "positive" skew. 

- If the data is spread out more to the left direction then the skew is "negative"
- If the data is spread out more to the right direction then the skew is "positive"

![](https://s3.amazonaws.com/libapps/accounts/73082/images/Skeweness.jpg)

Skewness can be understood in terms of its affect on the relationship between the **mean** and **median**.

- The **mean** is the average of all data points in the dataset.
- The **median** is the middle point of a number set, in which half the numbers are above the median and half are below.

> The median is the 50th percentile of the data so it is the "second quartile" and can be denoted as Q2 similarly to the first and third quartiles Q1 and Q3 (which are the 25th and 75th percentile of the data).

In a **symmetric** distribution, the mean and median are the same. However, when data is skewed, the mean and median will differ.

- In a **positive skew** context, the mean will be greater than the median. This is because the mean is influenced by the high values in the tail of the distribution and gets pulled in the direction of the skew.
- In a **negative skew** context, the mean will be less than the median. This is because the mean is influenced by the low values in the tail of the distribution and gets pulled in the direction of the skew.

A simple way to remember this relationship is that the mean is 'pulled' in the direction of the skew.

**Modality** refers to the general natures of the "peaks" that may be present in a numerical data set. 
> These "peaks" are often informally referred to as "modes" but these should not be confused with the technical definition of **mode** which refers to the most common unique value in a data set.

![](https://miro.medium.com/v2/format:webp/0*m_Fd3Opt6L70LiYS.png)

Box plots are great for comparing the distributions of different groups of data; and, they also clearly indicate the presence of **skew** in a numerical data set; **but, be careful as box plots cannot represent multimodality in a data set; and, box plots do not indicate the amount of data they represent without the addition of an explicit annotations indicating this; while, Histograms, on the other hand, automatically give indications of both modality and sample size (through their "y-axis").**

![](https://www.simplypsychology.org/wp-content/uploads/box-plots-distribution.jpg)

#### Normal distributions 

An interesting "special case" of unimodal distributions is the **normal distribution**.
The figures below illustrates the quantiles (Q1 and Q3) for a normal distribution based on the corresponding boxplot.
The "sigma" σ character in the figure below signifies the **standard deviation** and the bottom most figure shows the percentiles that correspond to different multiplicative ranges of the standard deviation.

In statistics a **normal distribution** is a fundamental concept that describes how data points are spread out. It is also known as the **Gaussian distribution**, named after the mathematician Carl Friedrich Gauss. The **normal distribution** is essential because it often appears in real-world data and underpins many statistical methods.

> The standard deviation is essentially defined specifically for normal distributions and its meaning is exactly clear in the context of normal distributions. It is somewhat harder to interpret the meaning of standard deviation the further from "normality" (the more "non normal") the distribution under consideration is. The greater the degree that a data distribution is skewed or non unimodal, the less clear it is what the meaning of the standard deviation is.  

![](https://jblomo.github.io/datamining290/slides/img/quartiles.png)

**Characteristics of a Normal Distribution**

A normal distribution has several key features:

- Symmetry: The distribution is perfectly symmetric about the mean.
- Central Tendency: The mean, median, and mode of the distribution are all equal and located at the center of the distribution.
- Bell-Shaped Curve: The distribution forms a bell-shaped curve, with the highest point at the mean.
- Spread: The spread of the distribution is determined by the standard deviation. A larger standard deviation means the data is more spread out from the mean, while a smaller standard deviation means the data is more clustered around the mean.
- Know Probabilities: about 2/3's of the area of a **normal distribution** is within "plus and minus ONE standard deviation of the mean" while about 95% of the area of a **normal distribution** is within "plus and minus TWO standard deviations of the mean"


## LEC Extensions

### Modern Plotting

There are many *data visualization libraries*, and modern ChatBots are very familiar with them. Making *data visualizations* is therefore just a matter of being aware of what's possible and available in *different visualization libraries* and requesting the code to accomplish your objectives from ChatBots. And learning what's possible and available in *different visualization libraries* is just a matter of keeping your eyes open to see what's possible. All you need to do to become an expert in *data visualization* is to just start exploring *data visualization* "galleries" (like those of [_plotly_](https://plotly.com/python/), [_seaborn_](https://seaborn.pydata.org/examples/index.html), which is built on [_matplotlib_](https://matplotlib.org/stable/gallery/index.html) which [_pandas_](https://pandas.pydata.org/docs/user_guide/visualization.html) provides direct access to, and [bokeh](https://docs.bokeh.org/en/latest/docs/gallery.html), [ggplot](https://exts.ggplot2.tidyverse.org/gallery/), [shiny](https://shiny.posit.co/r/gallery/), or [D3.js](https://observablehq.com/@d3/gallery), and other *data visualization* compilation galleries (like [this one for python](https://python-graph-gallery.com) or [this one for R](https://r-graph-gallery.com) since exploring types of possible *data visualization* is truly a language agnostic question). 

We've chosen to emphasize `plotly` because it's a popular and attractive *data visualization library* that provides the most modern features that one would like from a visualization library, such as information panel access via "hovering" and rudimentary "interactive data dashboarding" and animation. But you're welcome to work in whatever tools you like, with the only caveat being that not all types figures render on GitHub (so TAs marking your homework submissions can see your figures), especially the fancier more complicated kinds of figures which are essentially some form of javascript "widget".  For example, unless you use `fig.show(renderer="png")` for all `plotly` figures that are part of GitHub (and MarkUs) submissions, the `plotly` figure will simply not appear on rendered GitHub (MarkUs) pages.

The foundations of *data visualization* have been considered and studied for a long time, e.g., following the seminal work of 
[Edward Tufte](https://www.edwardtufte.com) and popularized through mainstream treatments such as [How to Lie with Statistics](https://en.wikipedia.org/wiki/How_to_Lie_with_Statistics) (which, contrary to the apparent sentiment of the title, actually instead tries to educate the general population about how to be literate think critically about *data visualization*). We've come very far since those "early days", however, and modern taxonomies and organizational characterizations of visualization, such as those from [David McCandless](http://www.davidmccandless.com/) are much more focussed on optimizing the power of informative story telling through *data visualization*.  Data McCandless is a new breed of **Data Journalists** who report on the world empirically, using *data visualizations*, but not in an old, dry, boring way.  They make looking at data fun and awesome. My favourite **Data Journalism** comes from [The Pudding](https://pudding.cool), which makes reading a news article a crazy, interactive, immersion experience.  To make the kinds of awesome "knock your socks off" *data visualizations* that we're seeing from **Data Journalists** these days requires the previously mentioned ([D3.js](https://d3js.org)). If you haven't figured it out, the ".js" stands for **javascript**. But as you may actually have figured out, with the advent of ChatBots, that's not going to matter. You can work with a ChatBot to make awesome *data visualizations* with [D3.js](https://observablehq.com/@d3/gallery).  While we'll not be formally doing this as part of STA130, the only thing that's stopping you from doing so is... you.  And your good excuses and explanations for this at this point in time (which do not include "I can't code in javascript or D3.js").

**Legends, annotations, figure panels, etc.**

There are many standard components, elements, and ornaments available for *data visualization* plots.  The three listed above **legends**, **annotations**, and **figure panels** are some basic "standard" aspects often leveraged in *data visualization*.  But the more examples of *data visualizations* that you'll see, the more familiar you'll be with the nature of the canvas and the availability of the tools at your disposal for *data visualization* purposes. So if you don't have any idea what **legends**, **annotations**, and **figure panels** are, the HW and LEC will introduce you to these topics and over time you'll become increasingly comfortable incorporating these kinds of elements into your *data visualizations*. And at that point you'll simply be asking a ChatBot to provide the necessary code to execute the plans you've evisioned for your figure. Additionally, the vision you've decided upon might be dynamically evolved and updated and improved through an interactive process with a ChatBot. ChatBots can be very good soundboards for exploring ideas, and suggestion possible variations and extensions relative to what you're currently attempting to do. 

### Smoothed Histograms 

The previously introduced **histograms** (and **box plots**, etc.) provide a simple way to understand the **empirical distribution** of data though simple visualization that reflects the **location** (central tendency) and **spread** (scale) of the data. The choice of the **number of bins** to use for a **histograms** is in some sense arbitrary, but can be sensibly made. And, for example, based on your specification of this, `plotly` **histograms** simple determine a **number of bins** which will be approximately the number that you asked for. And, really, the **number of bins** only matters for differences like 5 bins, versus 25 bins, versus 100 bins because this really impacts the nature of the **histograms** visualization in terms of how coarsely it represents the data. So these differences significantly change the degree of the "data compression" and "simplification" that is presented to the viewer; whereas, changes of a "plus or minus a few bins" really will not qualitatively affect the nature of the information presentation. 

It can nonetheless feel a little jarring that **histograms** can indeed start to look different if the bins get moved a little to left or to the right, or if "plus or minus a few bins". This makes it feel like **histograms** have some degree of "arbitrariness" to them (which they do). One might then wonder if there was a way to remove the "artifacts" caused by the exact details of the **binning** specifications. And, indeed, there is a "continuous" approach to remove the "discrete" nature of **histograms** caused by their **binning** mechanism.  This is to instead visualize so-called **kernel density estimation (KDE)**. A **KDE** is essentially a "local average of the number of points" (within a local area of the data).  While **histograms** provide a **discretely binned** approximation of **empirical distribution** of data, a **KDE** can represent the approximation of **empirical distribution** of data as a smooth curved function.  

```python
import plotly.express as px
# Create a violin plot
df = px.data.tips()
fig = px.violin(df[df.day=='Sat'], y="total_bill", x="day", color="sex", 
                box=True, points="all", hover_data=df.columns)
fig.show()
```

Most plotting libraries, like `plotly`, provide access to **kernel density estimation (KDE)** functionality through so-called **violin** plots.  The `plotly` also provides an alternative interface to this functionality through `ff.create_distplot` although this is no longer preferred and is now **depreciated**. Nonetheless the following code and visualization is still informative since it shows that the **violin** plot is just mirrored image reflection of a **KDE**.

```python
import plotly.figure_factory as ff

# Separate data for each sex
df_saturday = df[df['day'] == 'Sat']
male_bills = df_saturday[df_saturday['sex'] == 'Male']['total_bill']
female_bills = df_saturday[df_saturday['sex'] == 'Female']['total_bill']

fig = ff.create_distplot([male_bills, female_bills], 
                         group_labels=['Male', 'Female'], 
                         show_hist=False, show_rug=True)
fig.update_layout(title='KDE Plot of Total Bill on Saturday by Sex',
                  xaxis_title='Total Bill', yaxis_title='Density')
fig.show()
```

To see this even more explicitly, here we make a **violin** plot like the first one we made, but rather than making two different **violin** plots by `sex` we instead make a single **violin** plot where the **KDE** on each "side" of the **violin** plot is for either of the two levels of `sex` considered in our dataset.

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add Male distribution on the left side of the violin with a teal color
fig.add_trace(go.Violin(y=df_saturday['total_bill'][df_saturday['sex'] == 'Male'],
                        x=df_saturday['day'][df_saturday['sex'] == 'Male'],
                        side='negative',  # Plots on the left side
                        line_color='teal', fillcolor='rgba(56, 108, 176, 0.6)', 
                        name='Male', box_visible=True,
                        points='all', meanline_visible=True))

# Add Female distribution on the right side of the violin with a coral color
fig.add_trace(go.Violin(y=df_saturday['total_bill'][df_saturday['sex'] == 'Female'],
                        x=df_saturday['day'][df_saturday['sex'] == 'Female'],
                        side='positive',  # Plots on the right side
                        line_color='coral', fillcolor='rgba(244, 114, 114, 0.6)', 
                        name='Female', box_visible=True,
                        points='all', meanline_visible=True))

fig.update_layout(title='Total Bill Distribution on Saturday by Sex (Single Violin)',
                  yaxis_title='Total Bill', xaxis_title='Day', violinmode='overlay')
fig.show()
```

The nice thing about **KDEs** is that they give us a "histogram" but it doesn't have **bins** and is instead just a smooth curved function approximation of **empirical distribution** of data. Technically speaking, a **KDE** is a **non-parametric estimation** of the **probability density function (PDF)** of a **continuous random variable**. It is useful for visualizing the distribution of a dataset when we want a smooth curve, rather than a binned representation like a histogram. 

In a **KDE** plot, each data point is replaced by a smooth, symmetric **kernel** (often **Gaussian**) centered at that point. The sum of these **kernels** across all data points produces a smooth curve that represents the **estimated probability density** underlying the data. It is the  smooth curved function approximation of **empirical distribution** of data. A **violin** plots takes **KDEs** one step further by representing this in a visually pleasing manner. Unlike **box plots**, which display summary statistics (median, quartiles, etc.), **violin** plots show the entire distribution by mirroring the **KDE** on both sides of the axis, giving the plot its characteristic "violin" shape. Or, as demonstrated above, each side of the **violin** plot can represent two sides of a dichotomous division of the data. 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Comparison_of_1D_histogram_and_KDE.png/500px-Comparison_of_1D_histogram_and_KDE.png)

A **violin** plot is especially useful for **comparing distributions** across different categories and displaying **multi-modal distributions** which a **box plot** would be unable to visualize. Visually examining an **empirical data distribution** is often more informative than just examining summary statistics, and a **violin** plot is often more aesthetically attractive than a **histogram** while being able to provide the same level of information and detail. 

```python
fig = go.Figure()

# Loop over unique days to add traces
for day in df['day'].unique():
    
    # Male distribution
    male_subset = df[(df['day'] == day) & (df['sex'] == 'Male')]
    fig.add_trace(
        go.Violin(y=male_subset['total_bill'],
                  x=[day] * len(male_subset),  # x is the day for all entries
                  side='negative',  # Left for Male
                  line_color='teal', fillcolor='rgba(56, 108, 176, 0.6)',
                  name='Male', box_visible=True, points='all', meanline_visible=True))
    
    # Female distribution
    female_subset = df[(df['day'] == day) & (df['sex'] == 'Female')]
    fig.add_trace(
        go.Violin(y=female_subset['total_bill'],
                  x=[day] * len(female_subset),  # x is the day for all entries
                  side='positive',  # Right for Female
                  line_color='coral', fillcolor='rgba(244, 114, 114, 0.6)',
                  name='Female', box_visible=True, points='all', meanline_visible=True))

fig.update_layout(title='Total Bill Distribution by Day and Sex',
                  yaxis_title='Total Bill', xaxis_title='Day', violinmode='overlay')
fig.show()
```

A `plotly` **violin** plot is an excellent choice when you want both a clear picture of the **empirical data distribution** and the ability to compare between different categories, while also enjoying the interactive benefits that `plotly` provides.

```python
import seaborn as sns

df = sns.load_dataset('penguins').dropna()  # Load the penguins dataset
fig = go.Figure()
fig = px.violin(df, y="bill_length_mm",
                box=True, points="all", hover_data=df.columns)
fig.show()
```

Since a **violin** plot using a **KDE** presents the same information as a **histogram** but only with a slightly different approach, it's not surprising that it has an analogous "arbitrariness" issue as the choice of the **number of bins**. And this is that there is an "artifact" which will appear as an aspect of a **KDE**; namely, the **KDE** depends on the choice of a so-called **bandwidth** parameter.  This **bandwidth** parameter is essentially the "width" of the **kernel** used to construct the **KDE**. So, if each data point becomes a mini **Gaussian**-shaped mound and all these mounds are "added together" to produce the over all smooth curve function approximating the **empirical data distribution** (as shown in the linked image -- not the `plotly` code figures -- above), then the "width" of the **kernel** (and other details about the "arbitrary" choice of the **kernel** function) will affect the final smooth curve function **KDE**.  The **bandwidth** parameter of the **kernel** then analogously corresponds to the **number of bins** in a **histogram**.  More **histogram bins** means a finer less course less simplified visually summer of the data, and this corresponds to have a "narrower" **KDE kernel bandwidth**; or, vice-versa, fewer **histogram bins** means a coarser more simplified visually summer of the data, and this corresponds to have a "wider" **KDE kernel bandwidth**.So the **bandwidth** for **KDE** and the **number of bins** for **histograms** play the determining role in controlling the granularity and smoothness of the data representation. A "narrower" **bandwidth** leads to a more sensitive estimate, capturing more details of the data's structure, potentially revealing features like multimodality (just as having a larger **number of bins** does when using a **histogram**). However, this can also lead to an overly complex representation of the data (just as having too many **bins** in a **histogram** does) by not summarizing the data at a level that's appropriate relative to the amount of information that's actually available in the data. Having a single bin for every data point doesn't make for much of a "summary" of a dataset.  So too for overly "narrow" choices of a **bandwidth** parameter for a **KDE**. So **bandwidth** and **number of bins** involve a trade-off between detail and summarization smoothness. Finding the right balance is key: too few **bins** or too "wide" a **bandwidth** can oversimplify the data, while too many **bins** or too "narrow" **bandwidth** can overcomplicate its representation.


### log transformations

When data is extremely **right-skewed** we can do a **log transformation** to allow us to work with the data on a more "normal" scale. 

```python
import numpy as np
from plotly.subplots import make_subplots

url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
df = pd.read_csv(url)

# Log transformation of the amount column
df['log_amount'] = np.log(df['Amount'] + 1)

# Downsample the non-fraudulent class to 10,000 samples
non_fraudulent = df[df['Class'] == 0].sample(n=10000, random_state=42)
fraudulent = df[df['Class'] == 1]

# Combine the downsampled non-fraudulent class with the full fraudulent class
balanced_df = pd.concat([non_fraudulent, fraudulent])

# Calculate mean and standard deviation for original and log-transformed amounts
mean_amount = balanced_df['Amount'].mean()
std_amount = balanced_df['Amount'].std()
mean_log_amount = balanced_df['log_amount'].mean()
std_log_amount = balanced_df['log_amount'].std()

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Amount", "Log Transformed Amount"))

# Define positions for Non-Fraudulent and Fraudulent with increased separation
positions = balanced_df['Class'].map({0: -0.6, 1: 0.6})  # Increased separation

# Left violin plot for original amount (red for fraud)
fig.add_trace(go.Violin(
    y=balanced_df['Amount'][balanced_df['Class'] == 1],
    x=[0.6]*len(balanced_df['Amount'][balanced_df['Class'] == 1]),
    side='positive', line_color='red', fillcolor='rgba(244, 114, 114, 0.5)',
    name='Fraud', box_visible=True, points='all', meanline_visible=True,
    pointpos=-0.8, width=1.0, marker=dict(color='red', opacity=0.2)), row=1, col=1)

# Add a single trace for non-fraudulent transactions (purple)
fig.add_trace(go.Violin(
    y=balanced_df['Amount'][balanced_df['Class'] == 0],
    x=[-0.6]*len(balanced_df['Amount'][balanced_df['Class'] == 0]),
    side='positive', line_color='purple', fillcolor='rgba(128, 0, 128, 0.5)',
    name='Non-Fraud', box_visible=True, points='all', meanline_visible=True,
    pointpos=-0.8, width=1.0, marker=dict(color='purple', opacity=0.2)), row=1, col=1)

# Right violin plot for log-transformed amount (blue for fraud)
fig.add_trace(go.Violin(
    y=balanced_df['log_amount'][balanced_df['Class'] == 1],
    x=[0.6]*len(balanced_df['log_amount'][balanced_df['Class'] == 1]),
    side='positive', line_color='blue', fillcolor='rgba(56, 108, 176, 0.5)',
    name='Fraud Log', box_visible=True, points='all', meanline_visible=True,
    pointpos=-0.8, width=1.0, marker=dict(color='blue', opacity=0.2)), row=1, col=2)

# Add a single trace for non-fraudulent transactions (green)
fig.add_trace(go.Violin(
    y=balanced_df['log_amount'][balanced_df['Class'] == 0],
    x=[-0.6]*len(balanced_df['log_amount'][balanced_df['Class'] == 0]),
    side='positive', line_color='green', fillcolor='rgba(0, 128, 0, 0.5)',
    name='Non-Fraud Log', box_visible=True, points='all', meanline_visible=True,
    pointpos=-0.8, width=1.0, marker=dict(color='green', opacity=0.2)), row=1, col=2)

# Add rectangles for mean and mean + 1 std in the original scale (narrowed width)
fig.add_shape(type='rect',x0=-.75, x1=-0.25, row=1, col=1,
              y0=mean_amount, y1=mean_amount + std_amount,
              fillcolor='rgba(255, 255, 255, 0.5)', line=dict(color='yellow', width=3))

# Add rectangles for mean and mean + 1 std in the log-transformed scale (narrowed width)
fig.add_shape(type='rect', x0=-.75, x1=-0.25, row=1, col=2, 
    y0=mean_log_amount, y1=mean_log_amount + std_log_amount,
    fillcolor='rgba(255, 255, 255, 0.5)', line=dict(color='yellow', width=3))

# Add a box trace for mean ± std explanation to the legend
fig.add_trace(
    go.Box(y=[None],  # No data, just to create a legend entry
           name='Mean ± Std', marker=dict(color='yellow'), line=dict(width=3)))

# Update layout
fig.update_layout(title='Transaction Amount Distribution: Original vs Log Transformed',
                  yaxis_title='Amount', yaxis2_title='Log Amount')

# Set x-axis limits for both panels
fig.update_xaxes(range=[-1.3, 1.3], title_text='Class', row=1, col=1)
fig.update_xaxes(range=[-1.3, 1.3], title_text='Class', row=1, col=2)

# Update x-axis ticks and add labels
fig.update_xaxes(tickvals=[-1, 1], ticktext=['Non-Fraud', 'Fraud'], row=1, col=1)
fig.update_xaxes(tickvals=[-1, 1], ticktext=['Non-Fraud', 'Fraud'], row=1, col=2)

fig.update_yaxes(range=[0, balanced_df['Amount'].max() + 500], row=1, col=1)  # Adjust as needed
fig.show()
```

What does it mean to treat data more "normally"? Well, literally, we're trying to make it look more like a **normal distribution** because this is a simple **distribution** that's easy to think about about because it's super simple to understand what its **population mean parameter** indicates.  That's just basically "the middle" location where the **normal distribution** is placed. And we can also interpret what the **population standard deviation parameter** indicates.  Namely, for a **population** that's **normally distributed** about 95% of the area of the **distribution** is between "plus and minus two **standard deviations**.  We should careful here to distinguish between the **sample mean statistic** and the **population mean parameter**, and as well to distinguish between the **sample standard deviation statistic** and the **population standard deviation parameter**. But they carry the same meaning and interpretation if we're talking about an **empirical distribution** or a **population distribution** so long as their shape is approximately **normally distributed**. 

The **standard deviation** of a **right-skewed** sample of data is hard to interpret because this really depends on the nature of the outliers and "decay behavior" of the tail (how quickly the data "peters out" to the right). The **sample mean** itself can also be very challenging to interpret because how far the mean is pulled away from the median depends on the degree of **right-skew** (and again the **outliers**) that are present in the data. So, when we encounter a dataset with **right-skewed distribution** having a majority of values are clustered at the lower end with a long tail extending towards higher values, this can make statistical analyses and interpretations challenging, especially for statistics like the **mean** and **standard deviation**. This can all be fixed, though, if we work on the **log scale** based on using a **log transformation**. This can make the data literally look more "normal" and approximately have the **normal distribution** "shape".  We'll not discuss the mathematical operation that a **log transformation** executes, but the benefits of a **log transformation** are to (a) reduce **right-skewness** because the **log transformation** pulls he long tail of larger values is pulled closer to the bulk of smaller values and (b) make the meaning of **statistics** like the **mean** and **standard deviation** more interpretable (although we must remember that these only apply to the "log scale" that we're now working on as a result of the **log transformation**).  So the after a **log transformation** the **sample mean** is a better representation of the central tendency, as it is less influenced by **outliers** and **right-skew**, and the **standard deviation** can be meaningfully understood in terms of what it means regarding the spread of the data around the **sample mean**. The **log transformation** is a powerful technique for handling **right-skewed** data since by transforming the data to achieve a more normal distribution we enhance our ability to interpret and have a more meaningful understanding of the dataset. 

## LEC New Topics

### Populations and Distributions

A **populations** is generally a theoretical idea that imagines the collection of all possible values the **observations** made for a **variable** could hypothetically be. It can refer to concrete group, such as "All Canadians" or "All UofT Students" or "All UofT International Students"; but, for all but very small populations, it would for all practical purposes not be possible to actually measure every **observation** in a **population** that could possibly occur.

In statistics, we often imagine a theoretical population as an idealized group of "all possible data points"; and, when we represent these mathematical or numerical, we call them **distributions**. We have already seen several of these **distributions**.

```python
# The first we saw was the Multinomial distribution
from scipy import stats
number_to_select = 5  # n = 5
option_frequency = [0.6,0.3,0.1]  # k = 3 options
Multinomial_distribution_object = stats.multinomial(n=number_to_select, p=option_frequency)

# Two special cases of the Multinomial distribution where there are just two options are

# - The Binomial distribution
p = 0.5 # chance of getting a success ("1" as opposed to "0") <- doesn't need to be 0.5
Binomial_distribution_object = stats.binom(n=number_to_select, p=p)  # stats.multinomial(n=number_to_select, p=[p,1-p]) 

# - The Bernoulli distribution
Bernoulli_distribution_object = stats.bernoulli(p=p)  # stats.binom(n=1, p=p)

# Some other distributions are the Normal, Poisson, and Gamma distributions 

μ = 0 # mean parameter
σ = 1 # standard deviation parameter
Normal_distribution_object = stats.norm(loc=μ, scale=σ)

λ = 1 # mean-variance parameter
Poisson_distribution_object = stats.poisson(loc=λ)

α # shape parameter
θ # scale parameter
Gamma_distribution_object = stats.gamma(a=α, scale=θ)
```

### Sampling 

In statistics, **sampling** refers to the process of selecting a subset of individuals from a **population**. Ideally, the sample should be collected in such a way that it is **representative** of the **population**. This is because the primary purpose of **sampling** is to **estimate** the characteristics (called **parameters**) of the whole **population** when it is impractical or impossible to collect data from an entire **population**. Estimating **population parameters** based on a sample is called **statistical inference**.

Do you recall the notion of **statistical independent**? This is very important here. When we're collecting **samples** from a **population** it will be most efficient if selecting one individual **observation** doesn't affect which of the other individual **observations**. If the **samples** are **statistically dependent** it means they come in clusters, like if selecting one person for the **sample** then means we'll also select all the friends for the **sample**. This is **statistically dependence** and perhaps you can see the problem here. This clustering in the **sampling** is going to make it more challenging to collect a **sample** that is more **representative** of the **population**. What actually happens when you have $n$ **dependent** rather than **independent** **samples** is that you don't really actually have $n$ **independent** pieces of information. Because the samples are **dependent** it's like each of the **dependent samples** is not quite a full piece of information. So with **dependent sampling** you may have $n$ **samples** but you don't have $n$ **independent** pieces of information. So that's why **independent samples** are preferred over $n$ **dependent samples**.  Additionally, most **statistical inference** methods assume that the samples they're using are **independent samples**. So if this is not true and actually **dependent samples** are being used, then the **statistical inference** method will be overly confident and biased. All of the **sampling** demonstrated below is based on drawing **independent samples** from the **distributions** being sampled from. 

```python
# Samples can be taken from the Multinomial distributions (and special cases) in the code above 
# using the "random variable samples" `.rvs(size=number_of_repetitions)` method of the distribution objects

Multinomial_distribution_object.rvs(size=1)  # `number_of_repetitions` is many times "5 things are chosen from 3 options with `option_frequency`"
Binomial_distribution_object.rvs(size=1)  # `number_of_repetitions` sets how many times "5 things are chosen from 2 options with chance `p`"
Bernoulli_distribution_object.rvs(size=1)  # choose 0 or 1 with probability p

# Which can also be correspondingly created using the following
np.random.choice([0,1,2], p=option_frequency, size=number_to_select, replace=True)
np.random.choice([0,1], p=[p,1-p], size=number_to_select, replace=True)  
np.random.choice([0,1], p=[p,1-p], size=1, replace=True)  # Bernoulli

# And this is analogously done for the Normal, Poisson, and Gamma distributions objects
Normal_distribution_object.rvs(size=n)  # `number_of_repetitions` sets how many "draws" we "sample" from this distribution
Poisson_distribution_object.rvs(size=n)  # ditto
Gamma_distribution_object.rvs(size=n)  # ditto
```

The $n$ samples from any of the above calls would typically be notated as $x_1, x_2, \cdots, x_n$.

### Statistics Estimate Parameters 

The greek letters above (μ, σ, λ, α, and θ) are the parameters of their corresponding distributions. Parameters are the characteristics of population which we are trying to estimate by sampling. To make inferences on the population parameters, we estimate the parameter values with appropriately constructed statistics (which are mathematical functions of) samples. For example:

- The population mean of a Normal distribution μ is estimated by the sample mean 

  $$\bar x = \frac{1}{n}\sum_{n=1}^n x_i$$
- The population standard deviation of a normal distribution σ is estimated by the sample standard deviation 

  $$s = \sqrt{\frac{1}{n-1}\sum_{n=1}^n (x_i-\bar x)^2}$$
- The population mean of a Poisson distribution λ (which is also the variance poisson distribution population) is estimated by the sample mean $\bar x$ or the sample variance $s^2$
- And the shape α and scale θ parameters of a Gamma distribution can also be estimated, but the statistics for estimating these are a little more complicated than the examples above

Here we use the **sample mean** and **sample standard deviation** to estimate the **normal distribution** which best approximates the **log transformed** `Amount` data from the `creditcard.csv` dataset introduced above. 

```python
from scipy import stats

url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
df = pd.read_csv(url)
# Log transformation of the amount column
df['log_amount'] = np.log(df['Amount'] + 1)

mean_log_amount = df['log_amount'].mean()
std_log_amount = df['log_amount'].std()

fig = go.Figure()
fig.add_trace(
    go.Histogram(x=hist_data, nbinsx=30, histnorm='probability density',
                 name='Log Transformed Amount', opacity=0.6, marker_color='blue'))

# Create a range of x values over which to draw the normal distribution
x = np.linspace(df['log_amount'].min(), df['log_amount'].max(), 100)
# Calculate the mathematical function of the normal distribution
# for the sample mean and sample standard devation of  the log-transformed data
y = stats.norm.pdf(x, mean_log_amount, std_log_amount)

# Add the normal distribution estimated by the data over the data histogram
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution', 
                         line=dict(color='red')))
fig.update_layout(
    title='Normal Distribution Estimation of Log Transformed Amount',
    xaxis_title='Log Transformed Amount', yaxis_title='Probability Density',
    showlegend=True)
fig.show()
```

# Course Tutorial: Week 03 TUT

## STA130 TUT 03 (Sep20)<br><br>🎨 🤖 <u>Data Visualization with ChatBots<u>



## ♻️ 📚 Review  / Questions [15 minutes]

1. Follow up questions and clarifications regarding **`python` code** encountered in class so far 

> because figures are created with code, so comfort using **data types** , **arguments**, **`for` loops**, and **`if/else/elif` logical flow control** (from Sep13 TUT and Sep16 LEC), etc. is a prerequesitve for having any sensbile recognition of the logic of code creating a figure, *especially in the HW*...    
    
    
## 🚧 🏗️ Demo (on working from documentation)  [60 minutes]

#### 1. Use the [penguins](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv) dataset to demonstrate [bar plots](https://plotly.com/python/bar-charts/), [histograms](https://plotly.com/python/histograms/), and [box plots](https://plotly.com/python/box-plots/) using [_plotly_](https://plotly.com/python/) in a Jupyter notebook: click the links for the different types of plots, and figure out how to make the plot using some example from the [penguins](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv) dataset
 
> *This excercise has been done below, but it's easy to delete the code from the cells and do it again from scratch as a demonstration*
    
1. Use the cartoons to breifly discuss "quantitative versus qualitative data"
2. Discuss the "bins" (`nbins`) parameter (demonstrated [here](https://plotly.com/python/histograms/#choosing-the-number-of-bins))
3. Discuss how to "read a box plot" by explaining what the different parts of a box plot mean

#### 2. Use the plots you've created to explain the primary location (mean, median, and mode) and scale (standard deviation, range, interquartile range) concepts...
    
1. There is a fun [interactive widget](https://plotly.com/python/histograms/#histograms-in-dash) that might be helpful...<br><br>
        
2. `df.describe()` is probably useful for discussing medians, range, and interquartile range... and mode is useful in the context of a bar plot...<br><br>
        
3. If time permits, it might be helpful to use some alternative datasets to illustrate skew ([tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv)), outliers ([fraud](https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv)), and true modality ([old faithful](https://gist.githubusercontent.com/hogwild/c2704a1ae38c0a36983bc13121050dac/raw/7fd577be21752939375d92cd3a808558106e903b/oldFaithfulGeyserDataset.csv)) (as opposed to the modality from subpopulations, like how the modality in the penguins data goes away if you're looking only at individual species)
    


```python
import pandas as pd 
pingees = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
pingees
```

**You may need to do the following for `plotly` figures to render!**


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


```python
import plotly.express as px
#data_canada = px.data.gapminder().query("country == 'Canada'")
#fig = px.bar(data_canada, x='year', y='pop')

# OOPS! This needs an x-position and a y-height value!
fig = px.bar(pingees.species.value_counts().reset_index(), x='species', y='count')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# So what is this showing? 
# (And below we show what the `pingees.species.value_counts().reset_index()` "trick" is)
```


```python
pingees.species.value_counts()
```


```python
pingees.species.value_counts().reset_index()
```


```python
#import plotly.express as px
#df = px.data.tips()
#fig = px.histogram(df, x="total_bill")
fig = px.histogram(pingees, x="flipper_length_mm")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# So what is this showing? 

# This automatically calculates the "counts" and 
# doesn't put "spaces" between the bars (indicating it's numeric not categorical data)
```

**This probably applies to this data, e.g., the `flipper_length_mm` variable...** <br></br>
and is likely something to come back to when thinking about the interpretability of "means" and "standard deviations"...


![https://allisonhorst.com/data-science-art](https://cdn.myportfolio.com/45214904-6a61-4e23-98d6-b140f8654a40/e9665844-7316-4409-9340-84d7dec12b16_rwc_8x0x1894x1481x1894.png?h=2acdee343bc6b1863d0bbef964850192)


```python
#import plotly.express as px
#df = px.data.tips()
## Here we use a column with categorical data
#fig = px.histogram(df, x="day")
fig = px.histogram(pingees, x="species")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# So what happened here? 
# (There's a hint below showing that `Object` type data is treated as categorical data)
```


```python
pingees.dtypes # remember this?
```

![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhb3mixP0J2LWeQsK3VWxL8RM0tFg5-8KBLLMYSPfjzZrGw60PCyknoxmqJqyvn6yR2lS4J_8sblZs3blNsagoCq9JHxjSPhtv8PH3DnBZEmPl7zcNZaSjPwCoMQ1j6Cu8tFGULP8uZQmfV/s1600/horst+discrete+continuous.jpg)

![](https://pbs.twimg.com/media/Ehh6v4kVoAIbotc?format=jpg&name=4096x4096)


```python
#import plotly.express as px
#df = px.data.tips()
#fig = px.box(df, x="total_bill")
fig = px.box(pingees, x="flipper_length_mm") # or change orientation by using "y"
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# How does this compare to the alternative histogram presentation of this data?
```


```python
#import plotly.express as px
#df = px.data.tips()
#fig = px.box(df, x="time", y="total_bill")
fig = px.box(pingees, x="species", y="flipper_length_mm")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# Does this help explain things?
```


```python
# You have to look farther into the plotly histogram documention
# but if you scroll a little bit you can find this

#import plotly.express as px
#df = px.data.tips()
#fig = px.histogram(df, x="total_bill", color="sex")
fig = px.histogram(pingees, color="species", x="flipper_length_mm")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# Does this help explain things?
```


```python
# And if you look just a little farther back in the plotly box plot documention
# you can find this... just a little bit down

#import plotly.express as px
#df = px.data.tips()
fig = px.box(pingees, x="species", y="flipper_length_mm", points="all")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# Do you like this better? If so, why?
```

## 💬 🗣️ Communication Activity [25 minutes]

1. **[10 minutes]** Break into 4 groups of 6 students (or as many students as attendance allows, evenly distributed across the 4 groups) and prepare a speech describing the generic strategy or general sequence of steps you would take to understand a dataset

> The presentations are meant to reinforce the topics of the course that have been encountered so far, not introduce and explain new topics; so, the presentations should be sure to emphasize the topics of the course considered so far while sensibly and realistically addressing the prompt with the intention to usefully summarize the dataset for some interested audience (*which could be explicitly considered and addressed*); so, while advanced methods beyond what has currently been discussed in the course could be mentioned, they should not be a primary focus of the presentation
    
2. **[15 minutes]** Limit presentations to 4 minutes per group as follows: let the group that wants to go first presents, and then have subsequent groups extend or clarify to the previous preseantations


# Course Lecture: Week 03 LEC

## Data Visualization and Populations and Sampling

- Histograms versus box plots 
    - [panels](https://plotly.com/python/facet-plots/) (of histograms)
    - What are the pros and cons of the histogram visualization versus box plot visualization with respect to ease of comparision, examination of skewness and modeality and sample size, and detection of outliers? 

- Histograms versus kernel density estimates (KDEs)
    - [plotly](https://plotly.com/python/violin/) (not [this](https://plotly.com/python/distplot/)) VS [Waskom's](https://mwaskom.github.io) [seaborn](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)

- Comparing libraries 
    - [plotly](https://plotly.com/python/) VS [seaborn](https://seaborn.pydata.org/examples/index.html) VS [matplotlib](https://matplotlib.org) (VS [pandas](https://pandas.pydata.org/docs/user_guide/visualization.html)) VS [Hadley's](http://hadley.nz) [ggplot](https://ggplot2-book.org) (for `R` but also [available for python](https://plotnine.org)) VS [bokeh](http://bokeh.org) and [shiny](https://www.rstudio.com/products/shiny/) 
    - Data Journalism [The Pudding](https://pudding.cool) ([D3.js](https://d3js.org)) and https://informationisbeautiful.net and [Tufte's](https://www.edwardtufte.com) foundations

- $\log$ transformations, and skew, outliers, and modality 

- Samples versus populations / statistics versus parameters
    - `from scipy import stats` [normal](https://www.scribbr.com/statistics/normal-distribution/), multinomial, gamma


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


```python
import pandas as pd
import plotly.express as px

# load / reset df
df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/2e9bd5a67e09b14d01f616b00f7f7e0931515d24/data/2020/2020-07-07/coffee_ratings.csv")
df = df.rename(columns={'country_of_origin': 'origin', 'total_cup_points': 'points'})
df = df[df['points']>65]
df = df[~df['origin'].isna()]
df['origin'] = df['origin'].str.replace("?","'")
```


```python
# fix titles
df.origin = df.origin.str.replace(" (", "<br>(")
df.origin = df.origin.str.replace(", ", ",<br>")

fig = px.histogram(df, x='points', facet_col='origin', 
             facet_col_wrap=6, height=1000, facet_row_spacing=0.05)

fig.for_each_annotation(lambda a: a.update(text=a.text.replace("origin=", ""))) # fix titles
```


```python
df.origin = df.origin.str.replace("<br>", " ") # fix labels

fig = px.box(df, x='points', y="origin", height=750)

# order plot to be more visually interpretable
fig.update_yaxes(categoryorder='array', 
                 categoryarray=df.groupby("origin")['points'].mean().sort_values().index)
```


```python
# add in missing sample sizes
keys = df.origin.value_counts().index.values
vals = df.origin.value_counts().index.values + " (n="+df.origin.value_counts().values.astype(str)+")"
df.origin = df.origin.map({k:v for k,v in zip(keys,vals)})

fig = px.box(df, x='points', y="origin", height=750)
fig.update_yaxes(categoryorder='array', 
                 categoryarray=df.groupby("origin")['points'].mean().sort_values().index)
```


```python
fig = px.box(df, y='points', x="origin", points="all", height=750)
fig.update_xaxes(categoryorder='array', 
                 categoryarray=df.groupby("origin")['points'].mean().sort_values().index)
```


```python
# google "pandas remove groups with size 1"
# https://stackoverflow.com/questions/54584286/pandas-groupby-then-drop-groups-below-specified-size
df = df.groupby('origin').filter(lambda x: len(x) > 1)

fig = go.Figure()
fig.add_trace(go.Violin(x=df.origin, y=df.points,side='positive', width=5))
```


```python
# https://plotly.com/python/violin/#split-violin-plot
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Violin(x=df.origin, y=df.points, side='both', width=5))
```


```python
df = df.groupby('origin').filter(lambda x: len(x) > 30)
```


```python
fig = go.Figure()
fig.add_trace(go.Violin(x=df.origin, y=df.points,side='both', width=5))
```


```python
df = df.groupby('origin').filter(lambda x: len(x) > 100)
fig = go.Figure()
fig.add_trace(go.Violin(x=df.origin, y=df.points,side='both', width=1))
```


```python
df.groupby('origin').points.apply(list)
```


```python
import plotly.figure_factory as ff

# Group data together
hist_data = df.groupby('origin').points.apply(list).values.tolist()
group_labels = df.groupby('origin').points.apply(list).index.values

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.5]*4)
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
df
```


```python
import plotly.express as px
fig = px.histogram(df, x="points", color="origin", marginal="box",
                   color_discrete_sequence=['#A56CC1', '#A6ACEC', '#63F5EF', '#F66095'],
                   hover_data=df.columns) # "box" or "violin" or "rug"
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Group data together
hist_data = [df.groupby('origin').points.apply(list).values.sum()]
group_labels = ["<br>".join(df.groupby('origin').points.apply(list).index.values)]

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.5]*4)
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import seaborn as sns
sns.set_theme()
sns.set_palette("colorblind")#sns.set_palette("deep")#sns.set_palette("rocket")
sns.histplot(x=df.groupby('origin').points.apply(list).values.sum(), 
             stat='density', bins=30)
sns.kdeplot(x=df.groupby('origin').points.apply(list).values.sum(), 
            bw_adjust=0.5);
```

**See if you agree with you neighbor(s) -- what have you learned so far**?

- Comparing libraries 
    - [plotly](https://plotly.com/python/) VS [seaborn](https://seaborn.pydata.org/examples/index.html) VS [matplotlib](https://matplotlib.org) (VS [pandas](https://pandas.pydata.org/docs/user_guide/visualization.html)) VS [Hadley's](http://hadley.nz) [ggplot](https://ggplot2-book.org) (for `R` but also [available for python](https://plotnine.org)) VS [bokeh](http://bokeh.org) and [shiny](https://www.rstudio.com/products/shiny/) 
    - Data Journalism [The Pudding](https://pudding.cool) ([D3.js](https://d3js.org)) and https://informationisbeautiful.net and [Tufte's](https://www.edwardtufte.com) foundations


**See if you agree with you neighbor(s) -- what have you learned so far**?

**Quiz: What's the difference between these two**


```python
# https://chatgpt.com/share/66edd41b-4be0-8007-a3c8-fc5aca875e7b

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the data
df = px.data.tips()

# First figure (histogram + box, with color based on "sex" and stacked bars)
fig1 = px.histogram(df, x="total_bill", color="sex", marginal="box", hover_data=df.columns, barmode="stack")

# Second figure (histogram + box, with color based on "sex" and stacked bars)
fig2 = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="box", hover_data=df.columns, barmode="stack")

# Create a 2x2 subplot layout with adjusted row heights
fig_combined = make_subplots(
    rows=2, cols=2, 
    row_heights=[0.75, 0.25],  # 75% height for the top row, 25% for the bottom row
    subplot_titles=("Total Bill Histogram", "Total Bill vs Tip", "Box Plot 1", "Box Plot 2"),
    shared_xaxes=True, shared_yaxes=False
)

# Add the main histogram traces (both genders) from fig1 to the first subplot (top-left)
for trace in [fig1.data[0], fig1.data[2]]:  # First 2 traces are the stacked histograms
    fig_combined.add_trace(trace, row=1, col=1)

# Add the main histogram traces (both genders) from fig2 to the second subplot (top-right)
for trace in [fig2.data[0], fig2.data[2]]:  # First 2 traces are the stacked histograms
    fig_combined.add_trace(trace, row=1, col=2)

# Add only the box plot traces (bottom row)
fig_combined.add_trace(fig1.data[1], row=2, col=1)  # Box plot from fig1
fig_combined.add_trace(fig2.data[1], row=2, col=2)  # Box plot from fig2
fig_combined.add_trace(fig1.data[3], row=2, col=1)  # Box plot from fig1
fig_combined.add_trace(fig2.data[3], row=2, col=2)  # Box plot from fig2

# Update axes: remove x-axis labels from the top row and add them to the bottom row
fig_combined.update_xaxes(title_text="Total Bill", row=2, col=1)
fig_combined.update_xaxes(title_text="Total Bill", row=2, col=2)

# Update y-axis labels
fig_combined.update_yaxes(title_text="Count", row=1, col=1)
fig_combined.update_yaxes(title_text="Tip", row=1, col=2)

# Apply stacked barmode at the combined layout level
fig_combined.update_layout(
    height=500, width=800, 
    title_text="Stacked Histograms with Box Marginals (2x2 Grid)",
    barmode="stack", showlegend=False
)

# Show the combined figure
fig_combined.show()  # USE ...`.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

```


```python
df = pd.read_csv("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
fig = px.histogram(df, x="Amount")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import numpy as np

df['Log_Amount'] = np.log1p(df['Amount'])  # log1p is used for log(1 + x)
fig = px.histogram(df, x="Log_Amount", nbins=30, 
                   title="Histogram of Log-Transformed Amount")
fig.update_xaxes(title_text="Log(Amount + 1)")
fig.update_yaxes(title_text="Frequency")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

![](https://cdn.serc.carleton.edu/images/mathyouneed/geomajors/histograms/histogram_shapes.v2_744.webp)


```python
from scipy import stats

# Parameters
location_μ = 10  # population mean
scale_σ = 2  # population standard deviation

normal_distribution = stats.norm(loc=location_μ, scale=scale_σ)  # population

# Sample
n = 500
x = normal_distribution.rvs(size=n)

# Statistics
x_bar = x.mean()
ssd = x.std()
print("Sample mean statistic", x_bar.round(3), 
      "\nestimates the population mean", location_μ, "parameter\n")
print("Sample standard deviation statistic", ssd.round(3), 
      "\nestimates the population standard deviation", scale_σ, "parameter")

# Create a range for the population
grid_size = 500
support_grid = np.linspace(location_μ - 4*scale_σ, 
                           location_μ + 4*scale_σ, grid_size)

# mathematical function representing the population
pdf = normal_distribution.pdf(support_grid)  # probability density function

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, histnorm='probability density', 
                           name='Sample', opacity=0.6))
fig.add_trace(go.Scatter(x=support_grid, y=pdf, mode='lines', 
                         name='Population', line=dict(color='blue')))
fig.update_layout(title='Normal Distribution: Sample vs. Population',
                  xaxis_title='Value', yaxis_title='Density',
                  barmode='overlay')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Parameters
p = [0.1, 0.15, 0.2, 0.24, 0.1, 0.1, 0.1, 0.01]  # category probabilities (sum to 1)

multinomial_distribution = stats.multinomial(n=1, p=p)

# Sample
n = 500  # number to choose
x = multinomial_distribution.rvs(size=n)

# Calculate sample proportions
sample_proportions = x.sum(axis=0)/n

# Categories (Favorite Chips)
categories = [
    'Salsa Verde Doritos',
    'Jalapeño Cheddar Cheetos',
    'Harvest Cheddar Sun Chips',
    "Cape Cod Sea Salt and Vinegar Kettle Chips",
    'Old Dutch BBQ',
    'Sour Cream & Onion Pringles',
    "Miss Vickey's Dill Kettle Chips",
    "Classic Lays"]

fig = go.Figure()
fig.add_trace(go.Bar(x=categories, y=sample_proportions, 
                     name='Sample Proportions', opacity=0.6))
for category, proportion in zip(categories, p):
    fig.add_trace(go.Scatter(
        x=[category, category], 
        y=[0, proportion], 
        mode='lines+markers',  # Use lines and markers
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=10),
        name=category  # Use category name for legend
    ))
fig.update_layout(title='Multinomial Distribution: Sample Proportions vs. Expected Proportions',
                  xaxis_title='Favorite Chips', yaxis_title='Proportion',
                  barmode='overlay')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Parameters for Gamma distribution
alpha_α = 2  # shape parameter (α)
beta_β = 0.5  # rate parameter (β)

gamma_distribution = stats.gamma(a=alpha_α, scale=1/beta_β)  # population

# Sample
n = 500
x = gamma_distribution.rvs(size=n)

# Statistics
x_bar = x.mean()
ssd = x.std()
print("Sample mean statistic", x_bar.round(3), 
      "\nestimates the population mean", np.round(alpha_α/beta_β,3), "parameter\n")
print("Sample standard deviation statistic", ssd.round(3), 
      "\nestimates the population standard deviation", 
      np.round((alpha_α/beta_β**2)**0.5, 3), "parameter")

# Create a range for the population
grid_size = 500
support_grid = np.linspace(0, alpha_α/beta_β + 10/beta_β, grid_size)

# Mathematical function representing the population
pdf = gamma_distribution.pdf(support_grid)  # probability density function

fig = go.Figure()
fig.add_trace(go.Histogram(x=x, histnorm='probability density', 
                           name='Sample', opacity=0.6))
fig.add_trace(go.Scatter(x=support_grid, y=pdf, mode='lines', 
                         name='Population', line=dict(color='blue')))
fig.update_layout(title='Gamma Distribution: Sample vs. Population',
                  xaxis_title='Value', yaxis_title='Density',
                  barmode='overlay')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv")
df # df.columns
```


```python
df.isna().sum()
```


```python
fig = px.histogram(df, x="calories", histnorm='probability density',
                   marginal='box')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
df.loc[df["calories"]==0, "calories"] = 10
df.loc[df["calories"]>1200, "calories"] = 1200
estimated_alpha, loc, estimated_scale = stats.gamma.fit(df["calories"]) 

grid_size = 500
support_grid = np.linspace(0, estimated_alpha*estimated_scale + 10*estimated_scale, 
                           grid_size)
# Mathematical function representing the estimated population
estimated_pdf = stats.gamma.pdf(support_grid,  
                                a=estimated_alpha, scale=estimated_scale) 

# Add the estimated population
fig.add_traces(go.Scatter(x=support_grid+loc, y=estimated_pdf, mode='lines', 
                          name='Estimated Population', line=dict(color='blue')))

# Show the figure
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.box(df, x='calories', points="all")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
df["carbohydrates (% daily value)"]
```


```python
# Estimate normal distribution parameters from the 'carbohydrates (% daily value)' column
estimated_mu, estimated_sigma = stats.norm.fit(df["carbohydrates (% daily value)"].dropna())

# Create a grid of values over which to evaluate the PDF
grid_size = 500
support_grid = np.linspace(estimated_mu - 4*estimated_sigma, 
                           estimated_mu + 4*estimated_sigma, grid_size)

# Mathematical function representing the estimated population (Normal PDF)
estimated_pdf = stats.norm.pdf(support_grid, loc=estimated_mu, scale=estimated_sigma)

# Create a histogram for the 'carbohydrates (% daily value)' data
fig = px.histogram(df, x="carbohydrates (% daily value)", nbins=40, histnorm='probability density')

# Add the estimated normal population PDF as a line
fig.add_trace(go.Scatter(x=support_grid, y=estimated_pdf, mode='lines', 
                         name='Estimated Population', line=dict(color='blue')))

# Show the figure
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
for col in df.columns:
    fig = px.histogram(df, x=col, histnorm='probability density',
                       marginal='box')
    fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.histogram(df, x='serving size (oz)', nbins=40, marginal='violin')#'rug','box'
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.box(df, x='serving size (oz)', points="all")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import seaborn as sns
sns.set_theme()
sns.set_palette("rocket")#sns.set_palette("colorblind")#sns.set_palette("deep")
sns.histplot(data=df, x='serving size (oz)', stat='density', bins=30)
sns.kdeplot(data=df, x='serving size (oz)', bw_adjust=0.25);
```


```python
df['category id'].value_counts()
```


```python
df['category id'] = df['category id'].astype(object)
fig = px.bar(df, x='category id')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.box(df, x='category id', y='serving size (oz)')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
sns.set_style("whitegrid")#sns.set_style("dark")
sns.boxplot(data=df[df['category id']<110], x='category id', y='serving size (oz)', 
            hue='category id', palette="colorblind");
```


```python
fig = px.histogram(df[df['category id']<120], x='serving size (oz)', 
                   facet_col='category id', facet_col_wrap=5)
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))  # Keep only the value, remove 'category id='
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```



# Course Homework: Week 03 HW

## STA130 Homework 03 

Please see the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) for the list of topics covered in this homework assignment, and a list of topics that might appear during ChatBot conversations which are "out of scope" for the purposes of this homework assignment (and hence can be safely ignored if encountered)

<details class="details-example"><summary style="color:blue"><u>Introduction</u></summary>

### Introduction
    
A reasonable characterization of STA130 Homework is that it simply defines a weekly reading comprehension assignment. 
Indeed, STA130 Homework essentially boils down to completing various understanding confirmation exercises oriented around coding and writing tasks.
However, rather than reading a textbook, STA130 Homework is based on ChatBots so students can interactively follow up to clarify questions or confusion that they may still have regarding learning objective assignments.

> Communication is a fundamental skill underlying statistics and data science, so STA130 Homework based on ChatBots helps practice effective two-way communication as part of a "realistic" dialogue activity supporting underlying conceptual understanding building. 

It will likely become increasingly tempting to rely on ChatBots to "do the work for you". But when you find yourself frustrated with a ChatBots inability to give you the results you're looking for, this is a "hint" that you've become overreliant on the ChatBots. Your objective should not be to have ChatBots "do the work for you", but to use ChatBots to help you build your understanding so you can efficiently leverage ChatBots (and other resources) to help you work more efficiently.<br><br>

</details>

<details class="details-example"><summary style="color:blue"><u>Instructions</u></summary>

### Instructions
    
1. Code and write all your answers (for both the "Pre-lecture" and "Post-lecture" HW) in a python notebook (in code and markdown cells) 
    
> It is *suggested but not mandatory* that you complete the "Pre-lecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
    
2. Paste summaries of your ChatBot sessions (including link(s) to chat log histories if you're using ChatGPT) within your notebook
    
> Create summaries of your ChatBot sessions by using concluding prompts such as "Please provide a summary of our exchanges here so I can submit them as a record of our interactions as part of a homework assignment" or, "Please provide me with the final working verson of the code that we created together"
    
3. Save your python jupyter notebook in your own account and "repo" on [github.com](github.com) and submit a link to that notebook though Quercus for assignment marking<br><br>

</details>

<details class="details-example"><summary style="color:blue"><u>Prompt Engineering?</u></summary>
    
### Prompt Engineering? 
    
The questions (as copy-pasted prompts) are designed to initialize appropriate ChatBot conversations which can be explored in the manner of an interactive and dynamic textbook; but, it is nonetheless **strongly recommendated** that your rephrase the questions in a way that you find natural to ensure a clear understanding of the question. Given sensible prompts the represent a question well, the two primary challenges observed to arise from ChatBots are 

1. conversations going beyond the intended scope of the material addressed by the question; and, 
2. unrecoverable confusion as a result of sequential layers logial inquiry that cannot be resolved. 

In the case of the former (1), adding constraints specifying the limits of considerations of interest tends to be helpful; whereas, the latter (2) is often the result of initial prompting that leads to poor developments in navigating the material, which are likely just best resolve by a "hard reset" with a new initial approach to prompting.  Indeed, this is exactly the behavior [hardcoded into copilot](https://answers.microsoft.com/en-us/bing/forum/all/is-this-even-normal/0b6dcab3-7d6c-4373-8efe-d74158af3c00)...

</details>


### Marking Rubric (which may award partial credit) 

- [0.1 points]: All relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] are reported within the notebook
- [0.2 points]: Assignment completion confirmed by visual submission for "2" 
- [0.3 points]: Evaluation of written communication for "3" 
- [0.1 points]: Correct answers for "4"
- [0.3 points]: Evidence of meaningful activity for "6"

<!-- - [0.1 points]: Assignment completion confirmed by ChatBot interaction summaries for "5" -->


### "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]


#### 1. Use _fig.add_[h/v]line()_ and *fig.add_[h/v]rect()* to mark, respspectively, location (mean and median) and scale (range, interquartile range, and a range defined by two standard deviations away from the mean in both directions) of *flipper_length_mm* for each `species` onto `plotly` histograms of _flipper_length_mm_ for each `species` in the penguins dataset<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> The code referenced above [`fig.add_[h/v]line()`](https://plotly.com/python/horizontal-vertical-shapes/) and [`fig.add_[h/v]rect()`](https://plotly.com/python/line-charts/) refer to `fig.add_hline()` and `fig.add_hline()` and `fig.add_hrect()` and `fig.add_vrect()` which overly lines rectangles onto a figure using a slightly different interface 
> 
> - _There are several considerations in this problem..._
>     - _The histograms can be on the same figure, on separate figures, or separated into different panels in the same figure_
>     - _The elements within a figure should be well annotated, probobably using a so-called legend to help make sure annotations don't overlap each other and are clear and readible_
> - _There are several ways to approach this problem..._
>     - _You will likely be very pleased when you run the code returned to you as the result of pasting this question in as a prompt into a ChatBot session; but, you will also likely need to interact with the ChatBot to ask for adjustments to the code which give a final satisfactory figure (and this is the recommended approach to get the experience this problem intends you to have)_
>     - _**When using a ChatBot, if the code provided by your ChatBot results in an error, show the error to your ChatBot and iterate this process with the adjusted "fixed" code provided by the ChatBot... this process usually converges some something workable that's pretty close to what you were going for**_
>     - <u>**And don't forget, a ChatBot can explain what how code it provides works, if you ask it to...**</u>
>     - _You could alternatively figure out how to code this plot up for yourself by looking at the provided documentation links and perhaps using some additional google searchers or ChatBot queries to help out with specific issues or examples; and, if you end up interested in figuring out a little more how the code works that's great and definitely feel free to go ahead and do so, but at this stage the point of this problem is to understand the general ideas of figures themselves as opposed to being an expert about the code that generated them_
    
</details>


#### 2. Transition your ChatBot session from the previous problem to repeat the previous problem, but this time using [_seaborn_ **kernel density estimation** (KDE) plots](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) to produce the desired figures organized in row of three plots<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> The `seaborn` library extends `matplotlib` so [_ax.axhspan(...)_](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axhspan_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axhspan-demo-py) or [_ax.fill_between(...)_](https://matplotlib.org/stable/gallery/lines_bars_and_markers/span_regions.html) from `matplotlib` could be combined with the `seaborn` KDE plot... this might be something to share with your ChatBot if it [tries to keep using _plotly_ or a KDE function rather than a _plotly_](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/GPT/SLS/00001_gpt3p5_plotlyseaborn_plotting.md) plotting functionality...
> 
> - _When using a ChatBot, if the code provided by your ChatBot results in an error, show the error to your ChatBot and iterate this process with the adjusted "fixed" code provided by the ChatBot... this process usually converges some something workable that's pretty close to what you were going for_
> - _**Also consider the ways that you might be able to split up the instructions for the ChatBot into multiple steps, creating a sequence of additional directions and extensions along the way as you mold the figure more and more into a form increasingly matching your desired output.**_
> - And don't forget, a ChatBot can explain what how code it provides works, if you ask it to...
> 
> The technical details of the following are beyond the scope of STA130, but if you were interested, you could very briefly examine the [_seaborn_ themes](https://seaborn.pydata.org/tutorial/aesthetics.html) based on `sns.set_style()` and `sns.set_theme()` and [_colors_](https://seaborn.pydata.org/tutorial/color_palettes.html) based on the `palette` parameter, e.g.,
> 
> ```python
> sns.set_style("whitegrid") # sns.set_style("dark")
> # `sns.set_palette()` exists but functions often access and set that directly
> sns.boxplot(..., hue='column', palette="colorblind") 
> ```    
> 
> and then attempt to interact with the ChatBot to change the coloring of the figure to something that you like and looks more clear to you... 

</details>


#### 3. Search online for some images of **box plots**, **histograms**, and **kernel density estimators** (perhaps for the same data set); describe to a ChatBot what you think the contrasting descriptions of these three "data distribution" visualization methods are; and then see if the ChatBot agrees and what "pros and cons" list of these three "data distribution" visualization methods your ChatBot can come up with; finally, describe your preference for one or the other and your rationale for this preference<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> This 
> 
> The details of the ["kernel"](https://en.wikipedia.org/wiki/Kernel_density_estimation) and how it works in [kernel density estimation](https://plotly.com/python/violin/#split-violin-plot) are beyond the scope of STA130; but, there is typically a so-called "bandwidth" **argument** (e.g., `bw_adjust` in [_seaborn_](https://stackoverflow.com/questions/37932283/confusion-with-bandwidth-on-seaborns-kdeplot)) that "controls the width of the kernel" which is analgous to the "number of bins parameter" of a histogram (e.g., `nbins` in [_plotly_](https://www.google.com/search?client=safari&rls=en&q=plotly+nbins&ie=UTF-8&oe=UTF-8))  <!-- 4. Report on your preferences between `plotly` and `seaborn` in terms of usability and the general visual aestetics -->
> 
> _Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)_
    
</details>

#### 4. Run the code below and look at the resulting figure of distrubutions and then answer the following questions

1. Which datasets have similar means and similar variances
2. Which datasets have similar means but quite different variances
3. Which datasets have similar variances but quite different means
4. Which datasets have quite different means and quite different variances
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> Can you answer these questions immediately? If not, first review what the basic ideas of **sample means** and **sample standard deviations** (and **sample variances**) are. Their mathematical definitions are given below, and are useful for understanding the intuition of these concepts in terms of "averages" of things, like "observations" or "squared differences" (and then perhaps square-rooted). But there are other ways to "intuitively visually" understand **sample means** and **sample standard deviations** (and **sample variances**) which a ChatBot would be able to discuss with you.
>
> - sample mean $\displaystyle \bar x = \frac{1}{n}\sum_{i=1}^n x_i$ 
> - sample variance $\displaystyle s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2$
> - sample standard deviation $\displaystyle s = \sqrt{s^2}$
>
> It's potentially maybe possible that you or a ChatBot could answer these questions by looking at the code that produced the data you're considering. But if you're trying to check and understand things that way, you should instead consider just calculate the statistics that answer the questions themselves...
> - `np.mean(df.col)` or `df.col.mean()`
> - `np.std(df.col, dof=1)` / `np.var(df.col, dof=1)` or `df.col.std(dof=1)` / `df.col.var(dof=1)`
>
> _If you are resorting to calculating the statistics that answer the questions, try to understand the answers after you have them... just getting the "right" answers kind of defeats the point of this exercise..._
>
> - The difference between trying to answer this question using the code that produced the data versus calculating the statistics from the data comes down to the difference between **parameters** and **statistics**, but this will be discussed in the lecture... in the meantime, howevever, if you're curious about this... you could consider prompting a ChatBot to explain the difference between **parameters** and **statistics**...
>     - ... this would naturally lead to some discussion of the relationship between **populations** and **samples**, and from there it would only be a little further to start working to understand the relationship between **statistics** and **parameters** and how they connect to *populations* and *samples* (and hence each other)...    
    
</details>  


```python
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500
data1 = stats.uniform.rvs(0, 10, size=n)
data2 = stats.norm.rvs(5, 1.5, size=n)
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), stats.norm.rvs(8, 0.5, size=int(n/2))]
data4 = stats.norm.rvs(6, 0.5, size=n)

fig = make_subplots(rows=1, cols=4)

fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=4)

fig.update_layout(height=300, width=750, title_text="Row of Histograms")
fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)
fig.update_xaxes(range=[-0.5, 10.5])

for trace in fig.data:
    trace.xbins = dict(start=0, end=10)
    
# This code was produced by just making requests to Microsoft Copilot
# https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/COP/SLS/0001_concise_makeAplotV1.md

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Pre-lecture VS Post-lecture HW

Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: the next questions will just continue working on data visualization related topics, so, it's just a choice whether or not you want to work a head a little bit... 

- The previous suggestions regarding **parameters** versus **statistics** would be a very good thing to look at carefully in preparation for the upcoming lecture...
    
*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details> 



### "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]

#### 5. Start a new ChatBot session to explore the general relationship between the *mean* and *median* and "right" and "left" skewness (and why this is); what the following code does and how it works; and then explain (in your own words) the relationship between the *mean* and *median* and "right" and "left" skewness and what causes this, using and extending the code to demonstrate your explanation through a sequence of notebook cells.<br>

```python
from scipy import stats
import pandas as pd
import numpy as np
  
sample1 = stats.gamma(a=2,scale=2).rvs(size=1000)
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data")
# USE `fig1.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

sample1.mean()
np.quantile(sample1, [0.5]) # median

sample2 = -stats.gamma(a=2,scale=2).rvs(size=1000)
```

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> You could start this session perhaps something like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/GPT/SLS/00003_GPT3p5_meanVmedian.md)?
> 
> _Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)..._

</details> 



#### 6. Go find an interesting dataset and use summary statistics and visualizations to understand and demonstate some interesting aspects of the data<br>

1. Your approach should likely follow what was suggested for the **Week 02 TUT Communication Activity from TUT**
2. In the **Week 03 TUT Communication Activity from TUT** you will be put in groups and determine which group members dataset introduction will be presented by the group

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> A good place to browse datasets is [TidyTuesday](https://github.com/rfordatascience/tidytuesday/blob/master/README.md) as working with ChatBots to find unconventional and entertaining datasets is not particularly productive and only seems to end up with the datasets seen here and other (more interesting?) suggestions like [iris](https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv), [superheros](https://raw.githubusercontent.com/steview-d/superhero-dashboard/master/static/data/superheroData.csv), [hauntings](https://raw.githubusercontent.com/andreamoleri/Hauntings/main/hauntings.csv), [bigfoot](https://raw.githubusercontent.com/hannahramirez/BigfootVsUfos/main/bigfoot_mod.csv), [ufos](https://raw.githubusercontent.com/hannahramirez/BigfootVsUfos/main/ufo_mod.csv), [sharks](https://raw.githubusercontent.com/IbaiGallego/DataCleaning_SharkAttack/main/data/jaws.csv), [legos](https://raw.githubusercontent.com/seankross/lego/master/data-tidy/legosets.csv), [bees](https://gist.githubusercontent.com/bootshine2/ba15d3cb38e2ed31129aeca403405a12/raw/10949901cd8a6a75aa46c86b804c42ff410f929e/Bee%2520Colony%2520Loss.csv), [housing](https://raw.githubusercontent.com/slavaspirin/Toronto-housing-price-prediction/master/houses_edited.csv), and [gapminder](https://raw.githubusercontent.com/kirenz/datasets/master/gapminder.csv)
> ```python
> # Maybe something like this? Feel free to use this one 
> # if it strikes your fancy after look around a bit
> import pandas as pd
> df = pd.read_csv("https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv")
> df # df.columns
> ```

</details>

#### 7. Watch the classic [Gapminder Video](https://www.youtube.com/watch?v=jbkSRLYSojo), then have a look at the [_plotly_ version](https://plotly.com/python/animations/) and recreate the animation (perhaps after optionally exploring and changing the [style](https://plotly.com/python/templates/), if you wish)

#### 8. Provide a second version of the figure from the previous problem where you edit the `fig = px.scatter()` function from the Gapminder code so that `x` is "percent change", `y` is "rank", `size` is "percent", and `color`="sex", `animation_frame` is "year", and `animation_group` and `hover_name` are "name". Then use `size_max=50`, `range_x=[-0.005,0.005])` and remove the `log_x=True` and `range_y` parameters

> ```python
> bn = pd.read_csv('https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv')
> bn['name'] = bn['name']+" "+bn['sex'] # make identical boy and girl names distinct
> bn['rank'] = bn.groupby('year')['percent'].rank(ascending=False)
> bn = bn.sort_values(['name','year'])
> # the next three lines create the increaes or decrease in name prevalence from the last year 
> bn['percent change'] = bn['percent'].diff()
> new_name = [True]+list(bn.name[:-1].values!=bn.name[1:].values)
> bn.loc[new_name,'percentage change'] = bn.loc[new_name,'percent'] 
> bn = bn.sort_values('year')
> bn = bn[bn.percent>0.001] # restrict to "common" names
> fig = px.scatter(bn, x="", y="", animation_frame="", animation_group="",
>                  size="", color="", hover_name="",size_max=50, range_x=[-0.005,0.005]) # range_y removed
> fig.update_yaxes(autorange='reversed') # this lets us put rank 1 on the top
> fig.show(renderer="png") # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
> ```


#### 9. Have you reviewed the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br><br>
  
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
    
</details>

# Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course

<br><details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations

|![](https://handsondataviz.org/images/14-detect/gdp-baseline-merged-annotated.png)|
|-|
| |

Mark Twain's statment that, "There are lies, damn lies, and statistics", reflects a general skepticism towards statistical analysis that has been reinforced through through popular books such as [How to Lie with Statistics](https://en.wikipedia.org/wiki/How_to_Lie_with_Statistics). One place "statistics" can be used to decieve is through misuse of charts.  As discussed [here](https://handsondataviz.org/how-to-lie-with-charts.html) and many other places, a primary tactic that can be used to give a misleading impression using a chart is the manipulation of axes or the addition of additional dimensions which distort the meaning of size. **What are the problems with the following graphs?**

|![](https://images.ctfassets.net/jicu8fwm4fvs/260tj0wxTFCAlbf4yTzSoy/2b002a49921831ab0dc05415616a1652/blog-misleading-gun-deaths-graph.jpeg)|![](https://photos1.blogger.com/blogger/5757/110/1600/macgraph.jpg)|
|-|-|
| | |

</details>    

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>
   
### Current Course Project Capability Level
    
**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
    
At this point in the course you should be able to create a `for` loop to iterate through and provide **visualizations** of some of the interesting columns in the course project data

1. Create a `for` loop with a **conditional logic structure** that appropriately controls the kind of visualization that gets made for a given column of data based on its data type

*Being able run your code with different subsets (of different types) of columns demonstrates the desirability of the programming design principle of "polymorphism" (which means "many uses") which states that code is best when it's "resuable" for different purposes... such as automatically providing the appropriate visualizations as interest in different variables dynamically changes...* 
    
</details>            
