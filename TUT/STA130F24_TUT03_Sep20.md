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
    
