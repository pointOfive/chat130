# STA130 LEC 03 (Sep 23)

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


```python

```
