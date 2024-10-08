# STA130 LEC 04 (Sep 30)
## Confidence Intervals / Statistical Inference

- Populations VS Samples 
    - `normal_distribution = stats.norm(loc=mean_μ, scale=std_σ)`
    - `normal_distribution.rvs(size=n)`
- Gamma Distributions VS Normal Distributions
    - `gamma_distribution = \`<br>
      `stats.gamma(shape=shape_α, scale=scale_θ)`
    - `gamma_distribution.rvs(size)`    
- Parameters VS Statistics
    - `shape_α` and `scale_θ`
    - `gamma_distribution.mean()` and `gamma_distribution.std()`
    - `gamma_distribution.rvs(size).mean()`    
- Estimation
    - Model Fitting with `.fit(data)`
- Inference
    - Bootstrapping `df['col'].sample(n=n, replace=True)`
    - Confindence intervals `np.quantile(bootstrapped_stats,[0.025,0.975])`
- Confidence Level


```python
from scipy import stats

# population
population_parameter_alpha_α = 2
population_parameter_theta_θ = 4
gamma_distribution = \
  stats.gamma(a=population_parameter_alpha_α, 
              scale=population_parameter_theta_θ)

# sample
n = 100 # adjust and experiment with this
# np.random.seed(130)
x = gamma_distribution.rvs(size=n) # "x" is a sample
# print(x)

# mean
print("The sample mean for the current sample is", x.mean()) 
# the sample mean "x-bar" is a (sample) "statistic" (not a "parameter")
# "x-bar" is the "average" of the numbers in a sample
```


```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

n = 100 # adjust and experiment with this
# np.random.seed(130)
x = gamma_distribution.rvs(size=n) 

fig = px.histogram(pd.DataFrame({'sampled values': x}), x='sampled values',
                   histnorm='probability density') # so the scale matches the pdf below
fig.add_vline(x=x.mean(), line_dash="dot", annotation_text='Sample mean '+str(x.mean()))

support = np.linspace(0,50,500)
fig.add_trace(go.Scatter(x=support, y=gamma_distribution.pdf(support), 
                         mode='lines', name='Poulation Model<br>(gamma distribution)'))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```


```python
print("shape parameter α is ", population_parameter_alpha_α, "\n",
      "scale parameter θ is ", population_parameter_theta_θ, "\n",
      "shape α times scale θ is ", population_parameter_alpha_α*population_parameter_theta_θ, "\n",
      "population mean (an 'unnamed' gamma population parameter) is ", gamma_distribution.mean(), "\n", 
      "gamma 'unnamed' μ = α * θ", sep="")
```


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


```python
n = 100 # adjust and experiment with this
x = gamma_distribution.rvs(size=n) 

import plotly.figure_factory as ff

hist_data = [x]
group_labels = ['Gamma Sample']
fig = ff.create_distplot(hist_data, group_labels, show_hist=True, 
                         show_rug=False, bin_size=3)

fig.add_trace(go.Scatter(x=support, y=gamma_distribution.pdf(support), 
                         mode='lines', line=dict(width=4), name='Poulation Model<br>(gamma distribution)'))

a, loc, scale = stats.gamma.fit(x, floc=0)
fig.add_trace(go.Scatter(x=support, 
                         y=stats.gamma(a=a, scale=scale).pdf(support),
                         mode='lines', line=dict(dash='dot', width=4), name='Estimated Poulation Model<br>(fitted gamma distribution)'))

fig.show()
```


```python
n = 100 # adjust and experiment with this
x = gamma_distribution.rvs(size=n) 

# `floc=0` means `loc` does not shift gamma left or right
a, loc, scale = stats.gamma.fit(x, floc=0)

print("Actual shape `population_parameter_alpha_α` is ", 
      gamma_distribution.kwds['a'], "\n",
      "Actual scale `population_parameter_theta_θ` is ",
      gamma_distribution.kwds['scale'], "\n",
      "Esimated population shape parameter is ", a, "\n",
      "Esimated population scale parameter is ", scale, "\n",
      "Esimated population 'unnamed' mean parameter is ", a*scale, "\n",
      "Esimated 'unnamed' mean equal to shape*scale is ",
      stats.gamma(a=a, scale=scale).mean(), sep="")
```


```python
# load / reset df
df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/2e9bd5a67e09b14d01f616b00f7f7e0931515d24/data/2020/2020-07-07/coffee_ratings.csv")
df = df.rename(columns={'country_of_origin': 'origin', 'total_cup_points': 'points'})

df = df[df['points']>65] # ignore some very low scores
df = df[~df['origin'].isna()] # remove rows with unknown origin

df['origin'] = df['origin'].str.replace("?","'") # fix character encoding issue
df['origin_original'] = df.origin.copy().values # save original (corrected) names
```


```python
# just run to get to the plot -- we are not interested in reviewing this code now

# add line breaks to titles
df.origin = df.origin_original.str.replace(" (", "<br>(").replace(", ", ",<br>")

fig = px.histogram(df[df.origin=='Guatemala'][1:], x='points')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
df.loc[df.origin=='Guatemala', 'points'] = \
    -df.loc[df.origin=='Guatemala', 'points'] 
df = df.loc[df.origin=='Guatemala']
df = df[1:].copy()
fig = px.histogram(df[df.origin=='Guatemala'], x='points')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# shift it to be positive
df.loc[df.origin=='Guatemala', 'points'] += 100
```


```python
fig = px.histogram(df[df.origin=='Guatemala'], x='points')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
a, loc, scale = \
stats.gamma.fit(df.loc[df.origin=='Guatemala', 'points'].values)
```


```python
hist_data = [df.loc[df.origin=='Guatemala', 'points'].values]
group_labels = ['Sample (Guatemala)']
fig = ff.create_distplot(hist_data, group_labels, show_hist=True, 
                         show_rug=False, bin_size=1)

support = np.linspace(0,15,500)

fig.add_trace(go.Scatter(x=support+loc, 
                         y=stats.gamma(a=a, scale=scale).pdf(support),
                         mode='lines', line=dict(dash='dot', width=4), name='Estimated Poulation Model<br>(fitted gamma distribution)'))

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
n = (df.origin=='Guatemala').sum()
print("Original sample size", n)

simulations = 1000
a_simulations = np.zeros(simulations)
loc_simulations = np.zeros(simulations)
scale_simulations = np.zeros(simulations)
xbar_simulations = np.zeros(simulations)

for i in range(simulations):
    simulated_sample = stats.gamma(a=a, loc=loc, scale=scale).rvs(size=n)
    a_simulations[i], \
    loc_simulations[i], \
    scale_simulations[i] = stats.gamma.fit(simulated_sample)
    xbar_simulations[i] = simulated_sample.mean()
```


```python
import plotly.subplots as sp
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Shape (a)", "Location (loc)", "Scale", "Sample Mean (x̄)"))
fig.add_trace(go.Histogram(x=a_simulations, nbinsx=30, name="Shape (a)", showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=loc_simulations, nbinsx=30, name="Location (loc)", showlegend=False), row=1, col=2)
fig.add_trace(go.Histogram(x=scale_simulations, nbinsx=30, name="Scale", showlegend=False), row=2, col=1)
fig.add_trace(go.Histogram(x=xbar_simulations, nbinsx=30, name="Sample Mean (x̄)", showlegend=False), row=2, col=2)
fig.update_layout(height=500, width=600, title_text="Histograms of Fitted Gamma Parameters and Sample Mean")
fig.update_xaxes(title_text="Shape (a)", row=1, col=1)
fig.update_xaxes(title_text="Location (loc)", row=1, col=2)
fig.update_xaxes(title_text="Scale", row=2, col=1)
fig.update_xaxes(title_text="Sample Mean (x̄)", row=2, col=2)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
n = (df.origin=='Guatemala').sum()
print("Original sample size", n)

bootstrap_samples = 1000
a_bootstrapped = np.zeros(bootstrap_samples)
loc_bootstrapped = np.zeros(bootstrap_samples)
scale_bootstrapped = np.zeros(bootstrap_samples)
xbar_bootstrapped = np.zeros(bootstrap_samples)

for i in range(bootstrap_samples):
    simulated_sample = df.sample(n=n, replace=True).points
    a_bootstrapped[i], \
    loc_bootstrapped[i], \
    scale_bootstrapped[i] = stats.gamma.fit(simulated_sample)
    xbar_bootstrapped[i] = simulated_sample.mean()
```


```python
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Shape (a)", "Location (loc)", "Scale", "Sample Mean (x̄)"))
fig.add_trace(go.Histogram(x=a_bootstrapped, nbinsx=30, name="Bootstrapped Shape (a)", showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=loc_bootstrapped, nbinsx=30, name="Bootstrapped Location (loc)", showlegend=False), row=1, col=2)
fig.add_trace(go.Histogram(x=scale_bootstrapped, nbinsx=30, name="Bootstrapped Scale", showlegend=False), row=2, col=1)
fig.add_trace(go.Histogram(x=xbar_bootstrapped, nbinsx=30, name="Bootstrapped Sample Mean (x̄)", showlegend=False), row=2, col=2)
fig.update_layout(height=500, width=600, title_text="Histograms of Bootstrapped Gamma Parameters and Sample Mean")
fig.update_xaxes(title_text="Bootstrapped Shape (a)", row=1, col=1)
fig.update_xaxes(title_text="Bootstrapped Location (loc)", row=1, col=2)
fig.update_xaxes(title_text="Bootstrapped Scale", row=2, col=1)
fig.update_xaxes(title_text="Bootstrapped Sample Mean (x̄)", row=2, col=2)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Define bins (same for both original and bootstrapped histograms)
bin_edges_a = np.histogram_bin_edges(a_simulations, bins=30)
bin_edges_loc = np.histogram_bin_edges(loc_simulations, bins=30)
bin_edges_scale = np.histogram_bin_edges(scale_simulations, bins=30)
bin_edges_xbar = np.histogram_bin_edges(xbar_simulations, bins=30)

# Create 2x2 subplots
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Shape (a)", "Location (loc)", "Scale", "Sample Mean (x̄)"))

# Overlay original and bootstrapped histograms with transparency and forced same bins
# Plot Shape (a)
fig.add_trace(go.Histogram(x=a_simulations, xbins=dict(start=bin_edges_a[0], end=bin_edges_a[-1], size=np.diff(bin_edges_a)[0]),
                           name="Shape (a)", opacity=0.5, marker_color='blue', showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=a_bootstrapped, xbins=dict(start=bin_edges_a[0], end=bin_edges_a[-1], size=np.diff(bin_edges_a)[0]),
                           name="Bootstrapped Shape (a)", opacity=0.5, marker_color='red', showlegend=False), row=1, col=1)

# Plot Location (loc)
fig.add_trace(go.Histogram(x=loc_simulations, xbins=dict(start=bin_edges_loc[0], end=bin_edges_loc[-1], size=np.diff(bin_edges_loc)[0]),
                           name="Location (loc)", opacity=0.5, marker_color='blue', showlegend=False), row=1, col=2)
fig.add_trace(go.Histogram(x=loc_bootstrapped, xbins=dict(start=bin_edges_loc[0], end=bin_edges_loc[-1], size=np.diff(bin_edges_loc)[0]),
                           name="Bootstrapped Location (loc)", opacity=0.5, marker_color='red', showlegend=False), row=1, col=2)

# Plot Scale
fig.add_trace(go.Histogram(x=scale_simulations, xbins=dict(start=bin_edges_scale[0], end=bin_edges_scale[-1], size=np.diff(bin_edges_scale)[0]),
                           name="Scale", opacity=0.5, marker_color='blue', showlegend=False), row=2, col=1)
fig.add_trace(go.Histogram(x=scale_bootstrapped, xbins=dict(start=bin_edges_scale[0], end=bin_edges_scale[-1], size=np.diff(bin_edges_scale)[0]),
                           name="Bootstrapped Scale", opacity=0.5, marker_color='red', showlegend=False), row=2, col=1)

# Plot Sample Mean (x̄)
fig.add_trace(go.Histogram(x=xbar_simulations, xbins=dict(start=bin_edges_xbar[0], end=bin_edges_xbar[-1], size=np.diff(bin_edges_xbar)[0]),
                           name="Sample Mean (x̄)", opacity=0.5, marker_color='blue', showlegend=False), row=2, col=2)
fig.add_trace(go.Histogram(x=xbar_bootstrapped, xbins=dict(start=bin_edges_xbar[0], end=bin_edges_xbar[-1], size=np.diff(bin_edges_xbar)[0]),
                           name="Bootstrapped Sample Mean (x̄)", opacity=0.5, marker_color='red', showlegend=False), row=2, col=2)

# Update layout to overlay the histograms
fig.update_layout(height=500, width=600, title_text="Overlaid Histograms with Forced Same Bins", barmode='overlay')

# Update x-axis labels
fig.update_xaxes(title_text="Shape (a)", row=1, col=1)
fig.update_xaxes(title_text="Location (loc)", row=1, col=2)
fig.update_xaxes(title_text="Scale", row=2, col=1)
fig.update_xaxes(title_text="Sample Mean (x̄)", row=2, col=2)

# Show the figure
fig.show()
```


```python
np.quantile(xbar_simulations,[0.025,0.975])
```


```python
np.quantile(xbar_bootstrapped,[0.025,0.975])
```
