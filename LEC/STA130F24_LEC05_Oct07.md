# STA130 LEC 07 (Oct 07)
## (Statistical) Hypothesis Testing

- **Inference** VS **Estimation** VS **Null Hypotheses**
- "Proving" **bootstrapped confidence intervals** (from last class)

- A reminder/review of how to use ChatBots
    - The [Bad](https://chatgpt.com/share/66fd24f3-3a00-8007-a67b-e389bb4bb940), the [Good](https://chatgpt.com/share/66fd2ea8-0dd8-8007-bb44-41e63de657be), and the [Lovely?](https://en.wikipedia.org/wiki/The_Good,_the_Bad_and_the_Ugly)

- Sampling Distribution of a statistic **under the null hypothesis**
    - **p-values**
    - VS the **bootstrapped Sampling Distribution** of a statistic
    - VS $\alpha$-**significance levels** and Type I and II Errors

- **Don't f@#k this up**; or, "mistakes you can make to make you dumb"
    - Just interpret p-values in terms of _Strength of Evidence_ against the null hypothesis

    - Or just use **confidence intervals**


## Inference VS Estimation VS Null Hypotheses

### Let's consider Week 04 HWq8 again


```python
import pandas as pd

patient_data = pd.DataFrame({
    "PatientID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [45, 34, 29, 52, 37, 41, 33, 48, 26, 39],
    "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    "InitialHealthScore": [84, 78, 83, 81, 81, 80, 79, 85, 76, 83],
    "FinalHealthScore": [86, 86, 80, 86, 84, 86, 86, 82, 83, 84]
})

patient_data['HealthScoreChange'] = patient_data.FinalHealthScore-patient_data.InitialHealthScore

patient_data
```


```python
import plotly.express as px

# Reshape the data to have both scores in one column for plotting
health_scores = patient_data.melt(id_vars=["PatientID", "Age", "Gender"], 
                                  value_vars=["InitialHealthScore", "FinalHealthScore"], 
                                  var_name="ScoreType", 
                                  value_name="HealthScore")
fig = px.box(health_scores, x="HealthScore", y="ScoreType", title="Boxplot of Initial and Final Health Scores")
fig.show()
```

**Questions** 

1. What's the variance?
2. This is probably a good visualization, but why might it not be?
3. Do you think the vaccine does anything? 
4. What's your **estimate** of the effect of the vaccine **on average**?
5. Is that a **parameter** (of what population?) or a **statistic** (of what sample?) that **estimates** the **parameter**?


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


```python
import plotly.graph_objects as go

# Calculate means
mean_initial = patient_data['InitialHealthScore'].mean()
mean_final = patient_data['FinalHealthScore'].mean()

# Reshape the data for histogram plotting
health_scores = patient_data.melt(id_vars=["PatientID", "Age", "Gender"], 
                                  value_vars=["InitialHealthScore", "FinalHealthScore"], 
                                  var_name="ScoreType", 
                                  value_name="HealthScore")

# Create histograms for Initial and Final Health Scores
fig = px.histogram(health_scores, x="HealthScore", color="ScoreType", 
                   title="Histogram of Initial and Final Health Scores", 
                   barmode='overlay', nbins=10)

# Add vertical lines for the sample means
fig.add_vline(x=mean_initial, line_dash="dash", line_color="blue", 
              annotation_text=f"Mean Initial: {mean_initial:.2f}", annotation_position="top left")

fig.add_vline(x=mean_final, line_dash="dash", line_color="red", 
              annotation_text=f"Mean Final: {mean_final:.2f}", annotation_position="top right")

# Show the figure
fig.show()
```


```python
mean_change = patient_data['HealthScoreChange'].mean()

fig = px.histogram(patient_data, x='HealthScoreChange', nbins=10, 
                   title="Histogram of Health Score Change")

fig.add_vline(x=mean_change, line_dash="dash", line_color="red", 
              annotation_text=f"Mean: {mean_change:.2f}", annotation_position="top right")
fig.update_xaxes(tickmode='linear', tick0=patient_data['HealthScoreChange'].min(), dtick=1)
fig.show()
```

**Questions** 

1. What does it mean to ask, "Does the vaccine have an effect?"
2. What does it mean to ask, "Does the vaccine help?"

**Hint: why should we add the phrase _on average_ here?**

3. What does it mean if we add the phrase **on average** here?
4. Are we talking about a **population paramter** or a **sample statistic** when we say **on average**?
5. Is that really different than talking about a **sample average** (despite how similarly close they sound)?
6. What is the **sample mean** used for relative to the **population mean**?


```python
# Evidence against null hypothesis using confidence intervals

import numpy as np

# Bootstrapping
# np.random.seed(130)  # make simulation reproducible
number_of_simulations = 1000 
n_size = len(patient_data)  # 10
bootstrap_means = np.zeros(1000)  # array to store bootstrapped means

for i in range(number_of_simulations):
    
    # bootstrap sample size is the same ("apples to apples") as the original sample size
    sample = patient_data.sample(n=n_size, replace=True)  # `replace=True`!!
    bootstrap_means[i] = sample['HealthScoreChange'].mean()  # bootstrapped mean

# Calculating the 95% confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
ci_lower, ci_upper
```

**Questions**

1. How did we get this?
2. What is this?
3. What do we believe this tells us?
4. How do we state what be believe this tells us?
5. This IS NOT **estimation**: what would **estimation** be here?
6. This IS **inference**: why is **inference** MUCH MORE interesting than **estimation**? 
7. What is **variance**? What does it tell us?
8. What **standard error**? What does it tell us?
9. How does **standard error** factor in to what we have here?
10. What does a **null hypothesis** have to do with anything here?
11. What is a "natural" **null hypothesis** to consider here?
12. How do we evaluate the **null hypothesis** using what we have here?
13. Why do we believe what we believe this tells us (re: 3)? 


```python
# figure for demonstration only: code details not of primary concern

import plotly.graph_objs as go
import plotly.figure_factory as ff

hist_data = [bootstrap_means]
group_labels = ['Bootstrapped<br>Sampling Distribution<br>of the Sample Mean']
fig = ff.create_distplot(hist_data, group_labels, 
                         show_hist=True, show_rug=False, bin_size=0.4)

# Add a line for the lower confidence interval
ci_y = 0.35  # Adjust height as needed
fig.add_shape(type="line", x0=ci_lower, y0=0, x1=ci_lower, y1=ci_y,
              line=dict(color="Red", width=2), name="95% CI Lower")
# Add a line for the upper confidence interval
fig.add_shape(type="line", x0=ci_upper, y0=0, x1=ci_upper, y1=ci_y,
              line=dict(color="Red", width=2), name="95% CI Upper")
# Add a transparent rectangle for the confidence interval region
fig.add_shape(type="rect", x0=ci_lower, y0=0, x1=ci_upper, y1=ci_y,
    fillcolor="LightSkyBlue", opacity=0.5, line_width=0)
# Add annotations for the confidence interval lines
fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[ci_y+0.01, ci_y+0.01],  
              text=["95% CI Lower", "95% CI Upper"], mode="text", showlegend=False))

fig.update_layout(
    title="Bootstrapped Sampling Distribution with 95% Confidence Interval",
    xaxis_title="Mean Health Score Change", yaxis_title="Density")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

## Fast Review of Last Class



```python
# load / reset df
df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/2e9bd5a67e09b14d01f616b00f7f7e0931515d24/data/2020/2020-07-07/coffee_ratings.csv")
df = df.rename(columns={'country_of_origin': 'origin', 'total_cup_points': 'points'})

df = df[df['points']>65] # ignore some very low scores
df = df[~df['origin'].isna()] # remove rows with unknown origin

df['origin'] = df['origin'].str.replace("?","'") # fix character encoding issue
df['origin_original'] = df.origin.copy().values # save original (corrected) names

df.loc[df.origin=='Guatemala', 'points'] = \
    -df.loc[df.origin=='Guatemala', 'points'] 
df = df.loc[df.origin=='Guatemala']
df = df[1:].copy()

# shift it to be positive
df.loc[df.origin=='Guatemala', 'points'] += 100

fig = px.histogram(df[df.origin == 'Guatemala'], x='points', 
                   labels={'points': "Statistician's transformation to allow Statistical Modelling<br>(we can always translate our results back to the original scale)"}, 
                   title='Histogram of "Transformed Points" for Guatemala')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

**Questions**

1. What does the code below do? 
    1. What are the formal distribution **parameters**?
    2. What are some other characteristics of the population<br> (that we might call "'unnamed' parameters")? 
2. Regarding **inference** versus **estimation** what is the role of the **sample size** $n$
    1. For the **variance**
    2. For the **standard error**?
    3. For the eventual **bootstrapped confidence interval**?     


```python
a, loc, scale = \
stats.gamma.fit(df.loc[df.origin=='Guatemala', 'points'].values)
```


```python
support = np.linspace(df.loc[df.origin == 'Guatemala', 'points'].min(), 
                       df.loc[df.origin == 'Guatemala', 'points'].max(), 100)
gamma_pdf = stats.gamma.pdf(support, a, loc=loc, scale=scale)

# Create a line plot of the Gamma distribution 
# PDF (probability density function) model of the population 
fig = go.Figure()
fig.add_trace(go.Scatter(x=support, y=gamma_pdf, mode='lines', 
                         name='Gamma PDF', line=dict(color='blue')))

pop_mean = a*scale+loc  # gamma_pdf.mean() wrong when loc is used
pop_std_dev = np.sqrt(a)*scale  # gamma_pdf.std() wrong when loc is used

fig.add_vline(x=pop_mean, line_dash="dash", line_color="green", 
              annotation_text=f"Mean: {mean:.2f}", annotation_position="top right")

fig.add_vline(x=pop_mean - pop_std_dev, line_dash="dash", line_color="red", 
              annotation_text=f"-1 SD: {pop_mean - pop_std_dev:.2f}", annotation_position="bottom left")

fig.add_vline(x=pop_mean + pop_std_dev, line_dash="dash", line_color="red", 
              annotation_text=f"+1 SD: {pop_mean + pop_std_dev:.2f}", annotation_position="bottom right")

fig.update_layout(title="Gamma Distribution Fit for Points (Guatemala)", 
                  xaxis_title="Points", yaxis_title="Density")
fig.show()
```


```python
from scipy import stats

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

**Questions**

1. What is **variance**?
2. What is **standard error**?
3. How do we get a **bootstrapped confidence interval** for the **sample mean**?
4. What's the difference between the analysis above and the analysis below?


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

## "Proving" bootstrapped confidence intervals


```python
simulations = 1000  
bootstrapped_ci_caputred = np.zeros(simulations)
bootstrap_samples = 500
xbar_bootstrapped = np.zeros(bootstrap_samples)

n_ = n  # 10
for j in range(simulations):
    simulated_sample = stats.gamma(a=a, loc=loc, scale=scale).rvs(size=n_)
    simulated_sample = pd.DataFrame({'points': simulated_sample})

    for i in range(bootstrap_samples):
        bootstrapped_sample = simulated_sample.sample(n=n_, replace=True).points
        xbar_bootstrapped[i] = bootstrapped_sample.mean()
        
    bootstrapped_ci_caputred[j] =\
        1 == (pop_mean <= np.quantile(xbar_bootstrapped, [0.025, 0.975])).sum()
    
print(bootstrapped_ci_caputred.sum()/simulations)    
```

**Questions** 

1. What's the difference between "number of simulated samples" VS "number of bootstraped samples"?
2. Did we "prove" boostrapping works as promised?
3. Did we demonstate how to correctly intepret **confidence intervals**?

## A reminder/review of how to use ChatBots

1. Don't do [this](https://chatgpt.com/share/66fd24f3-3a00-8007-a67b-e389bb4bb940)
2. This is [better](https://chatgpt.com/share/66fd2ea8-0dd8-8007-bb44-41e63de657be)
3. This is about the same, but also actually probably better


```python
from IPython.display import YouTubeVideo
YouTubeVideo('IM4wwDFUmXE', width=800, height=500)
```


```python
YouTubeVideo('P8OgTEmJTeU', width=800, height=500)
```

## Sampling Distribution of a statistic <br>*under the null hypothesis*



```python
# We're going to ask a slightly different question

patient_data['Improved'] = patient_data['HealthScoreChange']>0
patient_data
```

$$\begin{align*}
H_0 &{}: \mu_{\textrm{vaccine}}=\mu_{\textrm{no vaccine}}\\
&{} \;\;\;\textrm{There is no difference }\textbf{on average}\textrm{ do to the vaccine} \textbf{ is 50%}\\\\
H_0 &{}: \textrm{The vaccine has no effect }\textbf{(on average)}\textrm{ on patient health}\\
H_0 &{}: \textrm{The chance the vaccine improves patient health} \textbf{ is 50%}\\\\
H_0  &{}: p=0.5 \quad \textrm{instead of the equivalent} \\
&{} \quad\quad\quad\quad\;\; H_0: \textrm{The chance the vaccine improves patient health} \textbf{ is 50%}
\end{align*}$$


```python
population_parameter_value_under_H0 = 0.5

observed_statistic = (patient_data.HealthScoreChange>0).mean()

print('The p=0.5 Null Hypothesis of "no effect" on average')
print('but an observed statistic of', observed_statistic, "of the sample show increased health")
```


```python
np.random.seed(1)  # make simulation reproducible
number_of_simulations = 10000  # experiment with this... what does this do?
n_size = len(patient_data)  # 10
IncreaseProportionSimulations_underH0random = np.zeros(number_of_simulations)

# generate "random improvement" proportions assuming H0 (vaccine has no average effect) is true 
# meaning that the "before and after" differences are positive or negative at "random"
for i in range(number_of_simulations):
    
    # why is this equivalent to the suggested idea above?
    random_improvement = np.random.choice([0,1], size=n_size, replace=True)  # <<< `replace=True` ^^^

    # why is .mean() a proportion? 
    IncreaseProportionSimulations_underH0random[i] = random_improvement.mean()
    # why is this the statistic we're interested in? Hint: next section...
```

### A p-value is the the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true


```python
# "as or more extreme" relative to the hypothesized parameter of the statistic!
simulated_statistics = IncreaseProportionSimulations_underH0random

SimulatedStats_as_or_more_extreme_than_ObservedStat = \
    abs(simulated_statistics - population_parameter_value_under_H0) >= \
    abs(observed_statistic - population_parameter_value_under_H0) 
    
print('''Which simulated statistics are "as or more extreme"
than the observed statistic? (of ''', observed_statistic, ')', sep="")

# figure for demonstration only: code details not of primary concern

hist_data = [IncreaseProportionSimulations_underH0random+np.random.uniform(-0.05,0.05,size=len(IncreaseProportionSimulations_underH0random))]
group_labels = ['Bootstrap<br>Sampling<br>Distribution<br>of the<br>Sample<br>Mean<br><br>assuming<br>that the<br>H0 null<br>hypothesis<br>IS TRUE']
fig = ff.create_distplot(hist_data, group_labels, curve_type='normal',
                         show_hist=True, show_rug=False, bin_size=0.1)
pv_y = 2.5
pv_y_ = .25
fig.add_shape(type="line", x0=observed_statistic, y0=0, 
              x1=observed_statistic, y1=pv_y,
              line=dict(color="Green", width=4), name="Observed Statistic")
fig.add_trace(go.Scatter(x=[observed_statistic], y=[pv_y+pv_y_], 
                         text=["Observed<br>Statistic<br>^"], mode="text", showlegend=False))
# "as or more extreme" also include the "symmetric" observed statistic...
symmetric_statistic = population_parameter_value_under_H0 -\
                      abs(observed_statistic-population_parameter_value_under_H0)
fig.add_shape(type="line", x0=symmetric_statistic, y0=0, 
              x1=symmetric_statistic, y1=pv_y,
              line=dict(color="Green", width=4), name="Observed Statistic")
fig.add_trace(go.Scatter(x=[symmetric_statistic], y=[pv_y+pv_y_], 
                         text=['"Symmetric" Observed Statistic<br>addrdssing for "as or more extreme"<br>^'], mode="text", showlegend=False))

# Add a transparent rectangle for the lower extreme region
fig.add_shape(type="rect", x0=-0.25, y0=0, x1=symmetric_statistic, y1=pv_y,
              fillcolor="LightCoral", opacity=0.5, line_width=0)
# Add a transparent rectangle for the upper extreme region
fig.add_shape(type="rect", x0=observed_statistic, y0=0, x1=1.25, y1=pv_y,
              fillcolor="LightCoral", opacity=0.5, line_width=0)

# Update layout
fig.update_layout(
    title="Bootstrapped Sampling Distribution<br>under H0 with p-value regions",
    xaxis_title="Mean Health Score Change", yaxis_title="Density", yaxis=dict(range=[0, pv_y+2*pv_y_]))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### Synthetics! 

![](https://www.scified.com/topics/1804189048037466.png)

## VS the<br>_bootstrapped Sampling Distribution_ of a statistic

|![](https://uselessetymology.com/wp-content/uploads/2019/10/bootstraps.png)|![](https://img.huffingtonpost.com/asset/5b6b3f1f2000002d00349e9d.jpeg?cache=92VfjlAeaf&ops=1200_630)|
|-|-|
| | |


```python
hist_data = [bootstrap_means]
group_labels = ['Bootstrapped<br>Sampling Distribution<br>of the Sample Mean']
fig = ff.create_distplot(hist_data, group_labels, 
                         show_hist=True, show_rug=False, bin_size=0.4)

# Add a line for the lower confidence interval
ci_y = 0.35  # Adjust height as needed
fig.add_shape(type="line", x0=ci_lower, y0=0, x1=ci_lower, y1=ci_y,
              line=dict(color="Red", width=2), name="95% CI Lower")
# Add a line for the upper confidence interval
fig.add_shape(type="line", x0=ci_upper, y0=0, x1=ci_upper, y1=ci_y,
              line=dict(color="Red", width=2), name="95% CI Upper")
# Add a transparent rectangle for the confidence interval region
fig.add_shape(type="rect", x0=ci_lower, y0=0, x1=ci_upper, y1=ci_y,
    fillcolor="LightSkyBlue", opacity=0.5, line_width=0)
# Add annotations for the confidence interval lines
fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[ci_y+0.01, ci_y+0.01],  
              text=["95% CI Lower", "95% CI Upper"], mode="text", showlegend=False))

fig.update_layout(
    title="Bootstrapped Sampling Distribution with 95% Confidence Interval",
    xaxis_title="Mean Health Score Change", yaxis_title="Density")
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

|![](https://cdn.dribbble.com/users/1064236/screenshots/5753511/redbull_fly_800x600.gif)|![](https://media1.giphy.com/media/VeGYtq4kReVJmt5XVM/giphy.gif)|
|-|-|
| | |

## VS $\alpha$-_significance levels_ and Type I and II Errors

Check if you **p-value** is less than or equal to some pre-defined $\alpha$-**significance level**, such as $\alpha=0.5$ (which is the most commonly used version of a **statistical hypothsis test**)

- A **hypothesis test** based on an $\alpha$-**significance level**

- IS THE SAME as a **hypothesis test** with a **confidence interval** with a $(1-\alpha) \times 100$%  **confidence level**

- Hypothesis test at **significance level** $\alpha=0.5 \Longleftrightarrow 95\%$ **confidence level** decision using a $95\%$ confidence interval

You MUST choose $\alpha$ before you observe the data and calculate a statistic, otherwise the following meaning of $\alpha$ will not be true

- The $\alpha$-**significance level** is the chance your **independent** and **identically distributed** (i.i.d.) **sample** will produce a **p-value** less than $\alpha$ if the null hypothesis IS true, thereby resulting in a WRONG rejection of the null hypothesis at the $\alpha$-**significance level**

### Type I and II Errors

| Decision       | Null Hypothesis is True   | Null Hypothesis is False |
|:--------------:|:-------------------------:|:------------------------:|
| Reject Null    | **Type I Error<br>(α chance this results<br> from an i.i.d. sample)**       | Correct Decision         |
| Fail to Reject | Correct Decision           | **Type II Error <br>(β chance this results<br> from an i.i.d. sample)**|


## Don't f@#k this up<br><sub>AKA mistakes that you make that make you dumb</sub>

### Heaven

- This is a 95% confidence interval.
- I have 95% confidence this constructed interval captures the actual true population parameter value.
- I have used a confidence interval procedure which will "work" for 95% of hypothetical i.i.d. samples.
- There's a 95% chance this confidence interval "worked" and does "capture" the actual true population parameter value.

### HELL _AND_ WRATH OF SCOTT

- There's a 95% chance the parameter is in this confidence interval. 
    - **NOPE, sounds too much like we're saying parameters have "chance", but parameters don't have "a chance" of being "this or that".**
- There's a 95% probability the parameter is in this confidence interval. 
    - **NOPE, sounds too much like we're saying parameters have "probability", but parameters don't behave "probabilistically".**

### Heaven

- **A p-value is the the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true**.

- See if it's smaller than an $\alpha(=0.05?)$ and **reject the null hypothesis** at this **significance level** if so, which is interpretated the same as when you use a $(1-\alpha)\times 100$% **confidence interval**.

- Just use the p-value to characterize the **strength of evidence against the null hypothesis** based on the table below.

### HELL _AND_ WRATH OF SCOTT

- A p-value is the probability the null hypothesis is true. 
    - **OMG. NO. THIS IS NOT TRUE.**
    
- A p-value is the chance we wrongly reject the null hypothesis. 
    - **What? NO. That's the $\alpha$-significance level. Why are you confusing this with a p-value??**
    
- A p-value is the probability that the hypothesized parameter value is correct. 
    - **Omg Jesus Christ kill me. We JUST finished talking about WHY we're so careful about the way we talk about confidence intervals; because, parameters don't have "chances" or "probabilities"...**


### Just interpret p-values in terms of _Strength of Evidence_ against the null hypothesis

|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|



## Or just use _confidence intervals_

**Questions**

1. Why is a **confidence interval** better than a **hypothesis test**?

|![](https://cdn.dribbble.com/users/1064236/screenshots/5753511/redbull_fly_800x600.gif)|![](https://media1.giphy.com/media/VeGYtq4kReVJmt5XVM/giphy.gif)|
|-|-|
| | |



