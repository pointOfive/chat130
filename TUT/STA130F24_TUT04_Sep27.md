# STA130 TUT 04 (Sep27)<br><br>📊 🎯 <u>Confidence Intervals / Statistical Inference</u><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(using Sampling Distributions)

## 💬 🗣️ Communication Activity \#1 [25 Min]

Return to your *four groups* from the "Tutorial Communication Acivity" from previous week, and spend *five minutes* to decide which students analysis your group would like to present to everyone subject to the restriction that *no two groups may present the same data set*. Each group will then have *five minutes* (strictly enforced) to present their data analysis to the everyone, by either

- looking at figures in a notebook on a github repo
- or sharing the notebook with the analysis with the TA and using the TAs computer to present to the class<br>

<details class="details-example"><summary style="color:blue"><u>Instructions from the Previous TUT/HW Reminder</u></summary>

### Week 03 TUT Communication Activity
Break into 4 groups of 6 students (or as many as possible, evenly distributed across groups) and prepare a speech describing a generic strategy or general sequence of steps you would take to understand a data set

### Last Week's HW Question "6"

Go find an interesting dataset and use summary statistics and visualizations to understand and demonstate some interesting aspects of the data

1. Your approach should likely follow what was suggested for the **Week 03 TUT Communication Activity**
2. In the **Week 03 TUT Communication Activity from TUT** you will be put in groups and determine which group members dataset introduction will be presented by the group
    
</details>



## ♻️ 📚 Review / Questions [20 Min]

### 1. *[15 of the 20 minutes]* Follow up questions and clarifications regarding the content of Sep20 TUT and Sep23 LEC covering Data Visualization and Populations and Sampling

> 1. Everyone should be clear about what a **histogram** is what it describes about the data. 
> 
> 2. Do you remember that **box plots** are simpler than **histograms** and likely preferable for the purposes of comparision; except, **box plots** don't explicitly indicate sample size and modality?
>
> 3. What is the difference between a **KDE** or **violin** plot, a **population**, and a **distribution**?
>
> 4. Everyone should be clear about how to sample from a **distribution** and make a **histogram** (or **box plot** or **violin** plot).


### 2. *[5 of the 20 minutes]* Brifely remind the students of these figures from last weeks LEC (recreated below) which will be the focus of the subsequent TUT **Demo** and answer the following questions found in-line below

> For which countries do you think we can most accurately <br>estimate the average 'points' score of cups of coffee?
>
> - *Only remind students about these data and figures* (and to reinforce the **Review / Questions**)*; and, do not explain this code as students as they can later figure this out later by prompting a ChatBot with,* "Can you explain to me what is happening in this code? < pasted code >"


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


```python
# just run to get to the plot -- we are not interested in reviewing this code now

import pandas as pd
import plotly.express as px

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

# create a histogram 
fig = px.histogram(df, x='points', facet_col='origin', 
                   facet_col_wrap=6, height=1000, facet_row_spacing=0.05)

# shorten titles
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("origin=", "")))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# just run to get to the plot -- we are not interested in reviewing this code now

# include sample sizes in origin names
keys = df.origin_original.value_counts().index.values
vals = df.origin_original.value_counts().index.values + " (n="+df.origin_original.value_counts().values.astype(str)+")"
df.origin = df.origin_original.map({k:v for k,v in zip(keys,vals)})

df.origin = df.origin.str.replace("<br>", " ") # remove line breaks in origin names

# create ordered box plots with swarm                              # to turn off the data swarm
fig = px.box(df, x='points', y="origin", points="all", height=750, # remove `points="all",`
             title="EXPLICITLY adding sample size (and even the data swarm to really emphasize it...)")
fig.update_yaxes(categoryarray=df.groupby("origin")['points'].mean().sort_values().index)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# just run to get to the plot -- we are not interested in reviewing this code now

import plotly.graph_objects as go
fig = go.Figure()

# https://plotly.com/python/violin/#split-violin-plot
fig.add_trace(go.Violin(x=df.groupby('origin').filter(lambda x: len(x) > 1).origin, 
                        y=df.points,side='positive', width=5))#, points='all'
# google "pandas remove groups with size 1"
# https://stackoverflow.com/questions/54584286/pandas-groupby-then-drop-groups-below-specified-size

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

#### For which countries do you think we can most accurately <br>estimate the average 'points' score of cups of coffee?


```python
# just run to get to the plot -- we are not interested in reviewing this code now

# add line breaks to titles
df.origin = df.origin.str.replace(" (", "<br>(").replace(", ", ",<br>")

# build a histogram (but do not yet show it)
fig = px.histogram(df, x='points', facet_col='origin', 
                   facet_col_wrap=6, height=1000, facet_row_spacing=0.05,
                   title='''<br>For which countries do you think we can most accurately
                            <br>estimate the average 'points' score of cups of coffee?''')
```


```python
# just run to get to the plot -- we are not interested in reviewing this code now

# shorten titles
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("origin=", ""))) 
fig.update_layout(title_automargin=True) # fix overall title

# indicate the sampe averages in the histograms 
for i,average in enumerate(dict(df.groupby('origin').points.mean()[df.origin.unique()]).values()):
    fig.add_vline(x=average, line_dash="dot", row=6-int(i/6), col=(1+i)%6)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

#### For which countries do you think we can most accurately <br>estimate the average 'points' score of cups of coffee?

## 🚧 🏗️ Demo [30 minutes]

Estimating Averages?

- You'll want to know the behavior of the variability of averages for the sample size...

### 1. For which countries do you think we can most accurately estimate the average "points" score of cups of coffee?
    
- **That is, the average of ALL cups of coffee from a given country... not just those in the sample**
    - The sample means have been added to the histograms above
    - But the answer to this question is really answered by the question...

### 2. How does the variability/uncertainty of means of simulated samples change as a function of sample size?

> The code below demonstrates repeatedly drawing samples in order to simulate a lot of means to see what their variability/uncertainty is

**Population (distribution) $\rightarrow$ (independent) Sample $\rightarrow$ Mean (statistic)**


```python
from scipy import stats
import numpy as np

# Population
population_parameter_mu_μ = 0
population_parameter_sigma_σ = 1
normal_distribution = stats.norm(loc=population_parameter_mu_μ, 
                                 scale=population_parameter_sigma_σ) 
# Sample
n = 100 # adjust and experiment with this
# np.random.seed(130)
x = normal_distribution.rvs(size=n) # "x" is the sample of size "n"
# print(x) # uncomment this if you also want to see the sample 

# Mean
print("The sample mean for the current sample is", x.mean()) 
# the sample mean "x-bar" is a (sample) "statistic" (not a "parameter")
# "x-bar" is the "average" of the numbers in a sample
```


```python
# This code visualizes the population being sampled from (as a red line),
# a sample from this population (as a histogram), and the sample mean

# create another sample
n = 100 # adjust and experiment with this
# np.random.seed(130)
x = normal_distribution.rvs(size=n) # "x" is the sample of size "n"

# create a histogram of the sample and annotate the sample mean
fig = px.histogram(pd.DataFrame({'sampled values': x}), x='sampled values',
                   histnorm='probability density') # so the scale matches the pdf below
fig.add_vline(x=x.mean(), line_dash="dot", annotation_text='Sample mean '+str(x.mean()))

# add population visualization into the figure
support = np.linspace(-4,4,100) 
fig.add_trace(go.Scatter(mode='lines', name='Poulation Model<br>(normal distribution)',
                         y=normal_distribution.pdf(support), x=support))
                                             # pdf means "probability density function"
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

#### Repeated Sampling $\rightarrow$ Distribution of Sample Means 

- Clarification Question: What's the difference between `number_of_simulations` and the sample size `n`?


```python
# What is this doing?

number_of_simulations = 1000 # adjust and experiment with this
simulated_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
n = 100 # adjust and experiment with this
for i in range(number_of_simulations):
    # np.random.seed(130) # ?
    simulated_means[i] = stats.norm(loc=0, scale=1).rvs(size=n).mean()
```


```python
# This time the code creates a histogram of example simulations of sample means
# and visualizes the theoretical distribution of the sample means (as a red line)
    
title1 = "The sampling distribution of the sample mean"
title2 = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n)
fig = px.histogram(pd.DataFrame({title2: simulated_means}), title=title1, x=title2, 
                   histnorm='probability density')    

# add a visualization of "the sampling distribution of the sample mean" into the figure
support = 4*np.array([-1,1])/np.sqrt(n) # the support code here automatically chooses
support = np.linspace(support[0], support[1], 100) # the plotting range for the x-axis
legend = 'The "sampling distribution<br>of the sample mean" is the<br>theoretical distribution of<br>"averages" of sample values'
fig.add_trace(go.Scatter(mode='lines', name=legend, x=support, 
                         y=stats.norm(0,scale=1/np.sqrt(n)).pdf(support)))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Now let's consider sample sizes from the data above...
df.origin.unique()
```


```python
# And in order to repeatedly simulate samples (means) like the ones we have, let's
# FALSELY pretend the population parameters are their corresponding statistics
country = 'Mexico<br>(n=236)'

PRETEND_population_parameter_mu_μ_is_sample_mean = df[df.origin==country].points.mean()
PRETEND_population_parameter_sigma_σ_is_sample_std = df[df.origin==country].points.std()

n = (df.origin==country).sum()

print("PRETENDING that the population mean μ is", PRETEND_population_parameter_mu_μ_is_sample_mean)
print("and the population standard deviation σ is", PRETEND_population_parameter_sigma_σ_is_sample_std)
print("we can repeatedly draw samples of size", n, "and compute their means...")
```


```python
# What is this doing?

number_of_simulations = 1000 # adjust and experiment with this
simulated_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
for i in range(number_of_simulations):
    simulated_means[i] = stats.norm(loc=PRETEND_population_parameter_mu_μ_is_sample_mean,
                                    scale=PRETEND_population_parameter_sigma_σ_is_sample_std).rvs(size=n).mean()
```


```python
title1 = "The sampling distribution of the sample mean"
title2 = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n)
title = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n)
fig = px.histogram(pd.DataFrame({title2: simulated_means}), title=title1, x=title2,
                   histnorm='probability density')    

# support is more complicated this time around, but it is still the same 
# automatic calculation to determine the plotting range for the x-axis
support = PRETEND_population_parameter_mu_μ_is_sample_mean + \
          4*np.array([-1,1])*PRETEND_population_parameter_sigma_σ_is_sample_std/np.sqrt(n)
support = np.linspace(support[0], support[1], 100)
legend = 'The "sampling distribution<br>of the sample mean" is the<br>theoretical distribution of<br>"averages" of sample values'
fig.add_trace(go.Scatter(mode='lines', name=legend, x=support, 
                         y=stats.norm(loc=df[df.origin==country].points.mean(),
                                      scale=df[df.origin==country].points.std()/np.sqrt(n)).pdf(support)))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

***Reminder: the questions were...***

1. For which countries do you think we can most accurately estimate the average "points" score of cups of coffee?<br>

2. How does the variability of means of simulated samples change as a function of sample size?<br>

*And what really matters here is the x-axis above...*


## 📢 👂 Communication Actvity \#2 [25 minutes] 

> Break into *five groups* of students, assigning each group to one of the questions below. Groups discuss questions for *5-10 minutes*, and then each group (in order) provides their answer to the class for *3 minutes each*.
>
> - The last two questions are "challenge questions" for adventurous groups who are interested in attempting to push the envelope and extend what they've learned so far into two new versions of this **repeated sampling** exercise.
>
> - *Because the last two questions are "challenge questions" the TA may determine to focus more attention on the first three questions depending on the progress of the TUT in understanding and presenting the key ideas of these questions.*

1. **Question 2** (the last 5 of the 20 minutes) of the **Review / Questions** section of TUT (following the initial 25 minute **Communication Activity #1**) re-introduced the panel of histograms of coffee ratings first seen in the last LEC. The samples were then compared side-by-side using **box plots** and **violin**/**KDE** plots. This is NOT what the **TUT Demo** is doing. Compare and contrast the purpose of the exercise of the **TUT Demo** verses qualitatively comparing samples using box plots.


2. In the **TUT Demo** we used two different normal distributions. Compare and contrast the two different examples and explain the purpose of having two different examples.


3. How does the variability/uncertainty of means of simulated samples change as a function of sample size? Can you explain why this is in an intuitive way? Feel free to use a ChatBot (or internet or course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) search) to help answer this question if you like (and share what you found to be effective in doing so!). *Remember to focus a ChatBot on smaller subsets of your bigger questions to build up to the full understanding systematically if needed! Following up overly verbose initial chat responses by asking for a shorter and more succinct or more specific responses will likely lead to good results.*


4. In the **TUT Demo** we sampled from a symmetric `normal` population, but this is replaced with a skewed `gamma` population in the first set of code below. Run the code related to the skewed `gamma` population with different sample sizes and discuss how the behavior  exhibited by the two (symmetric and skewed) examples is similar and/or different. Feel free to use a ChatBot (or internet or course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) search) to help understand what the code is doing if you need to (and share what you found to be effective in doing so, and in understanding the behavior at different sample sizes!). *Remember to focus a ChatBot on smaller subsets of your bigger questions to build up to the full understanding systematically if needed! Following up overly verbose initial chat responses by asking for a shorter and more succinct or more specific responses will likely lead to good results.*


5. In the **TUT Demo** we sampled from a symmetric `normal` population, but the final set of code below introduces creating samples using **bootstrapping**. Explain what the process of bootstrapping given below is, contrasting it with the illustrations of the **TUT Demo** sampling from `normal` (or `gamma`) populations. Feel free to use a ChatBot (or internet or course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) search) to help understand what the code is doing if you need to and help answer this question (and share what you found to be effective in doing so!). *Remember to focus a ChatBot on smaller subsets of your bigger questions to build up to the full understanding systematically if needed! Following up overly verbose initial chat responses by asking for a shorter and more succinct or more specific responses will likely lead to good results.*


<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
- Sharing the relevant code content with the ChatBot is probably necessary in order to provide the necessary context to allow the ChatBot to appropriately address your inquiry<br><br>
    
- The standard error (SE) of the mean (SEM) and confidence intervals may be introduced by the response of the ChatBot (e.g., like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk4/COP/00003_creative_contrastbootstrappingV2.md)), but they are beyond the scope of this question [and will be addressed in the HW and in LEC] and should not be a part of your answer: we are interested on specifically in understanding what the process of bootstrapping is 

</details>

## Potential "Challenge Questions" Material for Two Advanced Groups...

The code introduced above demonstrated repeatedly drawing samples in order to simulate a lot of means to see what their variability/uncertainty was. The code below does it again in but in two different ways.

1. The code above was based on (symmetric) `normal` distributions
2. The first version of the code below is based on (skewed) `gamma` distributions 
3. The final version of the code uses (other empirical shapes) via **bootstrapping**

> **If you need help understanding ANY of the code used in this notebook, a ChatBot is going to be able to do a very good job explaining it to you.**

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> These examples explore the variability/uncertainty of sample means when sampling from populations that are symmetric (`normal`), skewed (`gamma`), or having other empirical shapes (**bootstrapping**)
>
> - A more advanced treatment of this topic would address the *Central Limit Theorem (CLT)* or the *Law of Large Numbers (LLN)*, but these are beyond the scope of STA130; *instead, we are curious about the role of the sample size in this context, and whether or not the nature of the population (symmetric versus skewed versus using bootstrapping) affects the general behaviour we observe*
>
> *The point of these examples is to understand the simulation the code allows you to experiment with. Understanding all aspects of the code here more fully can be done at a later time (e.g., with a ChatBot interactions like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk4/GPT/SLS/00001_gpt3p5_LawOfLargeNumbers_demo.md))*.

</details>

## (skewed) `gamma` distributions

**If we sample from a `gamma` population rather than a `normal` population, does skewness seem to affect the behavior we're seeing?**


```python
# population
pop_parameter_alpha_α = 2
pop_parameter_theta_θ = 4
gamma_distribution = stats.gamma(a=pop_parameter_alpha_α, scale=pop_parameter_theta_θ)

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
n = 100 # adjust and experiment with this
# np.random.seed(130)
x = gamma_distribution.rvs(size=n) 

fig = px.histogram(pd.DataFrame({'sampled values': x}), x='sampled values',
                   histnorm='probability density') # so the scale matches the pdf below
fig.add_vline(x=x.mean(), line_dash="dot", annotation_text='Sample mean '+str(x.mean()))

support = np.linspace(0,50,100)
fig.add_trace(go.Scatter(x=support, y=gamma_distribution.pdf(support), 
                         mode='lines', name='Poulation Model<br>(gamma distribution)'))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```


```python
# what is this doing?

number_of_simulations = 1000 # adjust and experiment with this
simulated_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
n = 100 # adjust and experiment with this

for i in range(number_of_simulations):
    # np.random.seed(130) # ?
    simulated_means[i] = gamma_distribution.rvs(size=n).mean()
```


```python
hint = '<br>At at small sample sizes, the "shape" of the means reflext the population skew'
hint += '<br>showing that the "theoretical" expectation is an approximation that can be "off"'
title1 = "The sampling distribution of the sample mean..."+hint
title2 = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n)
fig = px.histogram(pd.DataFrame({title2: simulated_means}), title=title1, x=title2,
                   histnorm='probability density')    

support = gamma_distribution.mean() + 4*np.array([-1,1])*gamma_distribution.std()/np.sqrt(n)
support = np.linspace(support[0], support[1], 100)
legend = 'The "sampling distribution<br>of the sample mean" is the<br>theoretical distribution of<br>"averages" of sample values'
fig.add_trace(go.Scatter(mode='lines', name=legend, x=support, 
                         y=stats.norm(loc=gamma_distribution.mean(),
                                      scale=gamma_distribution.std()/np.sqrt(n)).pdf(support)))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

## (other empirical shapes) when using *bootstrapping*

**Does the behavior we're seeing persist if we "*pretend the sample is the population*"?**


**Bootstrapping: pretending a sample is the population while keep the same sample size**

When bootstrapping...
1. Why `replace=False`?
2. Why is `n_` the same as the original sample size `n`?

*Hint: why are we interested in understanding the variability of sample means at the sample size of the original sample...*



```python
keep = (df.origin_original=='Guatemala') | (df.origin_original=='Mexico')
fig = px.histogram(df[keep], x='points', facet_col='origin', facet_col_wrap=2, height=300)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```


```python
country = 'Mexico<br>(n=236)' 

# bootstrapping is when `replace=True` and `n` is the original sample size
# and we do this over and over to see the behavior of sample statistics
n_ = (df.origin==country).sum() # This is the correct choice for bootstrapping, but why?
replace_ = True # This is the correct choice for bootstrapping, but why?

# We don't have a "poulation" any more... only the (original) "Sample"
x = df[df.origin==country].sample(n=n_, replace=replace_).points
# print(x)
print("The bootstrap sample mean for the current bootstrapped sample is", x.mean()) 
```


```python
# what is this doing?

# np.random.seed(130) # ?
x = df[df.origin==country].sample(n=n_, replace=replace_).points

dat = pd.DataFrame({'sample values': np.r_[df[df.origin==country].points.values, x],
                    'sample': np.r_[['Orginal Sample']*(df.origin==country).sum(),
                                    ['Bootstrap Sample']*n_]})             

fig = px.histogram(dat, x="sample values", color="sample", barmode="overlay")
fig.add_vline(x=x.mean(), line_dash="dot", annotation_text='Bootstrapped sample mean<br>'+str(x.mean()))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

**If "n" is not the original sample size this is NOT BOOTSTRAPPING... <br>in this case it's ONLY pretending the sample is the population...**
- If you don't set "n" to be the original sample size, you can still use this to demonstrate sampling from a population that has the "shape" as the sample


```python
# what is this doing?

number_of_simulations = 1000 # adjust and experiment with this
simulated_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
n_ = 100 # adjust and experiment with this # 236 # 
# for bootstrapping it should be (df.origin==country).sum()
# if it's not, it's not bootstrapping... it's something else...
# ...not sure what it would then be called... 

replace_ = True # ? ... what would happen if the bootstrap sample size n_ 
# was the same size as the origina sample size n, and this was `False`?

for i in range(number_of_simulations):
    simulated_means[i] = df[df.origin==country].sample(n=n_, replace=replace_).points.mean()
```


```python
title1 = "The BOOTSTRAPPED sampling distribution of the sample mean"
title2 = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n_)
fig = px.histogram(pd.DataFrame({title2: simulated_means}), title=title1, x=title2,
                   histnorm='probability density')    

support = PRETEND_population_parameter_mu_μ_is_sample_mean + \
          4*np.array([-1,1])*PRETEND_population_parameter_sigma_σ_is_sample_std/np.sqrt(n)
support = np.linspace(support[0], support[1], 100)
legend = 'The "sampling distribution<br>of the sample mean" is the<br>theoretical distribution of<br>"averages" of sample values'
fig.add_trace(go.Scatter(mode='lines', name=legend, x=support, 
                         y=stats.norm(loc=df[df.origin==country].points.mean(),
                                      scale=df[df.origin==country].points.std()/np.sqrt(n_)).pdf(support)))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

**The above demonstrates that the behavior we see from bootstrapping (pretending the sample is the population) works just as if sampling from an actual population**
