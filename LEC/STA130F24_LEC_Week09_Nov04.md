# STA130 LEC Week 09 (Nov 04)

## Didn't we already do Group Comparison? NO! Only a little.

1. **WELCOME BACK TO EARTH REVIEWQ**
    1. It's time to ~~GET TO WORK~~ PLAY ANOTHER GAME
    2. "When you're doing something fun it's not work" -- some wiseass oldass person said this, I'm sure
    3. Scientific evidence suggests that "challenge" moments where you really get tested and have to figure out if you ACTUALLY know something are EXTREMELY POWERFUL AND FORMATIVE (and no I can't be bothered to actually gather and present this evidence for you in a organized distillled this is a "trust me I'm an authority" type situation and just because that's a logical fallocy in an argument doesn't mean it's not true)  
    
    
2. **THINK-PAIR-SHARE**<br>*Sample A has 90 data points. Sample B has 110 data points. No individuals are the same across samples.*
    1. What hypothesis testing question might we answer here? 
    2. What statistics would you use for these?
    3. How would that change for different types of data? 
        1. Continuous?
        2. Binary??
        3. Categorical??? 
    4. Assuming you probably figured out a difference statistic you should use, what will you do? **I don't think you know**
    
    
3. Doing statistics for two (unpaired) samples

    1. **Permutation test**
    2. **Two-sample bootstrapped confidence interval**
    3. **Indicator variable contrasts**
    4. And what's the difference between ALL of these methods?


4. **Self Evaluation: what's the correlation of your understand versus the true of the following items?<br>AKA what's your 0%-100% (or, techically -100%-100%) understanding level for the following topics?**
    1. Bootstrapped Confidence Intervals
    2. "Coin Flippling" sampling distribution hypothesis testing for "paired samples"
    3. Calculating p-values based on observed statistics and "sampling distributions under the null"
    4. Correlation
    5. The normal "Simple Linear Regression" model
    6. Fitting Simple Linear Regression models
    7. Making predictions from linear models
    8. Using Simple Linear Regression to evaluate the evidence of association between two continue variables
    9. Assessming the assumptions of Simple Linear Regression using residuals
    10. Hypothesis testing for two unpaired samples using a permutation test (as opposed to hypothesis testing based on differences for "paired samples")
    11. Hypothesis testing for two groups (unpaired samples) using indicator variables in Simple Linear Regression
    12. "Double" bootstrap confidence intervals estimating difference parameters for two groups (unpaired samples)


5. **Student Lecture Summary**




```python
import pandas as pd

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# fail https://github.com/KeithGalli/pandas/blob/master/pokemon_data.csv
pokeaman = pd.read_csv(url)
pokeaman
```


```python
pokeaman.describe()
```

### 2. THINK-PAIR-SHARE "answers"

1. **Continuous?**<br><br>

    1. Paired sample? 
    
    2. *Just One Sample.* How will you test this? There are two ways you might be able to think to do this. 
    
       $H_0: \mu_{\textrm{attack}}=100$   
       Are there any other ideas? $$ $$
       <!-- $$H_0: p_{\textrm{attack}\geq100}=0.5$$ -->

    3. **UNPAIRED sample?**


2. **Binary??**<br><br>

    1. ONE sample

    $H_0: p_{\textrm{legendary}}=0.5$ or maybe instead $H_0: p_{\textrm{legendary}}=0.01$ $$ $$

    2. Paired sample? $$ $$
    
    3. **UNPAIRED sample?** 
    

3. **Categorical???**




#### 1. Continuous?

A. Paired sample?

$H_0: \mu_{\textrm{attack}}=\mu_{\textrm{defense}}$ 

$H_0: p_{\textrm{attack} \geq \textrm{defense}} = 0.5$ 



```python
import numpy as np 

# Set parameters for bootstrap
n_bootstraps = 1000  # Number of bootstrap samples
sample_size = len(pokeaman)  # Sample size matches the original dataset
bootstrap_means = np.zeros(n_bootstraps)

for i in range(n_bootstraps):
    bootstrap_means[i] = (pokeaman["Attack"]-pokeaman["Defense"]).sample(n=sample_size, replace=True).mean()

np.quantile(bootstrap_means, [0.05, 0.95])
```


```python
simulated_proportions = bootstrap_means.copy()
for i in range(n_bootstraps):
    simulated_proportions[i] = (np.random.choice([0,1], p=[0.5,0.5], replace=True, size=sample_size)).mean()

fig = px.histogram(pd.DataFrame({"simulated_proportions": simulated_proportions}), nbins=30,
                                title="50/50 'Coin Flip' Sampling Distribution for Attack>=100")

attack_biggerthan_defense = (pokeaman["Attack"]>=pokeaman["Defense"]).mean()
fig.add_vline(x=attack_biggerthan_defense, line_dash="dash", line_color="red",
              annotation_text=f"Proportion >= 100: {attack_biggerthan_defense:.2f}",
              annotation_position="top right")
fig.add_vline(x=0.5-(attack_biggerthan_defense-0.5), line_dash="dash", line_color="red",
              annotation_text=f"Proportion <= 100: {0.5-(attack_biggerthan_defense-0.5):.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(simulated_proportions-0.5) >= abs(attack_biggerthan_defense-0.5)).sum()/n_bootstraps)
```


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```

#### 1. Continuous?

B. *Just One Sample.* How will you test this? There are two ways you might be able to think to do this. 
    
$H_0: \mu_{\textrm{attack}}=100$   
Are there any other ideas? $$ $$
<!-- $$H_0: p_{\textrm{attack}\geq100}=0.5$$ -->



```python
for i in range(n_bootstraps):
    bootstrap_means[i] = pokeaman["Attack"].sample(n=sample_size, replace=True).mean()

np.quantile(bootstrap_means, [0.05, 0.95])
```


```python
bootstrap_proportions = bootstrap_means.copy()
for i in range(n_bootstraps):
    bootstrap_proportions[i] = (pokeaman["Attack"].sample(n=sample_size, replace=True)>=100).mean()

np.quantile(bootstrap_proportions, [0.05, 0.95])
```


```python
for i in range(n_bootstraps):
    simulated_proportions[i] = (np.random.choice([0,1], p=[0.5,0.5], replace=True, size=sample_size)).mean()

fig = px.histogram(pd.DataFrame({"simulated_proportions": simulated_proportions}), nbins=30,
                                title="50/50 'Coin Flip' Sampling Distribution for Attack>=100")

cutoff=100#75
attack_biggerthan_proportion = (pokeaman["Attack"]>=cutoff).mean()
fig.add_vline(x=attack_biggerthan_proportion, line_dash="dash", line_color="red",
              annotation_text=f"Proportion >= 100: {attack_100plus_proportion:.2f}",
              annotation_position="top right")
fig.add_vline(x=0.5-(attack_biggerthan_proportion-0.5), line_dash="dash", line_color="red",
              annotation_text=f"Proportion <= 100: {0.5-(attack_biggerthan_proportion-0.5):.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(simulated_proportions-0.5) >= abs(attack_biggerthan_proportion-0.5)).sum()/n_bootstraps)
```


```python
# lemme show you one (no, two) more way(s) you probably haven't thought of for doing this...

from scipy import stats
import plotly.graph_objects as go

pokeaman["Attack (100 Average)"] = pokeaman["Attack"] - int(pokeaman["Attack"].mean()) + 100

fig = px.histogram(pokeaman.melt(value_vars=["Attack", "Attack (100 Average)"], 
                                 var_name="Type", value_name="Attack Value"),
                   x="Attack Value", color="Type", facet_col="Type", nbins=30, 
                   title="Distribution of Pokémon Attack, and instead if the average Attack was 100")

x_values = np.linspace(pokeaman["Attack (100 Average)"].min(), pokeaman["Attack (100 Average)"].max(), 100)
y_values = stats.norm(loc=100, scale=pokeaman["Attack"].std()).pdf(x_values)

# Overlay the normal distribution on the right panel
fig.add_trace(go.Scatter(x=x_values, y=y_values*8000,  # Scale by bin width and sample size
        mode="lines", name="Normal Distribution<br>Approximation"),
    row=1, col=2)


fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import plotly.express as px

simulated_means = simulated_proportions.copy()
for i in range(n_bootstraps):
    simulated_means[i] = pokeaman["Attack (100 Average)"].sample(n=sample_size, replace=True).mean()
    #simulated_means[i] = stats.norm(loc=100, scale=pokeaman["Attack (100 Average)"].std()).rvs(size=sample_size).mean()
    
    
fig = px.histogram(pd.DataFrame({"simulated_means": simulated_means}), nbins=30,
                                title="Sampling Distribution Attack if Average Attack is 100")

fig.add_vline(x=pokeaman["Attack"].mean(), line_dash="dash", line_color="red",
              annotation_text=f"Attack mean: {pokeaman['Attack'].mean():.2f}",
              annotation_position="top right")
fig.add_vline(x=100-(pokeaman["Attack"].mean()-100), line_dash="dash", line_color="red",
              annotation_text=f"Attach mean: {100-(pokeaman['Attack'].mean()-100):.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

#print("p-value",
#      (abs(simulated_proportions-0.5) >= abs(attack_biggerthan_proportion-0.5)).sum()/n_bootstraps)
```

#### 2. Binary??

A. ONE sample?



```python
import plotly.express as px

simulated_proportions = bootstrap_means.copy()
for i in range(n_bootstraps):
    simulated_proportions[i] = (np.random.choice([0,1], p=[0.99,0.01], replace=True, size=sample_size)).mean()

fig = px.histogram(pd.DataFrame({"simulated_proportions": simulated_proportions}), nbins=30,
                                title="50/50 'Coin Flip' Sampling Distribution for Attack>=100")

legendary_proportion = (pokeaman["Legendary"]).mean()
fig.add_vline(x=legendary_proportion, line_dash="dash", line_color="red",
              annotation_text=f"Proportion: {legendary_proportion:.2f}",
              annotation_position="top right")
fig.add_vline(x=0.01-(legendary_proportion-0.01), line_dash="dash", line_color="red",
              annotation_text=f"Proportion: {0.01-(legendary_proportion-0.01):.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(simulated_proportions-0.01) >= abs(legendary_proportion-0.01)).sum()/n_bootstraps)
```

#### 2. Binary??

B. Paired sample? 

C. **UNPAIRED sample?** 



```python
pokeaman.describe()
```


```python
pokeaman
```


```python
pokeaman.fillna("None", inplace=True)
pokeaman
```

#### 1. Continuous?

C. **UNPAIRED sample?**



```python
fig = px.box(pokeaman, x="Legendary", y="Attack", 
    title="Distribution of Pokémon Attack Across Legendary: Are These Different??",
    labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print(pokeaman.groupby('Legendary')['Attack'].mean())
print(pokeaman.groupby('Legendary')['Attack'].mean().diff())
```


```python
pokeaman['Shuffled Legendary Status'] = pokeaman['Legendary'].sample(n=sample_size, replace=True).values
fig = px.box(pokeaman, x="Shuffled Legendary Status", y="Attack", 
    title="Distribution of Pokémon Attack Across Legendary: If Legendary is SHUFFLED??",
    labels={"Attack": "Attack Stat", "Shuffled Legendary Status": "Legendary Pokémon"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print(pokeaman.groupby('Shuffled Legendary Status')['Attack'].mean())
print(pokeaman.groupby('Shuffled Legendary Status')['Attack'].mean().diff())
```


```python
label_permutation_mean_differences = bootstrap_means.copy()
for i in range(n_bootstraps):
    pokeaman['Shuffled Legendary Status'] = pokeaman['Legendary'].sample(n=sample_size, replace=True).values
    label_permutation_mean_differences[i] = \
        pokeaman.groupby('Shuffled Legendary Status')['Attack'].mean().diff().values[1]
```

$\huge \text{What does it mean to shuffle? Does it mean this?}$

$\Huge H_0: \text{Legendary label doesn't matter}$

$\Large \text{Is so, what would it mean to provide evidence against this null hypothesis?}$



```python
fig = px.histogram(pd.DataFrame({"label_permutation_mean_differences": label_permutation_mean_differences}), nbins=30,
                                title="Mean Difference Sampling under Legendary labels SHUFFLED")

mean_differene_statistic = pokeaman.groupby('Legendary')['Attack'].mean().diff().values[1]

fig.add_vline(x=mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f"Shuffled Statistic <= Observed Statistic: {mean_differene_statistic:.2f}",
              annotation_position="top left")
fig.add_vline(x=-mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f"Shuffled Statistic >= Observed Statistic: {-mean_differene_statistic:.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(simulated_proportions) >= abs(attack_biggerthan_proportion)).sum()/n_bootstraps)
```

$\huge \textrm{Let's call this the "Double Bootstrap''}$

$\Huge \textrm{What's This Doing? How's does this Work?}$



```python
within_group_bootstrapped_mean_differences = bootstrap_means.copy()
for i in range(n_bootstraps):
    double_bootstrap = \
        pokeaman.groupby("Legendary")[["Legendary","Attack"]].sample(frac=1, replace=True)
    within_group_bootstrapped_mean_differences[i] = \
        double_bootstrap.groupby('Legendary')["Attack"].mean().diff().values[1]
    
np.quantile(within_group_bootstrapped_mean_differences, [0.05,0.95])    
```


```python
fg1 = px.scatter(pokeaman, x="Defense", y="Attack", title="Pokémon Attack vs. Defense",
                 labels={"Attack": "Attack Stat", "Defense": "Defense Stat"},
                 hover_name="Name", color="Legendary")
fg2 = px.density_contour(pokeaman, x="Defense", y="Attack",
                         color="Legendary",
                         title="Kernel Density Estimate of Pokémon Attack by Legendary Status")
fig = go.Figure()
for trace in fg2.data:
    fig.add_trace(trace)    
for trace in fg1.data:
    fig.add_trace(trace)    
    
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.scatter(pokeaman, x="Defense", y="Attack", title="Pokémon Attack vs. Defense",
                 labels={"Attack": "Attack Stat", "Defense": "Defense Stat"},
                 hover_name="Name", color="Type 1")#"Generation")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.scatter(pokeaman, x="Defense", y="Attack", title="Pokémon Attack vs. Defense",
                 labels={"Attack": "Attack Stat", "Defense": "Defense Stat"},
                 hover_name="Name", trendline='ols')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

$\huge \textrm{How's that DIFFERENT than this??}$



```python
fig = px.box(pokeaman, x="Legendary", y="Attack", 
    title="Distribution of Pokémon Attack Across Legendary: Are These Different??",
    labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.violin(pokeaman, x="Legendary", y="Attack", box=True, points="all", 
    title="Distribution of Pokémon Attack Across Legendary Status (Violin Plot)",
    labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})

for trace in fig.data:
    if trace.type == 'violin' and 'points' in trace:
        trace.marker.opacity = 0.5  # Set alpha transparency for points

fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.strip(pokeaman, x="Legendary", y="Attack", color="Legendary",
               title="Swarm Plot of Pokémon Attack by Legendary Status",
               labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"},
               stripmode="overlay")  # Overlay points to increase density in the plot

# Adjust jitter to spread points horizontally
fig.update_traces(jitter=0.4, marker=dict(opacity=0.6, size=6))
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
pokeaman["Legendary_int"] = pokeaman["Legendary"].astype(int)
fig = px.scatter(pokeaman, x="Legendary_int", y="Attack", trendline='ols', 
                 title="Distribution of Pokémon Attack Across Legendary: Are These Different??",
                 labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})
fig.update_xaxes(range=[-1, 2])
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

$\huge \textrm{How's that DIFFERENT than this??}$


```python
fig = px.scatter(pokeaman, x="Defense", y="Attack", title="Pokémon Attack vs. Defense",
                 labels={"Attack": "Attack Stat", "Defense": "Defense Stat"},
                 hover_name="Name", trendline='ols')
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
import statsmodels.formula.api as smf

# Model (a): Predict Attack based on Defense
model_a_fit = smf.ols(formula="Attack ~ Defense", data=pokeaman).fit()
print("Model (a): Attack ~ Defense")
print("Model (a) R**2:", model_a_fit.rsquared)
model_a_fit.summary().tables[1]
```

$\huge \widehat{\textrm{Attack}} = 45.2842 + 0.4566\times \textrm{Defense}$

$ $

$\Huge \textrm{How's that DIFFERENT than this??}$

$ $

$\huge \widehat{\textrm{Attack}} = 75.6694 + 41.0075\times 1_{\textrm{TRUE}}(\textrm{Legendary})$

$\Large \textrm{And what are the predictions and how do you make them from this model??}$


```python
# Model (b): Predict Attack based on Legendary
# Ensure Legendary is treated as a categorical variable if it’s binary or categorical
model_b_fit = smf.ols(formula="Attack ~ Legendary", data=pokeaman).fit()

# Print summary of both models
print("\nModel (b): Attack ~ Legendary")
print("Model (b) R**2:", model_b_fit.rsquared)
model_b_fit.summary().tables[1]
```


```python
pokeaman["Legendary_int"] = pokeaman["Legendary"].astype(int)
fig = px.scatter(pokeaman, x="Legendary_int", y="Attack", trendline='ols', 
                 title="Distribution of Pokémon Attack Across Legendary: Are These Different??",
                 labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})
fig.update_xaxes(range=[-1, 2])
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.box(pokeaman, x="Legendary", y="Attack", 
    title="Distribution of Pokémon Attack Across Legendary: Are These Different??",
    labels={"Attack": "Attack Stat", "Legendary": "Legendary Pokémon"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
model_b_fit.summary().tables[1]
```

$\Huge \textrm{Simple Linear Regression}$

$\huge \textrm{How's that DIFFERENT than this??}$



```python
label_permutation_mean_differences = bootstrap_means.copy()
for i in range(n_bootstraps):
    pokeaman['Shuffled Legendary Status'] = pokeaman['Legendary'].sample(n=sample_size, replace=True).values
    label_permutation_mean_differences[i] = \
        pokeaman.groupby('Shuffled Legendary Status')['Attack'].mean().diff().values[1]
    
fig = px.histogram(pd.DataFrame({"label_permutation_mean_differences": label_permutation_mean_differences}), nbins=30,
                                title="Mean Difference Sampling under Legendary labels SHUFFLED")

mean_differene_statistic = pokeaman.groupby('Legendary')['Attack'].mean().diff().values[1]

fig.add_vline(x=mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f"Shuffled Statistic <= Observed Statistic: {mean_differene_statistic:.2f}",
              annotation_position="top left")
fig.add_vline(x=-mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f"Shuffled Statistic >= Observed Statistic: {-mean_differene_statistic:.2f}",
              annotation_position="top right")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(simulated_proportions) >= abs(attack_biggerthan_proportion)).sum()/n_bootstraps)
```

$\Huge \textrm{Permutation Testing}$

$\huge \textrm{How's that DIFFERENT than this??}$



```python
within_group_bootstrapped_mean_differences = bootstrap_means.copy()
for i in range(n_bootstraps):
    double_bootstrap = \
        pokeaman.groupby("Legendary")[["Legendary","Attack"]].sample(frac=1, replace=True)
    within_group_bootstrapped_mean_differences[i] = \
        double_bootstrap.groupby('Legendary')["Attack"].mean().diff().values[1]
    
np.quantile(within_group_bootstrapped_mean_differences, [0.05,0.95])    
```

$\Huge \textrm{"Double Bootstrapping"}$


#### 2. Binary??

C. **UNPAIRED sample?** 

#### 3. Categorical???


```python
fig = px.box(pokeaman, x="Type 1", y="Attack", 
    title="Distribution of Pokémon Attack Across Type 1",
    labels={"Attack": "Attack Stat", "Type 1": "Pokémon Type 1"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
fig = px.box(pokeaman, x="Type 2", y="Attack", 
    title="Distribution of Pokémon Attack Across Type 2",
    labels={"Attack": "Attack Stat", "Type 2": "Pokémon Type 2"})
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

# The Homework this time around is VERY DIFFERENT
### It's VERY LONG. It's VERY, VERY DEMANDING. You will do/understand COMPLICATED SIMULATIONS
### You don't turn it in until AFTER you get back from READING WEEK (Thursday before TUT as usual)
### Your Project Proposals ARE DUE ON MONDAY IMMEDIATELY UPON RETURN FROM READING WEEK

- The HW is longer since there's substantially more time to do it
- However, I still need to finalize the HW and make the rubric, which 
    - I expect to do tomorrow, Tuesday Oct 22.
    - My apologies for not being quite ready this time around
    - And similarly, the textbook for linear regression has not yet been finalized 
        - but I will do so ASAP, ideally by tomorrow-tomorrow, Wednesday Oct 22.
- A draft of the "Course Project Proposals" assignment is available in the CP folder on the course github
    - This is due on Monday, Nov 04 the day you return from your reading week
    - I will alert the class with an announcement when the final I need to 

