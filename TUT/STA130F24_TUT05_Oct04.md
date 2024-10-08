# STA130 TUT 05 (Oct04)<br><br>🤔❓ <u>"Single Sample" Hypothesis Testing<u>


## ♻️ 📚 Review  / Questions [15 minutes]

### 1. Follow up questions and clarifications regarding **bootstrapping, sampling distributions**, and **confidence intervals**
 
> such as how the **sampling distribution** of a **sample statistic** such as the **sample mean** $\bar x$ is used understand the **variability/uncertainty** of the **sample statistic** and thereby provide **statistical inference** beyond simple **point estimation** of a corresponding **population parameter** μ... *or address a* **null hypothesis** *about* μ
>
> - 1. Or, you might determine that, even before all of this, it would be more valuable to return to address the underlying fundamental notion of the meaning of **variance** 
> - 2. AKA what's the difference the variability of the **population** or the **sample** data points themselves versus the **variability/uncertainty** of a **sample statistic**?

### 2. Why is "single sample" in quotes in the TUT title? Hint: examine the data loaded in Demo I<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> Sep27 TUT and Sep30 LEC of the previous week addressed **sampling distributions** and **bootsrapped confidence intervals** and the **HW** breifly introduced the notion of using the **variability/uncertainty** of the **samping distribution** of a **sample statistic** (driven by the **sample size** $n$) to evaluate a **null hypothesis** about a corresponding **population parameter**
>
> This week builds on this concept and formally introduces **hypothesis testing** with **null** and **alternative hypotheses**, which will be much easier to understand if the concept and purpose of a **samping distribution** and **confidence intervals** is well understood...


## 🚧 🏗️ Demo I (introducing formal Hypothesis Testing) [15 minutes]

> The scientific method is most fundamentally the process of providing evidence against the current views. You have to provide evidence against old views in order to reject the old hypotheses before you can move on to a new paradigm.

|<img src="https://pictures.abebooks.com/inventory/md/md31377899338.jpg" alt="Scientific Revolusions" style="width: 300px; height: 250px;"/>|<img src="https://i.ytimg.com/vi/Yn8cCDtVd5w/maxresdefault.jpg" alt="Kuhn Cycle" style="width: 800px; height: 250px;"/>|
|-|-|
| | |

### Let's return to the "[Vaccine Data Analysis Assignment](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW04_DueOct03.ipynb)" [last week's (week 4) HW Question "8"] 

- Let's review the goal of that problem and remind you of the **simulation** based appproach that could address **null hypothesis** aspect of that problem; namely, **formal hypothesis testing** based on **bootstrapped confidence intervals**



```python
import pandas as pd

patient_data = pd.DataFrame({
    "PatientID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [45, 34, 29, 52, 37, 41, 33, 48, 26, 39],
    "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    "InitialHealthScore": [84, 78, 83, 81, 81, 80, 79, 85, 76, 83],
    "FinalHealthScore": [86, 86, 80, 86, 84, 86, 86, 82, 83, 84]
})
patient_data
```


```python
# First let's format this data in the manner of last week's HW "Prelecture" video
# from IPython.display import YouTubeVideo
# YouTubeVideo('Xz0x-8-cgaQ', width=800, height=500)  # https://www.youtube.com/watch?v=Xz0x-8-cgaQ

patient_data['HealthScoreChange'] = patient_data.FinalHealthScore-patient_data.InitialHealthScore
# why do we do the subtraction in this order?
patient_data
```


### The <u>Null Hypothesis</u> [and Alternative Hypothesis]

The **null hypothesis** usually simply states the "no effect" (on average) assumption

$\large H_0: \text{The vaccine has no effect }\textbf{(on average)}\text{ on patient health}\\
\large H_1: H_0 \text{ is false}$

To empasize that "**(on average)**" refers to the pupulation parameter $\mu$ (the average effect), it is helpful to more formally (and concisely) express this equivalently as 

$$\Large H_0: \mu=0 \quad \text{ and } \quad H_A: H_0 \text{ is false}$$<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
As introduced in the "Further Guidance" to last weeks (Week 4) HW Question "7"...

> **Statistical hypothesis testing** proceeds on the basis of the **scientific method** by defining the **null hypothesis** to be what we beleive until we have sufficient evidence to no longer believe it. As such, the **null hypotheses** is typically something that we _may not actually believe_; and, actually, the **null hypotheses** simply serves as a sort of "straw man" which we in fact really intend to give evidence against so as to no longer believe it (and hence move forward following the procedure of the **scientific method**).
</details>

<details class="details-example"><summary style="color:blue"><u>Even Further Guidance</u></summary>    

**There some assumptions "hidden" here.**
Differences in the "before and after" `HealthScore` could be due to a lot of factors; but, if the only thing we did as an intervention was giving the patients the vaccine treatment, then we would expect the other factors to be a wash over time and just kind of average out... right?
- Do we think something else could happen that would tend to generally increase everyone's health score after the initial measurement (besides our intervention)? 
    - If so, this would be called a **confounder**... otherwise we're saying we have "**no confounding**"
- Do we think we have a large enough sample size for "other factors" to "average out"? 
    - Usually we consider increased sample size from the perspective of reducing standard error to reduce estimation uncertainty; but, this consideration suggests we should also be concerned with sample size from the perspective of "averaging out" **confounding imbalances**...
</details>

### Now let's demonstrate formal hypothesis testing using simulation...


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

### Why and with what "confidence" do we reject $H_0$ based on the interval above?

- *Hint: the figure below shows the distribution of bootstrapped means which are the "plausible average Health Score Change" (for the given sample size, insofar as the sample is representative of the population...); so, "0" means "no effect on average"...*



```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```


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

## 🔨 💪🏼 Demo II (of Hypothesis Testing using p-values) [30 minutes]<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
The above illustrates **rejecting a null hypothesis** $H_0$ on the basis of a **bootstrapped confidence interval** at a 95% **confidence level** (since the interval "does not cover 0")

- This is an ideal way to address hypothesis testing, but it's (unfortunately)  also quite common to give "evidence against" a null hypothesis in the form of a p-value
</details>
         
A **p-value** is **the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true**
 
> To understand what the definition of a **p-value** means, let's consider the definition in reverse 
> 
> 1. What is the meaning of "if the null hypothesis was true"?
> 2. What is the meaning of "a statistic is as or more extreme than the observed statistic"? 
> 3. What is the meaning of "the probability that a statistic is..."?

### 1. "if the null hypothesis is true"...

> $$H_0: \text{The vaccine has no effect }\textbf{(on average)}\text{ on patient health}$$
> 
> implies that improvements or reductions between `FinalHealthScore` and `InitialHealthScore` in individual observations are actually really just "random"
>
> _We could therefore just simulate sampling distribution of the "proportion of cases that improved" under the assumption of the null hypothesis that the signs of the differences between `InitialHealthScore` and `FinalHealthScore` is actually really just as random as the process of flipping a fair coin._
>
> We'll therefore use the following slightly different version **null hypothesis**
> 
> $$H_0: \text{The chance the vaccine improves patient health} \textbf{ is 50%}$$

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
 
> _We're changing $H_0$ for two reasons; first, this is the version that we want to introduce and construct the foundation of hypothesis testing with; and, second, for a "technical" reason this null hypothesis is also more amenable to the simulation approaches that we're leveraging in STA130._
>
> - After seeing how we can use **simulation** to address $H_0: \text{The chance the vaccine improves patient health} \textbf{ is 50%}$ using formal **hypythesis testing**, a very good challenge for students for confirming understanding would be to determine how **bootstrapping** could be used to **estimate** the "chance the vaccine improves patient health" through a **confidence interval**.
    
</details>    


```python
# Do you get the idea here?
# Can you see what's chaning in the output below??

print(pd.DataFrame({'HealthScoreChange': patient_data['HealthScoreChange'],
                    '> 0 ?': patient_data['HealthScoreChange']>0}))

random_difference_sign = np.random.choice([-1, 1], size=len(patient_data))
pd.DataFrame({'HealthScoreChange': random_difference_sign*patient_data['HealthScoreChange'].abs(),
              '> 0 ?': (random_difference_sign*patient_data['HealthScoreChange'])>0})
```


```python
# And then can you see what's happening here???

np.random.seed(1)  # make simulation reproducible
number_of_simulations = 10000  # experiment with this... what does this do?
n_size = len(patient_data)  # 10
IncreaseProportionSimulations_underH0random = np.zeros(number_of_simulations)

# generate "random improvement" proportions assuming H0 (vaccine has no average effect) is true 
# meaning that the "before and after" differences are positive or negative at "random"
for i in range(number_of_simulations):
    
    # why is this equivalent to the suggested idea above?
    random_improvement = np.random.choice([0,1], size=len(patient_data), replace=True)  # <<< `replace=True` ^^^

    # why is .mean() a proportion? 
    IncreaseProportionSimulations_underH0random[i] = random_improvement.mean()
    # why is this the statistic we're interested in? Hint: next section...
```

### 2. "a statistic is as or more extreme than the observed statistic"...

> To understand "as or more extreme" we first need to consider $H_0$ formally in terms of the hypothesized population parameter 
> 
> \begin{align*}
H_0: p=0.5 \quad &{} \text{instead of the equivalent} \\
&{} H_0: \text{The chance the vaccine improves patient health} \textbf{ is 50%}
\end{align*}
> 
> **This is because "as or more extreme" is relative to a hypothesized population parameter which the statistic estimates**
> - **THEN, NEXT, we need to clearly differentiate and compare the "simulated statistcs" from the "observed statistic"**



```python
# "as or more extreme" relative to the hypothesized parameter of the statistic!
population_parameter_value_under_H0 = 0.5

observed_test_statistic = (patient_data.HealthScoreChange>0).mean()
simulated_test_statistics = IncreaseProportionSimulations_underH0random

SimTestStats_as_or_more_extreme_than_ObsTestStat = \
    abs(simulated_test_statistics - population_parameter_value_under_H0) >= \
    abs(observed_test_statistic - population_parameter_value_under_H0) 
    
print('''Which simulated statistics are "as or more extreme"
than the observed statistic? (of ''', observed_test_statistic, ')', sep="")

pd.DataFrame({'(Simulated) Statistic': simulated_test_statistics,
              '>= '+str(observed_test_statistic)+" ?": ['>= '+str(observed_test_statistic)+" ?"]*number_of_simulations, 
              '"as or more extreme"?': SimTestStats_as_or_more_extreme_than_ObsTestStat})
```

**When the simulation (two code cells back) is based on `np.random.seed(1)` the output above includes examples of `True` for `0.8 >= 0.8` AND**

**`0.1 >= 0.8`**

**WTFWTFWTFWTF omglmfao WHY???**

**WWWWWWHHHHHHHYYYYYYYYYYYYYYiiiiiiiiiiiiiii!!!!!!!!!!!!!????????????**

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Keep-calm-and-carry-on-scan.jpg/640px-Keep-calm-and-carry-on-scan.jpg)

We've got your answer down below.

![](https://www.gardencourtantiques.com/wp-content/uploads/2016/11/Keep-calm-carry-on-series.jpg)


```python
# figure for demonstration only: code details not of primary concern

hist_data = [IncreaseProportionSimulations_underH0random+np.random.uniform(-0.05,0.05,size=len(IncreaseProportionSimulations_underH0random))]
group_labels = ['Bootstrap<br>Sampling<br>Distribution<br>of the<br>Sample<br>Mean<br><br>assuming<br>that the<br>H0 null<br>hypothesis<br>IS TRUE']
fig = ff.create_distplot(hist_data, group_labels, curve_type='normal',
                         show_hist=True, show_rug=False, bin_size=0.1)
pv_y = 2.5
pv_y_ = .25
fig.add_shape(type="line", x0=observed_test_statistic, y0=0, 
              x1=observed_test_statistic, y1=pv_y,
              line=dict(color="Green", width=4), name="Observed Statistic")
fig.add_trace(go.Scatter(x=[observed_test_statistic], y=[pv_y+pv_y_], 
                         text=["Observed<br>Statistic<br>^"], mode="text", showlegend=False))
# "as or more extreme" also include the "symmetric" observed statistic...
symmetric_test_statistic = population_parameter_value_under_H0 -\
                           abs(observed_test_statistic-population_parameter_value_under_H0)
fig.add_shape(type="line", x0=symmetric_test_statistic, y0=0, 
              x1=symmetric_test_statistic, y1=pv_y,
              line=dict(color="Green", width=4), name="Observed Statistic")
fig.add_trace(go.Scatter(x=[symmetric_test_statistic], y=[pv_y+pv_y_], 
                         text=['"Symmetric" Observed Statistic<br>addrdssing for "as or more extreme"<br>^'], mode="text", showlegend=False))

# Add a transparent rectangle for the lower extreme region
fig.add_shape(type="rect", x0=-0.25, y0=0, x1=symmetric_test_statistic, y1=pv_y,
              fillcolor="LightCoral", opacity=0.5, line_width=0)
# Add a transparent rectangle for the upper extreme region
fig.add_shape(type="rect", x0=observed_test_statistic, y0=0, x1=1.25, y1=pv_y,
              fillcolor="LightCoral", opacity=0.5, line_width=0)

# Update layout
fig.update_layout(
    title="Bootstrapped Sampling Distribution<br>under H0 with p-value regions",
    xaxis_title="Mean Health Score Change", yaxis_title="Density", yaxis=dict(range=[0, pv_y+2*pv_y_]))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

**Very Clarifying Clarification Questions (if you understand the answer)**
1. What is the difference in the "middle value" of
   1. the **bootstrap sampling distribution of the sample mean**
   2. VERSUS the **bootstrapped sampling distribution of the sample mean under the (assumption of) null hypothesis?**
   3. *Hint: compare this figure to the initial bootstrapped confidence interval figure of the TUT notebook*
2. What's the difference between the **bootstrapped confidence intervals** "interval" compared to the "as or more extreme" regions which **p-values** are based on? 
    1. So does the **p-value** number below match the figure above (of course allowing for the nuance that the figure above is on a "density" rather than counts scale)?

### 3. "the probability that a statistic is..."<br>["as or more extreme" than the observed statistic]<br>(if the null hypothesis is true)




```python
# Calculate the p-value
# How many bootstrapped statistics generated under H0 
# are "as or more extreme" than the observed statistic 
# (relative to the hypothesized population parameter)? 

observed_test_statistic = (patient_data.HealthScoreChange>0).mean()
simulated_test_statistics = IncreaseProportionSimulations_underH0random

# Be careful with "as or more extreme" as it's symmetric!
SimTestStats_as_or_more_extreme_than_ObsTestStat = \
    abs(simulated_test_statistics - population_parameter_value_under_H0) >= \
    abs(observed_test_statistic - population_parameter_value_under_H0)
    
p_value = (SimTestStats_as_or_more_extreme_than_ObsTestStat).sum() / number_of_simulations
print("Number of Simulations: ", number_of_simulations, "\n\n",
      "Number of simulated statistics (under HO)\n",
      'that are "as or more extreme" than the observed statistic: ',
      SimTestStats_as_or_more_extreme_than_ObsTestStat.sum(), "\n\n",
      'p-value\n(= simulations "as or more extreme" / total simulations): ', p_value, sep="")
```

### But does a p-value mean?

This is easy: the smaller the p-value, the stronger the evidence against the null hypothesis

### Wait, but why? 

A **p-value** is "the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true"
- So if the **p-value** is small, then the observed statistic is very strange relative to the null hypothesis
- This means the data is very unusual if the null hypothesis is true, so it's probably more likely that the null hypothesis is false

## 💬 🗣️ Communication Activity<br>🎲 🃏 Stella McStat's Wheel of Destiny  [40 minutes]

**[~3 of the 40 minutes]** Break into 5 new groups of 4-5, assigning each group to one of the questions. 

**[~12 of the 40 minutes]** Review and discuss the questions within the group. 

**[~25 of the 40 minutes / 5 minutes per group]** As soon as a group (in order) is ready **[possibly even before the first 12 minutes are up]**, they should immediately **introduce their general topic and questions** and discuss their answers with the class; each group should build on the answers of the previous group **[perhaps requesting additional time to do so if the initial 12 minutes or review and discussion have not yet been used up]**, with the previous groups ensuring that the next groups are indeed taking advantage of the foundation their answers and discussions have provided.

> This is expected to be a dynamic sequentially dependent process (**not independent sampes!**) in which all groups should work together by contributing their part in order to complete the overall process within 40 minutes.

### The Wheel of Destiny

Stella McStat had been running a small-time gambling operation on campus for several months during her first year at UofT... 

- For each spin of the wheel, two gamblers take part. For a toonie each (\\$2 Canadian), Stella sells one a red ticket and one a black ticket  (i.e., total \\$4). Then Stella spins the Wheel of Destiny. The person who holds the colour on which the spinner stops gets \\$3.50 (Stella keeps \\$0.50 per spin for running the game and providing snacks).

Stella just bought a new spinner, the critical piece of equipment for this game. She's heard some mixed reviews about the manufacturer she has purchased from. Before she beings using this spinner, she wants to make sure that it is, in fact, fair (meaning, she wants both colours to come up equally often). Because of the set-up of the game, Stella has no incentive to cheat and wants the game to be as fair as possible.

Everything phystical and mechanical that Stella can examine about the wheel seems fine; there is the same number of sectors of each colour and they each have the same area. BUT! Stella has a great idea and decides to come to YOU, her statistical guru, and ask you to verify that the new spinner is fit to use. Is Stella's game is "fair" (even if somewhat illegal)?


| <img src="https://i.postimg.cc/BvqJwBwc/stella2.png" style="height: 450px;"/> |  <img src="https://i.postimg.cc/vm3GRxJR/fair.png" style="height: 450px;"/> |
|-|-|
|An Exercise for Illustrating the Logic of Hypothesis Testing|Adapted from Lawton, L. (2009), Journal of Stat. Education, 17(2)|




1. What's "data" here?<br><br>
    1. What is a **sample** here? Hint: the **population** would be every spin result ever 
    2. Do you think spins comprising a sample are **dependent** or **independent**?
    3. What is the difference between a **parameter** and a **statistic**, illustrated in this context?<br><br>

2. How could we create a **bootstrapped confidence interval** to estimate the proportion of times spins land on red? *Hint and Warning: this is not asking you to spin the wheel forever to arrive at your "best guess" of the proportion... this is a "purely academic" exercise in the process of contructing bootstrapped confidence intervals*<br><br>
    1. What statistic should the **confidence interval** be based on?
    2. What exactly would the process be to create a **bootstrapped confidence interval** for this context? That is, what exactly are the steps of the "**physical** and/or **simulation**" process you would carry out?
    3. Besides changing the **confidence level** (e.g., from 95% to 90%), how else could we make the confidence interval narrower (and why is this preferrable)?<br><br> 

3. How can we examine the wheel for fairness from a statistical perspective?<br><br>
    1. What is the difference between a **null hypothesis** and an **alternative hypothesis**? 
    2. What are the **null** and **alternative hypotheses** here?
    3. How could you use a **confidence interval** to make a decision about a **null hypothesis** that the wheel is fair?<br><br>

4. How could we **simulate** the **sampling distribution** of the **proportion of times spins land on red for a hypothetically fair wheel** (as opposed to the wheel Stella actually has)?<br><br>
    1. How could you simulate the data needed to create the **sampling distribution**?
    2. What **statistic** should the **sampling distribution** be based on, and what should the **sample size** be for the samples on which the **sampling distribution** is built?
    3. How is the proces different than the process for creating a **confidence interval** (from questions 2)?<br><br>
    
5. How could we provide a **p-value** for a **null hypothesis** of "fairness"?<br><br>
    1. What is the definition of a **p-value**?
    2. How would the **simulation** of the **sampling distribution** be used to calculate a **p-value** for this problem? *Hint: you'll need one more thing having to do with with the the* **sample size** *used to* **simulate** *the* **sampling distribution under the null**
    3. How would you interpret a p-value you obtained through this process in terms of the evidence it potentially provides against the null hypothesis? 
    
    
|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|    
