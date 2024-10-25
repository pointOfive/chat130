# STA130 TUT 7ate9 (Oct25)<br><br>📈❓ <u>Simple Linear Regression</u><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Model Fitting / Hypothesis Testing)

## ♻️ 📚 Review / Questions [10 minutes]

### 1. Follow up questions and clarifications regarding the ideas of **correlation** and the "straight line association" model of **Simple Linear Regression** from the Oct21 LEC<br>

<details class="details-example"><summary><u><span style="color:blue">Simple Linear Regression Terminology</span></u> (reference for <b>Communication Activity #1 question 2</b> below)</summary>

$$ \Large Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

- **Outcome** $Y_i$ is a **continuous numeric variable**

> **Outcome** $Y_i$ can also be called a **response**, **dependent**, or **endogenous variable** in some domains and contexts

- **Predictor variable** $x_i$ is a **numeric variable**

> - Fow now we'll consider $x_i$ to be a **continuous** numeric variable, but this is not necessary, and we will consider versions of $x_i$ later
> - **Predictor variable** $x_i$ can also be called an **explanatory**, **independent**, or **exogenous variable**, or a **covariate** or **feature** (which are the preferred terms in the statistics and machine learning domains, respectively)

- **Intercept** $\beta_0$ and **slope** $\beta_1$ are the two primary **parameters** of a **Simple Linear Regression** model

> **Intercept** and **slope** describe a **linear** ("straigh line") relationship between **outcome** $Y_i$ and **predictor variable** $x_i$

- **Error** $\epsilon_i$ (also sometimes called the **noise**) makes **Simple Linear Regression** a **statistical model** by introducing a **random variable** with a **distribution**

- The $\sigma^2$ **parameter** is a part of the **noise distribution** and controls how much vertical variability/spread there is in the $Y_i$ data off of the line: $\sigma^2$ is an "auxiliary" **parameter** in the sense that interest is usually in $\beta_0$ and $\beta_1$ rather than $\sigma^2$

> - **Errors** $\epsilon_i$ (in conjuction with the **linear form**) define the **assumptions** of the **Simple Linear regression** Model specification
> - <u>but these **assumptions** are not the focus of further detailed reviewed here</u>

</details>    
    
<details class="details-example"><summary><u><span style="color:blue">Further details regarding the assumptions</span></u> (which <b>should not the focus of further detailed reviewed here</b>)</summary>

> The first three assumptions associated with the **Simple Linear regression** model are that<br><br>
> 
> 1. the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
> 2. the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
> 3. the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
>
>    and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that<br><br>
> 4. the $x_i$ **predictor variable** is **measured without error**
> 5. and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
> 6. and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)

</details><br>  
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> This TUT will introduce **Hypothesis Testing** in the **Simple Linear Regression** context for the purposes of evaluating a **null hypothesis** assumption of "no association" between two numeric variables $Y$ and $x$ relative to an **alternative hypothesis** of "straight line association" (meaning that changes in the $x$ variable have corresponding changes in the $Y$ variable "on average")
    
</details>

### 2. Follow up questions and clarifications regarding concepts associated with the **sampling distribution** topic <u>*[REALLY needs to be addressed in OH at this point]*<br><br> HW this time is Going To Be DIFFERENT: you MUST understand simulation to do it</u>

> AKA **Hypothesis Testing**, **Sampling Distribution under the Null Hypothesis**, and related topics regarding interpretation from Oct04 TUT and Oct11 TUT; AND, **Sampling Distribution**, **Bootstrapped Confidence Intervals**, and related topics regarding interpretation from Sep27 TUT and Sep30 LEC and Oct07 LEC 

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> Understanding the fundamental underlying mechanism of a **sampling distribution** (most easily demonstrated through **simulation**) is necessary for creating a deep understanding of its (and *the*) two primary applications in **statistics**: **Hypothesis Testing** and **[Bootstrapped] Confidence Intervals**
> 
> Further clear understanding regarding the abstract components and process of **Hypothesis Testing** (decision making regarding parameters using null hypotheses and p-values) is additionally needed from here, as this serves as an unavoidably necessary pre-requesite foundation upon which the enevitable extension of **Hypothesis Testing** to more advanced analyses (such **Multiple Linear Regression** and **permutation testing**, and **Simple Linear Regression** which will be the focus of this TUT) are based
    
</details><br>



## 💬 🗣️ Communication Activity #1 [20 minutes]

To the best of your abilty, recreate the <u>**FIVE**</u> groups from the **Communication Acivity** of the previous (Oct04 and Oct11) TUTs <br><br>

<details class="details-example"><summary style="color:blue"><u>Stella McStat's Wheel of Destiny</u></summary>

We should all by now hopefully be VERY familiar with this by this point in time given that this was a focus of the Oct04 and Oct11 TUTs **and was heavily featured on the midterm exam**...
    
### The Wheel of Destiny

Stella McStat had been running a small-time gambling operation on campus for several months during her first year at UofT... 

- For each spin of the wheel, two gamblers take part. For a toonie each (\\$2 Canadian), Stella sells one a red ticket and one a black ticket  (i.e., total \\$4). Then Stella spins the Wheel of Destiny. The person who holds the colour on which the spinner stops gets \\$3.50 (Stella keeps \\$0.50 per spin for running the game and providing snacks).

Stella just bought a new spinner, the critical piece of equipment for this game. She's heard some mixed reviews about the manufacturer she has purchased from. Before she beings using this spinner, she wants to make sure that it is, in fact, fair (meaning, she wants both colours to come up equally often). Because of the set-up of the game, Stella has no incentive to cheat and wants the game to be as fair as possible.

Everything phystical and mechanical that Stella can examine about the wheel seems fine; there is the same number of sectors of each colour and they each have the same area. BUT! Stella has a great idea and decides to come to YOU, her statistical guru, and ask you to verify that the new spinner is fit to use. Is Stella's game is "fair" (even if somewhat illegal)?

| <img src="https://i.postimg.cc/BvqJwBwc/stella2.png" style="height: 450px;"/> |  <img src="https://i.postimg.cc/vm3GRxJR/fair.png" style="height: 450px;"/> |
|-|-|
|An Exercise for Illustrating the Logic of Hypothesis Testing|Adapted from Lawton, L. (2009), Journal of Stat. Education, 17(2)|
    
</details>

### Discuss the following

1. **[8 of the 20 minutes]** What is the **Null** (and **Alternative**) **Hypothesis** and what is the definition of (and using **simulation** how do you estimate) a **p-value**? 

> First answer this question specifically for the context of "Stella McStat's Wheel of Destiny", but then see if you can give an answer that is more abstract and to some degree "context free" (in terms of **parameters** and [observed versus simulated] **statistics**)

2. **[12 of the 20 minutes]** Examine the theoretical **Simple Linear Regression** model below and consider what a **Null** (and **Alternative**) **Hypothesis** and **p-value** could be for this context? 

$$\Large Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

> **Hints**
> 
> 1. What is the data and how many data points are there?
>  
>  
> 2. What is $\epsilon_i$ and how many of them are there?
>  
>  
> 3. What values can the **slope** $\beta_0$ and **intercept** $\beta_1$ can have?  
>  
> 
> 4. Are **Null** and **Alternative Hypotheses** about **samples** like $x_i$ or $Y_i$ or **sample stastics** like $\bar Y$ or $\bar x$, or **population parameters** like $\mu$?
>  
>  
> 5. There's not a **Null** and **Alternative Hypotheses** regarding $\epsilon_i$, but there are plenty of assumptions about it (which technically are *a part* of the **Null hypothesis**)... what are those assumptions about $\epsilon_i$? 
>  
>  
> 6. Do you have any intuition of how to think about the conceptual meaning of a **p-value** (defined as "the probability that a test statistic is as or more extreme than the observed test statistic if the null hypothesis is true") in terms of **simulation** in the context of **Simple Linear Regression**?
>  
>  
> To be discussed in more detail shortly, the **fitted model** $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ corresponding to the theoretical model above is based on observed **sample data**. So, e.g., the **fitted slope** $\hat \beta_1$ is a **statistic** (which has a **sampling distribution**) that corresponds to the theoretical **slope** parameter $\beta_1$...
   

## Demo (of Model Fitting and Hypothesis Testing for the Simple Linear Regression Model)  [45 minutes]

### Terminology [12 of the 45 minutes]

$$\LARGE \text{Based on data we get} \quad \hat y_i = \hat \beta_0 + \hat \beta_1 x_i \quad \text{from}$$

$$\LARGE Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

<br>

The $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ **fitted model** equation distinctly contrasts with the $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$ **theoretical model** specification. To emphasize and clarify the difference, we augment our **simple linear regression** model nomenclature (as given in the "**Review / Questions**" section above) with the contrasting alternative notations and terminology: 

- **Fitted intercept** $\hat \beta_0$ and **slope** $\hat \beta_1$ ***coefficients*** are given "hats" to distinguish that they **estimate** (based on observed **sample data**), respectively, the **intercept** $\beta_0$ and **slope** $\beta_1$ ***parameters***<br><br>

- **Fitted (predicted) values** $\hat y_i$ are made lower case and also given "hats" to distinguish them from the (upper case) **theoretical random variable** $Y_i$ implied by the **theoretical simple linear regression model**
  
> Technically, the **error** $\epsilon_i$ is the **random variable** specified by the **simple linear regression model** specification, and this implies the **random variable** nature of $Y_i$ 

- The **residuals** $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i = y_i - \hat \beta_0 + \hat \beta_1 x_i $ also distinctly contrast with the **errors** (or **noises**) $\epsilon_i$
    
> The **residuals** $\text{e}_i = \hat \epsilon_i$ are actually available, while the **error** (or **noises**) $\epsilon_i$ are just a theoretical concept
> 
> The **residuals** $\text{e}_i = \hat \epsilon_i$ nonetheless are therefore used to diagnostically assess the theoretical modeling assumptions of the  **errors** $\epsilon_i$, such as the **normality**, **homoskedasticity**, and **linear form** assumptions; and, <u>while this is a not necessarily beyond the scope of STA130 and would certainly be a relevant consideration for the course project, this will not be addressed here at this time</u>
    

### Observed Data Setup [5 of the 45 minutes]

Imagine you noticed that the prices of shuttlecocks in the nearby store have increased. At the same time, suppose you are also aware that there has been a recent surge in bird flu cases. You suddenly wonder if there might be a connection between these two events. So you get some historical data as given in the format below.

|Bird Flu Cases |Shuttlecock Price ($)|
|:-------------:|:----------------------------:|
|1000           |3.0                           |
|1522           |4.2                           |
|$$\vdots$$     |$$\vdots$$                    |
|1200           |3.2                           |

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> Actually, this data and all the related (analysis and plotting) code was made by just giving instructions to a ChatBot, and tweaking things a little bit. As long as you know what to ask for and what you're looking for a ChatBot can carry out **Simple Linear Regression** and related tasks (and follow any specific analyses adjustment directions you  request).

</details>

#### In the visual representation of the data below, what are we considering the (dependent) outcome $Y_i$ and what are we considering the (independent) predictor $x_i$? Does this seem sensible given the framing of our inquiry? 

|<img src="https://www.mumbailive.com/images/news/bird-flu1_151660804012.jpg?w=1368" alt="Bird Flu" style="width: 500px; height: 300px;"/>|<img src="https://www.badmintonskills.net/wp-content/uploads/2015/09/Badminton-004.jpg?x83573" alt="Shuttlecock" style="width: 250px; height: 300px;"/>|
|-:|:-|
|Assess a possible association between bird flu prevalence and the price of shuttlecocks| using **Simple Linear Regression**|



```python
import pandas as pd
import plotly.express as px

# Here's the data
data = {
    'Bird Flu Cases': [1000, 1522, 1300, 
        1450, 1550, 1350, 1250, 1500, 1150, 1650, 1300, 1400, 1750, 
        1800, 900, 1100, 1700, 1400, 1600, 1200],
    'Shuttlecock Price': [3.0, 4.3, 3.7, 
        3.9, 4.0, 3.5, 3.4, 4.0, 3.2, 4.4, 3.6, 3.7, 4.6, 
        4.9, 2.8, 3.1, 4.6, 3.9, 4.4, 3.2]}
df = pd.DataFrame(data)

# Here's the data visually
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', 
                 title='Shuttlecock Price vs. Bird Flu Cases',
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```

### *Model Fitting* via `statsmodels.formula.api as smf` and `smf.ols(...).fit()`<br>[16 of the 45 minutes]

- First we'll demonstrate getting a fitted Simple Linear Regression Model using `statsmodels`  and working with the key elements of the fitted model [10 of these 15 minutes]  


- Then we'll visually demonstrate the fitted Simple Linear Regression model based on its estimated intercept and slope parameters [5 of these 15 minutes]


```python
import statsmodels.formula.api as smf

# And here's how to do "Model Fitting" for Simple Linear Regression with `statsmodels`

# Use "Y~x" R-style formulas: https://www.statsmodels.org/stable/example_formulas.html
linear_specification = 'Q("Shuttlecock Price") ~ Q("Bird Flu Cases")'
# The notation above is admittidly starnge, but it's because 
# there are spaces in my column (variable) names in the data

# Put the data into the a `statsmodels` "model" object
model_data_specification = smf.ols(linear_specification, data=df)

# Fit the model
fitted_model = model_data_specification.fit()
```


```python
# See the results...
fitted_model.summary()
```


```python
# There's too much in the full `.summary()`: just focus on this table for now
fitted_model.summary().tables[1] # does this indexing make sense?
```

$$\LARGE \text{For the data above} \quad \hat y_i = 0.5361+0.0023 \times x_i$$

$$\LARGE \text{is the} \quad \hat y_i = \hat \beta_0 + \hat \beta_1 x_i \quad \text{estimating the model}$$

$$\LARGE Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

#### What are $\hat \beta_0$ and $\hat \beta_1$? What are $Y_i, x_i$, and $\hat y_i$ and $\text{e}_i = \hat \epsilon_i$? How do you interpret the fitted model?<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> When we create a **fitted Simple Linear Regression model** for an observed data dataset, we obtain **estimates** of what the data suggests the **intercept** and **slope** could be to form an equation of the (predicted) values for the data
> 
> Model fitting is typically based on the "ordinary least squares" (`ols`) concept (maximizing **R-squared**), although there are analytical closed form solutions for the **intercept** **slope coefficient estimates** for **Simple Linear Regression models**; but, these considerations will not be detailed further here (as they'll be the focus of a HW question) and <u>we'll now instead just focus on fitting models using the `python` `statsmodels` package</u>
    
</details>







```python
Y = df["Shuttlecock Price"]
x = df["Bird Flu Cases"]

# The printout coeficient values are rounded
# pd.DataFrame({"formula": 0.4291+0.0024*x,
#               "model": fitted_model.fittedvalues})

# So we use the exact fitted coefficient values using `fitted_model.params`# (actually, `fitted_model.params.values[0]`)
y_hat = fitted_model.fittedvalues
df['y-hat (from model)'] = y_hat 
df['y-hat (from formula)'] = 0.536107332688726+0.0023492341183347265*x
df

# `fitted_model.fittedvalues` is the same as `fitted_model.predict(df)`
# but the latter is more genereal 
# as it could be used to predict new values based on a different data frame...
```


```python
# residuals
e = Y - fitted_model.fittedvalues  # df['Shuttlecock Price'] - fitted_model.fittedvalues
df['e (Residuals)'] = e
# or you can just use `fitted_model.resid`
df['e (Residuals) v2'] = fitted_model.resid
df
```

- The **intercept** $\hat \beta_0$ is `0.5361`
- The **slope** $\hat \beta_1$  is `0.0023` and is labeled `Q("Bird Flu Cases")` in the output
    - The **outcome** $Y_i$ is `Q("Shuttlecock Price")`
    - The **predictor** $x_i$ is `Q("Bird Flu Cases")`
    - The **slope** is the "on average" change in $Y_i$ per "single unit" change in $x_i$
   
- A **fitted (predicted) value** $\hat y_i$ is found by "plugging in" $x_i$ and calculating $0.5361+0.0023 \times x_i$
- A **residual** is calculated as $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i = y_i - \hat \beta_0 + \hat \beta_1 x_i $



```python
# Code is commented to indicate its visualization/demonstration purpose:
# students may study smaller details beyond the "big picture" later 
# in a ChatBot session if so inclined

import numpy as np

# Here's the model fit visually
df['Original Data'] = 'Original Data' # hack to add legend item for data points
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', color='Original Data',
                 title='Shuttlecock Price vs. Bird Flu Cases',
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'},
                 trendline='ols')
fig.update_traces(marker=dict(size=10))
              
# This is what `trendline='ols'` does
fig.add_scatter(x=df['Bird Flu Cases'], y=fitted_model.fittedvalues,
                line=dict(color='blue', width=3), name="trendline='ols'")
    
# Adding the line of the math expression
x_range = np.array([df['Bird Flu Cases'].min(), df['Bird Flu Cases'].max()])
y_line = 0.536107332688726+0.0023492341183347265 * x_range
fig.add_scatter(x=x_range, y=y_line, mode='lines', name='0.5361 + 0.0023 * x', 
                line=dict(dash='dot', color='orange'))

# Adding predicted values as points
fig.add_scatter(x=df['Bird Flu Cases'], y=df['y-hat (from model)'], mode='markers', 
                name='Fitted (Predicted) Values', 
                marker=dict(color='black', symbol='cross', size=10))

fig.update_layout(legend_title=None)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### [Omitted] Model Diagnostics: evaluating the assumptions of Simple Linear Regression [0 of the 45 minutes]


#### To evaluate if the assumptions of "normality" and "heteroskedasticity" ($x_i$ agnostic variance) of the theoretical distribution of the error (noise) terms we see if the residuals appear to be normally distributed... this is not really enough data to determine this convincingly one way or another at this point

#### In the context of Simple Linear Regression (as opposed to Multiple Linear Regression), we could examine the scatter plot to see if the assumption of a "linear form of the model" appears "true" plot... in the original scatter plot of the data there appears to be some potential "curve" in the relationship, but again there's really not enough data to determine this convincingly "by eye" at this point


```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

from scipy import stats

n = len(df['e (Residuals)'])
normality_heteroskedasticity_diagnostic_judgement = \
'<br>[Seems to plausibly be a (n='+str(n)+') sample from a "normal" distribution]'
df['Observed Residuals'] = 'Observed Residuals' # hack to add legend item for data points
fig = px.histogram(df, x='e (Residuals)', color='Observed Residuals',
                   title='Histogram of Residuals'+normality_heteroskedasticity_diagnostic_judgement)

# rerun this cell to see repeated examples
random_normal_sample = stats.norm(loc=0, scale=df['e (Residuals)'].std()).rvs(size=n) 

fig.add_histogram(x=random_normal_sample, name='Random Normal Sample', 
                  histfunc='count', opacity=0.5, marker=dict(color='orange'))
fig.update_layout(barmode='overlay', legend_title=None)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

linearity_diagnostic_judgement = '"<br>[Straight line" fit appears "reasonable"]'
# uncomment/comment `trendline='ols'` to toggle the "straight line fit" on and off
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', color='Original Data',
                 #trendline='ols',
                 title='Shuttlecock Price vs. Bird Flu Cases'+linearity_diagnostic_judgement,
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### Hypothesis Testing (for Simple Linear Regression) [12 of the 45 minutes]

We can use **Simple Linear Regression** to test

$\large
\begin{align}
H_0: {}& \beta_1=0 \quad \text{ (there is no linear assocation between $Y_i$ and $x_i$ "on average")}\\
H_A: {}& H_0 \text{ is false}
\end{align}$

That is, we can assess the evidence of a linear association in the data based on a **null hypothesis** that the **slope** (the "on average" change in $Y_i$ per "single unit" change in $x_i$) is zero

#### Did your group get $H_0$ correct in your answers for *Communication Activity #1 question 2*?<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> We are essentially never (or only very rarely in very special circumstances) interested in a **null hypothesis** concerning the **intercept** $\beta_0$ (as opposed to $\beta_1$)
> 
> $\Large
\begin{align}
H_0: {}& \beta_0=0\\
H_A: {}& H_0 \text{ is false}
\end{align}$
>
> This is because the assumption that $\beta_0$ is zero essentially never (or only very rarely in very special circumstances) has any meaning, whereas the assumption that $\beta_1$ is zero has the very practically useful interpretation of "no linear association" which allows us to evaluate the  evidence of a linear association based on observed data
    
</details>

#### How do we use the fitted Simple Linear Regression model to assess $H_0$ regarding $\beta_1$? Where do we find the p-value we use to make our assessment of $H_0$ and how do we interpret the p-value to make a decision?<br>
 
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
 
Remember, the **p-value** is "the probability that a test statistic is as or more extreme than the observed test statistic if the null hypothesis is true"

- We do not prove $H_0$ false, we instead give evidence against the $H_0$
     - "We reject the null hypothesis with a p-value of abc, meaning we have xyz evidence against the null hypothesis"
- We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$
     - "We fail to reject the null hypothesis with a p-value of abc"
|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
   
</details>
    


```python
fitted_model.summary().tables[1] # does this indexing make sense?
```

## 📢 👂 Communication Activity #2 [final 15 minutes]

### If you don't complete this NOW in TUT students will have to complete it ON THEIR OWN for the <u>EXTREMELY VERY BIG AND VERY DIFFERENT</u> Homework 06 <sub>*HW06_Week07ate09_DueNov07STA130_HW06_Week07ate09_DueNov07STA130_HW06_Week07ate09_DueNov07STA130*</sub>

In order to follow up and explain the answers to **Communication Activity #1 question 2**, each of the <u>**FIVE**</u> groups from **Communication Activity #1** will sequentially volunteer to present answers to these questions (**taking average three minutes per group**) to the class (in order as quickly as possible, with groups dynamically helping each other answer their question if needed) 

1. Explain how the "uncertainty band" in the `seaborn.regplot` of the **Further Illustrations** below represents a **bootstrapped sampling distribution** (or slighly more accurately something like a "95% confidence interval") for "lines of best fit"  


2. Explain how this so-called "**sampling distribution**" of the "line of best fit" could be "sampled"
    1. by making a **bootstrapped sampling distribution** 
    2. by assuming the **population model** and creating **simulations** 


3. Explain how the **sampling distribution** of (just) the **estimated slope** $\hat \beta_1$ could be **simulated** and the **p-value** for a **null hypothesis** of "no linear association" created using **simulation** and used to assess the evidence against the **null hypothesis**  
    1. Also explain how a **95% bootstrapped confidence interval** of the **slope coefficient** $\hat \beta_1$ could be constructed


4. Find the "R-squared" in the `fitted_model.summary()` table (or accessible via `fitted_model.rsquared`) and compare this value with 
    1. `np.corrcoef(Y,x)[0,1]**2`,  
    2. `np.corrcoef(Y,y_hat)[0,1]**2`,   
    3. and `1-((Y-y_hat)**2).sum()/((Y-Y.mean())**2).sum()` (where `Y`,`x`, and `y_hat` have been defined in the notebook above for the orignal data); 
    4. then, explain (a) what the two `np.corrcoef...` expressions capture, (b) why the final expression can be interpreted as "the proportion of variation in (outcome) Y explained by the model (y_hat)", and (c) therefore why `fitted_model.rsquared` can be interpreted as a measure of the accuracy of the model  


5. Explain what our likely judgement about the **Model Diagnostics** in the <u>**Omitted**</u> section above would be for data **simulated** based on an assumed **population model** (as opposed to using **bootstrapping**), and what would cause an analogous judgement them to fail for observed data (and what this means about the appropriateness of the theoretical **Simple Linear Regression** model for this data)


### Further Illustration 

> The `seaborn` plotting library has a function which shows the uncertainty of the (trendline) "straight line fit" based on the available data: anywhere a straight line can be drawn through the "uncertainty band" is plausible as far as the evidence in the observed data is concerned

#### How does the vertical spread of the $Y_i$ outcomes (tightness around the "line of best fit") affect the evidence against the null hypothesis? What does this mean in terms of our belief about the evidence against a *null hypothesis* of no linear association "on average" between Bird Flu Cases and Shuttlecock Price? 



```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

import seaborn as sns
import matplotlib.pyplot as plt

spread = 20 # increase this to vertically spread the data (1 recreated the original data)
df["Synthetically Spread y"] = fitted_model.fittedvalues + spread*df['e (Residuals)']

linear_specification_ = 'Q("Synthetically Spread y") ~ Q("Bird Flu Cases")'
model_data_specification_ = smf.ols(linear_specification_, data=df)
fitted_model_ = model_data_specification_.fit()
print(fitted_model_.summary().tables[1])

sns.regplot(x='Bird Flu Cases', y='Synthetically Spread y', data=df) #, line_kws={'color': 'red'})
plt.title('Bird Flu Cases vs. Shuttlecock Price ($)')
plt.xlabel('Bird Flu Cases')
plt.ylabel('Shuttlecock Price ($)')
plt.show()
```


```python

```
