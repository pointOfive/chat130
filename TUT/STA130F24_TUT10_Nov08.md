# STA130 TUT 07 (Nov08)<br><br> 👪 📈 <u>Multiple Linear Regression</u>

## ♻️ 📚 Review  / Questions [15 minutes]

### 1. <u>Follow up questions and clarifications regarding concepts associated with **Simple Linear Regression**</u>
    
> First introduced on Oct21 Oct26, and conlcluded on Nov04 and Nov07... The theoretical "normal distribution" regression model, model fitting, hypothesis testing within this framework, and indicator variables and (two-sample) group comparision...
> 
> - *You have GOT TO KNOW THIS ac rf bc*
> - **Multiple linear regression** extends the **linear form** examined so far to multiple **predictor variables**; similarly extends the idea of **binary indicator variables** to **categorical variables**  with more than two levels; additionally provides both similar and extending **hypothesis testing** capabilities with the introduction of so-called **interaction variables** (which can be used, e.g., to examine the evidence that a linear association present between two variables differs across different groups of subpopulation groups in a dataset). 

<details class="details-example"><summary><u><span style="color:blue">The theoretical "normal distribution" regression model</span></u></summary>

### The theoretical "normal distribution" regression model
    
$$\Large Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

- **Outcome** $Y_i$ is a **continuous numeric variable**
    - **Outcome** $Y_i$ can also be called a **response**, **dependent**, or **endogenous variable** in some domains and contexts

    
- **Predictor variable** $x_i$ is a **numeric variable**
    - Fow now we'll consider $x_i$ to be a **continuous** numeric variable, but this is not necessary, and we will consider other versions of $x_i$ later
    - **Predictor variable** $x_i$ can also be called an **explanatory**, **independent**, or **exogenous variable**, or a **covariate** or **feature** (the last two of which are the preferred terms in the *statistics* and *machine learning* disciplines, respectively)

    
- **Intercept** $\beta_0$ and **slope** $\beta_1$ are the two primary **parameters** of a **Simple Linear Regression** model
    - **Intercept** and **slope** describe a **linear** ("straight line") relationship between **outcome** $Y_i$ and **predictor variable** $x_i$

    
- **Error** $\epsilon_i$ (also sometimes called the **noise**) makes **Simple Linear Regression** a **statistical model** by introducing a **random variable** with a **distribution**

    - The $\sigma^2$ **parameter** is a part of the **noise distribution** and controls how much vertical variability/spread there is in the $Y_i$ data away from the line: $\sigma^2$ (and $\beta_0$ as well to be honest) is an "auxiliary" **parameter** in the sense that interest is usually in $\beta_1$ rather than $\sigma^2$
    - **Errors** $\epsilon_i$ (in conjuction with the **linear form**) define the **assumptions** of the **Simple Linear regression** Model specification
    - <u>but these **assumptions** are not the focus of further detailed reviewed here</u>
    
</details>

<details class="details-example"><summary style="color:blue"><u>The implication of the specification on the models assumptions</u></summary>

### The implication of the specification on the models assumptions are as follows...
    
> The first three assumptions associated with the **Simple Linear regression** model are that
> 
> 1. the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
> 2. the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
> 3. and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
> 4. the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
>
>    and there are just a couple; but, a super deeper reflection on these at this point is "beyond the scope" of STA130; nonetheless, they are not too hard to understand and are that<br><br>
> 5. the $x_i$ **predictor variable** is assumed to be **measured without error** (so they are taken to have no *randomness* and therefore not have any *distributional* nature: they are just numbers **not random variables**)
> 6. and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)

</details> 
  
<details class="details-example"><summary><u><span style="color:blue">Model fitting for Simple Linear Regression</span></u></summary>

### Model fitting for Simple Linear Regression
    
> Do you remember how to use `statsmodels.ols(...).fit().summary()`? 
> 
> Hopefully so, but if not, today's demo with **multiple linear regression** will likely refresh your memory.
    
The $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ **fitted model** equation distinctly contrasts with the $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$ **theoretical model** specification. To emphasize and clarify the difference, we augment our **simple linear regression** model nomenclature (as given in the 'The theoretical "normal distribution" regression model' link above) with the contrasting alternative notations and terminology: 

- **Fitted intercept** $\hat \beta_0$ and **slope** $\hat \beta_1$ ***coefficients*** are given "hats" to distinguish that they **estimate** (based on observed **sample data**), respectively, the **intercept** $\beta_0$ and **slope** $\beta_1$ ***parameters***

    
- **Fitted (predicted) values** $\hat y_i$ are made lower case and also given "hats" to distinguish them from the (upper case) **theoretical random variable** $Y_i$ implied by the **theoretical simple linear regression model** 
    - Technically, the **error** $\epsilon_i$ is the **random variable** specified by the **simple linear regression model** specification, and/but this implies the **random variable** nature of $Y_i$

    
- The **residuals** $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i = y_i - \hat \beta_0 + \hat \beta_1 x_i $ are also distinct and definitively contrast with the **errors** (or **noises**) $\epsilon_i$
    - The **residuals** $\text{e}_i = \hat \epsilon_i$ are actually available, while the **error** (or **noises**) $\epsilon_i$ are just a theoretical concept
    - The **residuals** $\text{e}_i = \hat \epsilon_i$ nonetheless are therefore used to diagnostically assess the theoretical modeling assumptions of the  **errors** $\epsilon_i$, such as the **normality**, **homoskedasticity**, and **linear form** assumptions; and, <u>while this is a not necessarily beyond the scope of STA130 and would certainly be a relevant consideration for the course project, this will not be addressed here at this time</u>

</details> 
      
<details class="details-example"><summary><u><span style="color:blue">Hypothesis testing for Simple Linear Regression</span></u></summary>
    
### Hypothesis testing for Simple Linear Regression
    
We can use **Simple Linear Regression** to test
    
$$\Large 
\begin{align}
H_0: {}& \beta_1=0 \quad \text{ (there is no linear assocation between $Y_i$ and $x_i$ "on average")}\\
H_A: {}& H_0 \text{ is false}
\end{align}$$

That is, we can assess the evidence of a linear association in the data based on a **null hypothesis** that the **slope** (the "on average" change in $Y_i$ per "single unit" change in $x_i$) is zero

> We are essentially never (or only very rarely in very special circumstances) interested in a **null hypothesis** concerning the **intercept** $\beta_0$ (as opposed to $\beta_1$) because the assumption that $\beta_0$ is zero essentially never (or only very rarely in very special circumstances) has any meaning, whereas the assumption that $\beta_1$ is zero has the very practically useful interpretation of "no linear association" which allows us to evaluate the  evidence of a linear association based on observed data

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

<details class="details-example"><summary><u><span style="color:blue">Indicator variables and (two-sample) group comparision</span></u></summary>

### Indicator variables and (two-sample) group comparision
    
A **Simple Linear Regression** can specify a model of two normally distributed populations (say, "A" and "B") with different **means** (but a common **variance**)
    
$$\Large Y_i = \beta_0 + \beta_1 1_{[x_i=\textrm{"B"}]}(x_i) + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

and where the **indicator variable** notation $1_{[x_i=\textrm{"B"}]}(x_i)$ equates to $1$ whenever $x_i$ takes on the value "B" (as opposed to population "A" in this example), and $0$ otherwise. Thus, the mean of population "A" is $\mu_A = \beta_0$ while the mean of population "B" is $\mu_A = \beta_0+\beta_1$. The $\beta_1$ coefficient therefore now represents a so-called **contrast** which captures the difference between the two groups (of the "A" and "B" populations in this example). 
    
A two-group comparison for **independent** (as opposed to **paired**) was also considered from a simulation perspective.

- For **simulation-based hypothesis testing** we considered a **permutation test**, which approximated **the sampling distribution for an average difference statistic under the null hypothesis assumption of "no difference between treatment groups"**. In **permutation testing** this is done by recalculating the average difference between "samples" created by shuffling (or randomly permuting) the label assignments. Shuffling imposes the constraint that the created "samples" have the same size as the original two groups. It also fully conforms with the a null hypothesis assumption of "no difference between treatment groups" since if this is true the labels don't matter. 

- A **bootstrapped confidence interval** for the **difference in population means** can also be created by repeatedly resampling each of the samples (separately within each sample) to build up the **bootstrapped sampling distribution of the average difference statistics**.
    
Students may review the code below in their personal time at their convenience should they wish to review an explicit implementation of the above two-sample analysis specifications.

```python
# Slight editing of https://chatgpt.com/share/6b1bb97b-80b9-4a50-9323-56a51060d6c8
import pandas as pd
import numpy as np

np.random.seed(42)  # Seed for reproducibility
# Sample data... more work needed to "parameterize" this code to explore different sample sizes, etc.
data = {'group': ['A'] * 10 + ['B'] * 10,  # 10 observations for each group... change loc to change group offset
        'value': np.random.normal(loc=50, scale=5, size=10).tolist() + np.random.normal(loc=55, scale=5, size=10).tolist()}
df = pd.DataFrame(data)

# Observed difference in means between group A and B
observed_diff = df[df['group'] == 'A']['value'].mean() - df[df['group'] == 'B']['value'].mean()

n_permutations = 1000  # Number of permutations
permutation_diffs = np.zeros(1000)  # Store the permutation results

# Permutation test
for i in range(n_permutations):
    # Shuffle group labels randomly; Calculate the difference in means with shuffled labels
    shuffled = df['group'].sample(frac=1, replace=False).reset_index(drop=True)
    shuffled_diff = df[shuffled == 'A']['value'].mean() - df[shuffled == 'B']['value'].mean()
    permutation_diffs[i] = shuffled_diff

# Calculate p-value: the proportion of permutations with an 
# absolute difference greater than or equal to the observed difference
p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
print(f"Observed difference in means: {observed_diff:.3f}")
print(f"P-value from permutation test: {p_value:.3f}")
```
    
```python  
import plotly.express as px
import plotly.graph_objects as go

# Plotting the distribution of permutation differences
fig = px.histogram(permutation_diffs, nbins=30, title='Permutation Test: Sampling Distribution of the Difference in Means')
fig.update_traces(marker=dict(color="lightblue"), opacity=0.75)

# Add a vertical line for the observed difference
fig.add_vline(x=observed_diff, line_dash="dash", line_color="red", annotation_text="Observed Diff", annotation_position="top right")

# Highlight the area for p-value computation (two-tailed test)
fig.add_trace(go.Scatter(
    x=permutation_diffs[permutation_diffs >= np.abs(observed_diff)],
    y=[0] * len(permutation_diffs[permutation_diffs >= np.abs(observed_diff)]),
    mode='markers', marker=dict(color='red'), name=f'Extreme Values (p-value = {p_value:.3f})'))

fig.add_trace(go.Scatter(
    x=permutation_diffs[permutation_diffs <= -np.abs(observed_diff)],
    y=[0] * len(permutation_diffs[permutation_diffs <= -np.abs(observed_diff)]),
    mode='markers', marker=dict(color='red'), name='Extreme Values'))

# Update layout to make the plot more informative
fig.update_layout(xaxis_title='Difference in Means', yaxis_title='Frequency',
                  showlegend=True, legend=dict(title=None))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```

```python
n_bootstrap = 1000  # Number of bootstrap samples
bootstrap_diffs = np.zeros(n_bootstrap)  # Pre-allocate a numpy array to store the bootstrap differences

# Bootstrapping to create the sampling distribution of the difference in means
for i in range(n_bootstrap):
    # Resample with replacement for both groups A and B
    boot_A = df[df['group'] == 'A']['value'].sample(frac=1, replace=True).reset_index(drop=True)
    boot_B = df[df['group'] == 'B']['value'].sample(frac=1, replace=True).reset_index(drop=True)
    
    # Calculate the difference in means for the bootstrap sample and store in pre-allocated array
    bootstrap_diffs[i] = boot_A.mean() - boot_B.mean()

# Calculate the 95% confidence interval (2.5th and 97.5th percentiles)
lower_bound = np.percentile(bootstrap_diffs, 2.5)
upper_bound = np.percentile(bootstrap_diffs, 97.5)

# Print observed difference and confidence interval
print(f"Observed difference in means: {observed_diff:.3f}")
print(f"95% Confidence Interval: [{lower_bound:.3f}, {upper_bound:.3f}]")

# Plotting the bootstrap distribution
fig = px.histogram(bootstrap_diffs, nbins=30, title='Bootstrapped Sampling Distribution of the Difference in Means')
fig.update_traces(marker=dict(color="lightblue"), opacity=0.75)

# Add lines for the observed difference and confidence interval
fig.add_vline(x=observed_diff, line_dash="dash", line_color="red", annotation_text="Observed Diff", annotation_position="top right")
fig.add_vline(x=lower_bound, line_dash="dot", line_color="green", annotation_text="Lower 95% CI", annotation_position="bottom left")
fig.add_vline(x=upper_bound, line_dash="dot", line_color="green", annotation_text="Upper 95% CI", annotation_position="bottom right")

# Update layout to make the plot more informative
fig.update_layout(xaxis_title='Difference in Means', yaxis_title='Frequency', showlegend=False)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```    
    
</details>

## 💬 🗣️ Communication Activity #1 [35 minutes]

Your TA will group you into <u>**SIX**</u> groups, potentially in groups that correspond to **Course Project teams**<br><br>

#### *[25 of the 35 minutes]* Consider data of the form given below, with two columns of **continuous** (numeric decimal) data and one column of categorical data (here with at least three different levels), and answer the following questions within your groups

|i | study_hours |class_section |exam_score |
|:---|:----|:----|:----|
|0 |10.9934280 |A |86.530831 |
|1 |9.7234711 |A |84.632809| 
|2 |11.2953770 |B |87.036506 |
|3 |13.0460600 |C |97.952866 |
|4 |9.5316930 |C |79.749848|

#### *[10 of the 35 minutes]* TAs should interrupt the group discussion every few finutes to see if all groups think they have the answer (or want a little more time), and should facilitate group progress by having one of the groups (ideally not the same group over and over) explain the answer to all the other groups (to confirm everyone has the right answer, or help move groups that are stuck along)


1. How could you use ONLY TWO **binary indicator variables** in combination to represent the ALL THREE levels (A, B, and C) in the example above?<br><br>

    1. Hint 1: If $x_i$ is the `class_section` of **observation** $i$,  what do we know if $1_{[x_i=\textrm{"B"}]}(x_i)$ takes on the value of $1$ (rather than $0$) 
    2. Hint 2: If $x_i$ is the `class_section` of **observation** $i$,  what do we know if $1_{[x_i=\textrm{"C"}]}(x_i)$ takes on the value of $1$ (rather than $0$) 
    3. Hint 3: Using THREE **binary indicator variables** would be unnecessarily redundant for encoding the information about which of three groups you were in... can you see why? Suppose both of the above are $0$... what does this situation indicate?<br><br> 
    
2. What are the **means** of the different `class_section` groups in terms of the parameters of the following model specification?<br><br>
   $$Y_i = \beta_0 + 1_{[x_i=\textrm{"B"}]}(x_i)\beta_1 + 1_{[x_i=\textrm{"C"}]}(x_i)\beta_2 + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$<br>

3. What is the nature of the data generated under the following model specification if $Y_i$ is the `exam_score` of **observation** $i$, $z_i$ is the value of `study_hours` for **observation** $i$, and $x_i$ is as described above?<br><br>
   $$Y_i = \beta_0 + 1_{[x_i=\textrm{"B"}]}(x_i)\beta_1 + 1_{[x_i=\textrm{"C"}]}(x_i)\beta_2 + \beta_3 z_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$<br>

    1. Hint: the model specification below would be described as "a stright line relationship between the `exam_score` variable and `study_hours` variable with observed data noisily distributed around the line". So what changes if the indicator variables are included?<br>
    
    $$Y_i = \beta_0 + \beta_3 z_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$<br>

4. What is the practical interpretation of how `exam_score` changes relative to `class_section` according to the model specification of the previous question if $\beta_1$ and $\beta_2$ are not $0$?<br><br>  

    1. Hint: what is the meaning if $\beta_1$ and $\beta_2$ are actually $0$?<br><br>  
    
5. What is the practical interpretation of the behavior of the relationship between `exam_score` and `study_hours` within different `class_section` groups according to the model specification of the previous question?<br><br> 

    1. Hint: Does the nature of the relationship really change? Does the model specification prescribe that "the change in the outcome variable on average per unit change in the predictor" should differ across different `class_section` groups?<br><br>  

6. Is there a different kind of behavior that could be seen for the relationship between `exam_score` and `study_hours` between different `class_section` groups that might be different than what's prescribed by the model specification of the previous question?<br><br>  

    1. Hint 1: what is the meaning of the following model specification?<br>
    
    $$Y_i = \beta_0 + \beta_3 z_i + 1_{[x_i=\textrm{"B"}]}(x_i)\beta_1 + \beta_4 z_i \times 1_{[x_i=\textrm{"B"}]}(x_i) + 1_{[x_i=\textrm{"C"}]}(x_i)\beta_2 + \beta_5 z_i \times 1_{[x_i=\textrm{"C"}]}(x_i) + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$<br>

    1. Hint 2: this could also be re-expressed as...<br>
    
    $$Y_i = \left(\beta_0 +  1_{[x_i=\textrm{"B"}]}(x_i)\beta_1 + 1_{[x_i=\textrm{"C"}]}(x_i)\beta_2\right) + \left(\beta_3 + \beta_4 1_{[x_i=\textrm{"B"}]}(x_i) + \beta_5 \right) \times z_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$


##  🚧 🏗️ Demo (combining different kinds of predictor variables) [40 minutes]

### Demo statsmodels multiple linear regression "formula" specifications [16 of the 40 minutes]

- Get a dataset and briefly show the `statsmodels` "formula" implementions (https://www.statsmodels.org/dev/example_formulas.html) of the specifications below as you progress through introducing and discussing them; so, 
    - introduce, explain, discuss, and then `statsmodels` "formula" demo them quickly 
    - (and use a ChatBot to figure things out whenever needed)
    
### Explain notation, terminology, and meaning [24 of the 40 minutes]

- **_There won't be time to figure out the visualizations on the fly so TAs won't show these using `plotly` or `python` (unless they've pre-populated such demos); BUT, the TAs might COULD MAYBE just draw the illustrative pictures on the board?_**



```python
# Hopefully you're TA picks an interesting data set to demo!
# Students *could* potentially help by suggesting an intersting one to try out, too...
```

1. Two (or more) **indicator predictor variables** corresponding to THREE (or more) groups based on so called "contrasts" (or "offsets" or differnces) from an (arbitrarily chosen) "baseline" group

   $$\large Y_i = \beta_{\textrm{A}} + 1_{[x_i=\textrm{"B"}]}(x_i)\beta_{\textrm{B-offset}} + 1_{[x_i=\textrm{"C"}]}(x_i)\beta_{\textrm{C-offset}} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$



```python
# Anyway your TA's gonna be live coding for ya here...
```

2. Both **indicator** and **continuous predictor variables** together in a "parallel lines" specification where the number of such lines increases by one for each **indicator predictor variables**

  $$\large Y_i = \beta_{\textrm{A}} + 1_{[x_i=\textrm{"B"}]}(x_i)\beta_{\textrm{B-offset}} + \beta_z z_{i} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

    - Within each of the groups induced by the **indicator variables(s)**, the coefficient $\beta_z$ is (of course) still interpreted as a slope, as "the average change in the outcome variable for a one-unit increase in the $z_{k}$ predictor variable" but there is now also an additional difference "on average" (of $\beta_{\textrm{B-offset}}$ between groups "A" and "B" in this example) which is what produces the "parallel lines" nature of this specification


```python
# here
```

3. Two (or more) **continuous predictor variables**, say $z_{1}$ and $z_{2}$ (with $z_{1i}$ and $z_{2i}$ representing observation $i$), specifying simultaneous linear relationships with the outcome variable occurring in concert

  $$\large Y_i = \beta_0 + \beta_1 z_{1i} + \beta_2 z_{2i} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

    - Coefficients $\beta_k$ are still interpreted as "the average change in the outcome variable for a one-unit increase in the $z_{k}$ predictor variable" but now we should be careful to additionally add the stipulation "with all other predictor variables held constant"

|Which actually in fact just happens to be a (hyper)plane model|
|-|
|![](https://statsandr.com/blog/multiple-linear-regression-made-simple/images/multiple-linear-regression-plane.png)|


```python
# here
```

4. **Interactions** (not to be confused with **indicator predictor variables**) between **continuous** and **indicator predictor variables** allowing the strength of the "linear relationship" between the **outcome variable** and the **continuous predictor variables** to be different within different groups, creating a "non-parallel lines" behavior

    $$\large Y_i = \beta_0 + \beta_z z_i + \beta_{\textrm{B-offset}} 1_{[x_i=\textrm{"B"}]}(x_i) + \beta_{\textrm{z-change-in-B}} z_i \times 1_{[x_i=\textrm{"B"}]}(x_i) + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$<br>
    $$\Large\textrm{OR}$$
    $$\large Y_i = \beta_0 + \beta_{\textrm{B-offset}} 1_{[x_i=\textrm{"B"}]}(x_i) + \left(\beta_z +\beta_{\textrm{z-change-in-B}} 1_{[x_i=\textrm{"B"}]}(x_i) \right) \times z_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

    - So the coefficient $\beta_z$ is (still) interpreted as "the average change in the outcome variable for a one-unit increase in the $z$ predictor variable within the baseline group" but now this "average change" is different in other groups, so (for this example), "the average change in the outcome variable for a one-unit increase in the $z$ predictor variable WITHIN GROUP 'B' is $\beta_z + \beta_{\textrm{z-change-in-B}}$"<br><br>
    - And there remains here as well an additional difference "on average" (of $\beta_{\textrm{B-offset}}$ between groups "A" and "B" in this example) allowing the "non-parallel lines" to have different vertical shifts vertically as well as have different slopes
    
> When using [`plotly.express.scatter`](https://plotly.com/python/linear-fits/#fitting-multiple-lines-and-retrieving-the-model-parameters) with `x`, `y`, AND `color` AND `trendline="ols"` you automatically get the "interaction" visualization. 


```python
# here
```


5. **Interactions** between **continuous predictor variables** creating a "synergistic" behavior between the **predictor variables** by which "the average change in the outcome variable for a one-unit increase in one $z_k$ predictor variable depends on the value of another $z_{k'}$ predictor variable (assuming this other $z_{k'}$ predictor variable is held constant)"<br><br>
  $$\large Y_i = \beta_0 + \beta_1 z_{1i} + \beta_2 z_{2i} + \beta_{12} z_{1i} \times z_{2i} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$
 
    - For this example, "the average change in the outcome variable for a one-unit increase in the $z_{1}$  predictor variable (assuming $z_{2}$ predictor variable is held constant) is $\beta_1 + \beta_{12} z_{2}$" which can be seen to "synergistically" depend on the (fixed) value of $z_{2}$
    - And conversely, "the average change in the outcome variable for a one-unit increase in the $z_{2}$  predictor variable (assuming $z_{1}$ predictor variable is held constant) is $\beta_1 + \beta_{12} z_{1}$" which can be seen to "synergistically" depend on the (fixed) value of $z_{1}$<br><br>
    - But do you see why?

> A quite interesting **continuous predictor variables** "interaction" is that of *a variable with itsef*, such as $z_{1}^2$ and $z_{2}^2$.  
>
> ![](https://miro.medium.com/v2/resize:fit:1200/1*_UaCxPswsCxkj9JzYXCiWg.png)
> ![](https://www.researchgate.net/publication/316420362/figure/fig4/AS:486432612589585@1492985980566/First-eight-terms-of-a-2-D-Chebyshev-polynomial.png)


```python
# and here, or maybe add and use more cells or maybe not new cells are needed
```

    
6. Reading these **multiple linear regression** model specifications just requires understanding straight up simple math... the equations mean exactly what they say and should be interpreted as such; however, that said...

    - One thing to be sure to keep in mind when you're interpreting the equations, though, is that you interpret them with respect to one variable at a time, under the assumption "with all other predictor variables held constant"
    - This "with all other predictor variables held constant" assumption is "theoretical" in the sense that in observed data you probably can't "make a one-unit change to just one predictor variable while simultanesouly holding all other predictor variables constant"
    - But, from the perspective of the mathematical equation, you can indeed "theoretically" consider this possibility, and this is indeed the perspective that is used to interpret the relationship of each single predictor variable with the outcome on a one-by-one, case-by-case basis



```python
# ANYWAY, hopefully you're TA picks an interesting data set to demo!
# Students *could* potentially help by suggesting an intersting one to try out, too...
```

## 💬 🗣️ Communication Activity #2<br>[in the (final 10 minutes of) time remaining<br>(or more or less depending on the time needed for the demo above...)]

Return to you <u>**SIX**</u> groups of the first **Communication Activity**, ideally those corresponding to the **Course Project teams** of the TUT

1. Discuss **Individual Project Proposals** that might relate to **multiple linear regression** and work to write out a model specification that you could analyze for the **Canadian Social Connection Survey** dataset variables for you **Course Project**
2. Explore possible ideas for **outcome variables** (and related relavent **predictor variables**) that you might examine for your **Course Project**
3. Consider which of the kinds of **predictor variables** could (and should) be used for your specification, and if you can imagine other **multiple linear regression** model specifications that might take advantage of the variety of types of **predictor variables** that can be leveragecd in the **multiple linear regression** context. 

> 1. **For your course project**, you're probably working with categorical variables: look into **logistic regression** for now. That is, *if you're understanding* **multiple linear regression** then you can do **logistic regression**.  And next week we'll consider **classification decision trees** which will open up *even further* options for you if you're working with categorical variables. 
> 2. **Indicator variables** are likely going to be **VERY** important for you as well, regarding the **predictor variables** that you're interested in.
> 3. A simpler way to start with considering potential associations between **categorical variables** is to look at **bar plots** of one **categorical variable** across different levels over *another* **categorical variable**. And you can additionally either (a) look into using the **Fisher exact Test**, or (b) think about how you could design **YOUR OWN** hypothesis testing using **simulation** to creating a **sampling distribution under a null hypothesis assumption** in order to produce a **p-value**.
>
>
> 4. <u>**Or as I ALWAYS SAY, "better yet", figure how how to create a (perhaps "double") _bootstrapped confidence interval_**.</u>


```python

```
