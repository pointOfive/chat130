# Course Textbook: Week 04 Bootstrapping and Estimation

**TUT/HW Topics**

1. [simulation](week-04-Bootstrapping#Simulation) (with `for` loops and `from scipy import stats`)
2. [sampling distribution of the sample mean](week-04-Bootstrapping#VariabilityUncertainty-of-the-Sample-Mean)
3. [standard deviation versus standard error](week-04-Bootstrapping#Standard-Deviation-versus-Standard-Error)
4. [how n drives standard error](week-04-Bootstrapping#How-n-drives-Standard-Error)

**LEC Extensions**

1. [Independent Sampling](week-04-Bootstrapping#Independent-Samples) functions like `df.sample([n=n/frac=1], replace=False)`
    1. [Are Sampling Distributions Skewed?](week-04-Bootstrapping#Are-Sampling-Distributions-Skewed)
    2. [Bootstrapping](week-04-Bootstrapping#bootstrapping)
    3. [Not Bootstrapping](week-04-Bootstrapping#not-bootstrapping)

**LEC New Topics**

1. [Confidence Intervals](week-04-Bootstrapping#Confidence-Intervals)
2. [Bootstrapped Confidence Intervals](week-04-Bootstrapping#Bootstrapped-Confidence-Intervals)
3. ["Double" _for_ loops](week-04-Bootstrapping#double-for-loops)
    1. [Proving Bootstrapped Confidence Intervals using Simulation](week-04-Bootstrapping#Proving-Bootstrapping)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as the **Central Limit Theorem (CLT)**, **Law of Large Numbers (LLN)**, and theoretical "x-bar plus/minus about 2 standard errors" confidence intervals (based on the so-called "pivot" form)
4. ... the alternative sampling function `np.random.choice(list_of_options, p, replace=True)` which will be introduced for different purposes later


## TUT/HW Topics

### Simulation

In Statistics, **simulation** refers to the exercise of repeating a sampling process a large number of times in order to understand some aspect of the behavior of the process. The following code visualizes a **normal distribution**, and the subsequent code shows the simulation based visualization of the **normal distribution**.

**Population (modelled by a statistical distribution)**

```python
# plotly.express provides visualization, scipy.stats provides distributions
# numpy provides numerical support, and pandas organizes data
import plotly.express as px
from scipy import stats
import numpy as np
import pandas as pd

# mean (mu_μ) and standard deviation (sigma_σ) parameters 
# determine the location and spread of a normal distribution 
mean_mu_μ, std_sigma_σ = 1, 0.33
normal_distribution_object = stats.norm(loc= mean_mu_μ, scale=std_sigma_σ)

# `np.linspace` creates an array of values evenly spaced within a range 
# over which we will visualize the distribution
support = np.linspace(-1, 3, 100)

# probability density functions (PDFs) show the long term relative frequencies 
# that will be seen when sampling from a population defined by a distribution
pdf_df = pd.DataFrame({'x': support, 'density': normal_distribution_object.pdf(support)})
fig = px.line(pdf_df, x='x', y='density')
fig.show()
```

**Simulation of a population (using a distribution)**

```python
import plotly.graph_objects as go

n = 10000 # sample size
normal_distribution_sample = np.zeros(n)

for i in range(n):
    normal_distribution_sample[i] = normal_distribution_object.rvs(size=1)[0]

# or, more simply
normal_distribution_sample = normal_distribution_object.rvs(size=n) 

# Add histogram of the sample
fig = go.Figure()
fig.add_trace(go.Histogram(x=normal_distribution_sample, 
                           histnorm='probability density', 
                           nbinsx=30, name='Sample Histogram', opacity=0.6))

# Add the population
fig.add_trace(go.Scatter(x=pdf_df['x'], y=pdf_df['density'], mode='lines', name='Population'))

# Update layout
fig.update_layout(title='Normal Distribution Population and Sample Histogram',
                  xaxis_title='Value', yaxis_title='Density')
```

The sampling process being simulated is "sampling a single observation from a Normal distribution".  The number of simulations is `n` and the `for` loop repeats the simulation the sampling process a large number of times (as determined by `n`) in order to understand the behavior of the process of "sampling a single observation from a Normal distribution".  Of course, as you may have already recognized, the `.rvs(size=n)` method let's us do this without a `for` loop by instead just using `normal_distribution_object.rvs(size=n)`.

- Consider experimenting with different values of `n` to explore how the "size" of the simulation determines how well clearly the behavior of the process is understood.

**Another simulation**

Consider the following alteration on the previous simulation.

```python
import plotly.express as px

number_of_simulations = 1000
n = 100 # sample size
normal_distribution_sample_means = np.zeros(number_of_simulations)

for i in range(number_of_simulations):
    normal_distribution_sample_means[i] = normal_distribution_object.rvs(size=n).mean()

df = pd.DataFrame({'Sample Mean': normal_distribution_sample_means})
fig = px.histogram(df, x='Sample Mean', nbins=30, title='Histogram of Sample Means', 
                   labels={'Sample Mean': 'Sample Mean', 'count': 'Frequency'})
fig.show()
```

Here are some questions to answer to make sure you understand what this simulation is doing (compared to the previous simulation):

1. Why is `number_of_simulations` introduced and why is it different than `n`?
2. What is the effect of appending the `.mean()` **method** onto `normal_distribution_object.rvs(size=n)` inside the simulation `for` loop?
3. Are the histograms from the two simulations equivalent? If not, why are the differences and what is causing them? 

As this current simulation example shows, simulation can explore much more interesting downstream behaviours of sampling processes besides just the behaviour or "sampling a single observation from a Normal distribution" (or some other distribution if you were to sample from something other than a Normal distribution).


### Variability/Uncertainty of the Sample Mean

**Or, "x-bar" is itself a "sample" of size one**

The previously code simulated the so-called **sampling distribution of the mean**, which captures the **variability/uncertainty** of the sample mean. So, just as data points are sampled from some **distribution**, a **statistic** (which is just some mathematical function) of samples can as well be viewed on a higher level as themselves (that is, the **statistic** itself) as being "a sample" from the "distribution of the statistic". 

The concept that a **statistic** (such as the sample mean $\bar x$) would itself be drawn from its own distribution is very intuitive when viewed from the perspective of **simulation** (as demonstrated in the final example given in the previous section). 

> The theory related to the **sampling distribution of the mean** is known as **the Central Limit Theory (CLT)** and relates as well to the so called **Law of Large Numbers (LLN)**; but, these topics are beyond the scope of STA130 -- and not the focus of STA130 -- *so for STA130 it's very important that you* _**approach understanding sampling distributions through a simulation-based perspective**_.
>
> On a related note, a theoretical form that you will see related to the **standard error of the mean** $s_{\bar x}$ is 
> 
> $$ \bar x \pm 1.96 s_{\bar x}$$ 
> 
> which produces a 95% confidence interval based on the **Central Limit Theorem (CLT)** using the so-called "pivot" trick. 
> However, again, since STA130 is focussed on understanding the **variability/uncertainty** of the sample mean from a **simulation**-based perspective, _**this methodological framework is beyond the scope and not the focus of STA130.**_


### Standard Deviation versus Standard Error

A simple summary of the difference of these two distinct but related concepts is that 
1. "standard deviation" refers to the **standard deviation** of a **sample** or a **population**; whereas, 
2. "standard error" (that is, the **standard error of the mean**) refers to the "standard deviation of the **sampling distribution** of the sample mean"
 
The **sample standard deviation** is (as you know) defined as 

$$s = \sqrt{\frac{1}{n-1}\sum_{n=1}^n (x_i-\bar x)^2}$$

and (as you can see with some inspection and consideration) it measures how spread out the data is by capturing something like "how far on average" individual data points are from the sample mean $\bar x$ of the data set (since it's the square root of essentially "the average squared difference from the sample mean). **Standard deviation** describes the variation (or dispersion) in a set of data points. A large standard deviation indicates that the data points are spread out over a wide range of values, while a small standard deviation indicates that they are clustered closely around the mean.

> Be careful to distinguish between the **sample standard deviation** statistic $s$ and the Normal distribution **standard deviation parameter** σ (which $s$ estimates): $s$ is the **sample** analog of the **population** characteristic concept σ.
>
> There are key differences between the **sample standard deviation** and the **standard error of the mean** that should be readily differentiating. First, the **standard deviation** applies to the entire **sample** or **population** (depending if we're talking about the **statistic** or the **parameter**), reflecting the variability (or dispersion) among individual data points; whereas, the **standard error of the mean** only applies to the **sample mean** and reflects the **variability/uncertainty** of the **sample mean** $\bar x$ as an **estimate** of the **population mean** μ (owing to the variability propagated into $\bar x$ due the inherent variability present in the "random sampling" process. Second, the **standard deviation** does not depend on the **sample size** since is not a determining factor in how variability there is between individual data points (since it is the nature of the population which determines this); whereas, **the standard error of the mean** *decreases as the sample size increases*, indicating more "precise" estimates with larger samples.

### How n drives Standard Error

In contrast to the **standard deviation** (of a **sample** or a **population**), the **standard error of the mean** is (due to the **Law of Large Numbers (LLN)**) defined as 

$$s_{\bar x} = \frac{s}{\sqrt{n}}$$ 

and characterizes the potential **variability/uncertainty** the **sample mean** of the data $\bar x$ relative to the true **population mean** (usually referred to as μ if the sample was drawn from a **normal distribution**). The **standard error of the mean** therefore captures the "precision" of the **sample mean** as an **estimate** of the **population mean**. 

The above simple definition that the **standard error of the mean** shows that $s_{\bar x} = \frac{s}{\sqrt{n}}$ is easily calculated by dividing the **standard deviation** by the *square root* of the **sample size** $n$. But this theoretical value of the **standard error** is only true if the $n$ data points are from an **independent sample**. So, if the **sample** is made up of **independent** data points, then the larger the sample size $n$, the smaller **standard error** (so the smaller the "standard deviation of the sampling distribution of the sample mean" is), and thus the more "precisely" the **sample mean** $\bar x$ **estimates** the **population mean** μ.

> The correct way to understand what "precision" indicates here is to consider that a **sample mean** $\bar x$ is a "sample" from the **sampling distribution of the mean** which is centered on the corresponding **population mean** μ that $\bar x$ **estimates**, assuming the usual assumptions -- like **independence** -- underlying a statistical analysis are true! So as the **standard error** gets smaller, then the **variability/uncertainty** of the **sample mean** around the **population mean** μ it **estimates** is being reduced (hence making it a better **estimate** of μ). 


## LEC Extensions


### Independent Samples

We first motivated **independence** in **samples** by appealing to the fact that we want our **sample** to **represent** our **population**, so samples should be **independent** and not "clustered" or "biased" in any strange way.
But based on the previous section, we can now see the downstream implications of our preference for **independence**; namely, we desire the **standard error** to predictably and reliably decrease as we increase the **sample size** $n$.

> The notion that we need to rely on **independence** to do our analysis is so ubiquitous in Statistics that we might sometimes even forget to mention it.  It can also get tedious to say it so much, so as you continue into more advanced statistical courses you'll increasingly encounter a statisticians most favourite shorthand notation, "i.i.d", which simply means "identically and **independently** distributed** (when referring to the assumptions made about a sample of data as the basis of some subsequent methodological analysis).  

We've seen many appearances of **independent samples** via the `.rvs(size=n)` **method**. For example, we first saw it in the "roll five dice" as in the game "Yahtzee" using `stats.multinomial(n=1, p = [1/6] * 6).rvs(size=5)`. And here we just saw the example of `normal_distribution_object.rvs(size=n)`.  When we first introduced `stats.multinomial` we also showed the alternative interface of `np.random.choice(...)` which also produced **independent samples** (as it obviously must as an alternative to `stats.multinomial(...).rvs(size=n)`).

It probably won't surprise you to learn that `pandas` has a similar functionality to `np.random.choice` (especially since as you may recall `pandas` just extends `numpy` functionality to introduce the abstraction of the the `pd.DataFrame` object, and the myriad of associated functionalities implied therein). That functionality is the `.sample()` **method** which can be used as demonstrated below to randomly select rows of a `pd.DataFrame` object (or specific column(s) as in `df['col'].sample(...)` and `df[['col','col2']].sample(...)`.  As you consider this functionality, ask yourself which which create a **independent sample**... and, hint, only ONE of them produces an **independent sample**... 

```python
# randomly sample `number_of_rows` without duplication
df.sample(n=number_of_rows, replace=False)
# or as a fraction of rows `df.sample(frac=fraction, replace=False)`

# randomly sample `number_of_rows` with possible duplication
df.sample(n=df.dim[0], replace=True)
# where `df.dim[0]` is the number of rows of the data frame
# for some reason... known as FORESHADOWING ;)

# WARNING
# - `np.random.choice(...)` has default `replace=True` [if you don't specify it otherwise!]
#   which makes sense since this wants to model `stats.multinoial`
#   where we might draw the same choices from our options over and over
# - `df.sample(...)` in contrast uses the default `replace=False` [if you don't specify it otherwise!]
#   which ALSO makes sense since it's probably very natural to think of
#   sampling rows without duplication in the context of a pd.DataFrame
#
# However...
#
# Since we LOVE samples to be **independent** so much you might guess (from the FORESHADOWING) that
# we are in the end going to be VERY interested in `df.sample(n=df.dim[0], replace=True)`...

```


### Are Sampling Distributions Skewed?

The **Week 04 TUT Demo** introduced code analogous to what's given below, but for a **normal distribution** rather than a **gamma distribution** which is **right-skewed**. The code below, then, demonstrates the **sampling distribution** of the **sample mean** when the **sample** is drawn **independently** from a **population** specified by a **gamma distribution** (as opposed to a **normal distribution**). Even though the **population** the the **sample** is being drawn from is **right skewed**, as you can see, for "large enough" sample sizes $n$, the **sampling distribution** of the **sample mean** still basically looks to have the **symmetric** character as when the samples were already drawn from a **symmetric** (**normal**) **distribution**.  While one might think the **symmetric** or **skewed** character of the **population** being **sampled** would makes its way into influencing the "shape" of the **sampling distribution** of the **sample mean**, it turns out that it doesn't really because such affects (which are indeed present in "very small" **samples**) are quickly overcome by only moderate increases in the size of the **sample**. 

**STEP 1: Population (and a first example sample)**

```python
# population
population_parameter_alpha_α = 2
population_parameter_theta_θ = 4
gamma_distribution = stats.gamma(a=population_parameter_alpha_α, 
                                 scale=population_parameter_theta_θ)
# sample
n = 100  # adjust and experiment with this
# np.random.seed(130)  # and hopefully it's easy to figure out what this line does
x = gamma_distribution.rvs(size=n) # "x" is a sample
# print(x)

# mean
print("The sample mean for the current sample is", x.mean()) 
# the sample mean "x-bar" is a (sample) "statistic" (not a "parameter")
# "x-bar" is the "average" of the numbers in a sample
```

**STEP 2: Sample (compared to the population)**

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
fig.show() 
```

**STEP 3: Sampling Distribution of the Sample Mean (Symmetric for "large enough" n)**

```python
number_of_simulations = 1000 # adjust and experiment with this
simulated_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
n = 100 # adjust and experiment with this

for i in range(number_of_simulations):
    # np.random.seed(130) # ?
    simulated_means[i] = gamma_distribution.rvs(size=n).mean()
    
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
fig.show()
```

**But now what do we do if don't know the population??** All of this business of sampling from **normal** or **gamma** or whatever `scipy.stats` distribution you imagine is just "purely academic" if we're being honest about it...

```python
# Let's load this data set to help us "get real" and answer this question (in the next "Bootstrapping" section)

df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/2e9bd5a67e09b14d01f616b00f7f7e0931515d24/data/2020/2020-07-07/coffee_ratings.csv")
df = df.rename(columns={'country_of_origin': 'origin', 'total_cup_points': 'points'})

df = df[df['points']>65] # ignore some very low scores
df = df[~df['origin'].isna()] # remove rows with unknown origin

df['origin'] = df['origin'].str.replace("?","'") # fix character encoding issue
df['origin_original'] = df.origin.copy().values # save original (corrected) names

keep = (df.origin_original=='Guatemala') | (df.origin_original=='Mexico')
fig = px.histogram(df[keep], x='points', facet_col='origin', facet_col_wrap=2, height=300)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```


### Bootstrapping

The phrase "pulling yourself up by your bootstraps" conjures a funny image. Apparently we are to soar into the atmosphere using nothing but our will and our the sheer application of our upper arm strength to the loops in our boots (in an "upward" fashion). But of course the phrase itself means to "get the job done using just what you have available".  

So, if we have a **sample** of data, but we don't really know the **population** it comes from (which of course would be the most usual state of affairs, assuming we're talking about the real world here), then all we have is the **sample** itself. So **bootstrapping** means "we're gonna get the job done" -- the job of figuring out what the **sampling distribution** of the **sample mean**, that is (or, honestly, any other **statistic** you're interested in -- the **sample mean** is not the only **statistic** there is and ALL **statistics** have **sampling distributions**, as must be obvious, if you think about it), "using just what you have available". Which in this case is JUST the **sample**. But how in the gosh-darned heck are we gonna do that if we ONLY have the **sample** and don't know the **population** it comes from? Because in ALL the examples we've seen so far that **simulate** the **sampling distribution** of the **sample mean**, they do so based on KNOWING what the **population** of the sample is... That was how we did that with the example using the **normal distribution** in the **Week 04 TUT Demo**, and that was again how we just did it again in the code example above but that time we instead just used the (**right-skewed**) **gamma distribution**. 

Well I'll tell ya. 

Here's the trick. Let's just *pretend* the **sample** IS the **population**. That might sound crazy, but hear me out.  If it's an **independent sample** and the **sample** is also "large enough", then the **sample** is going to be sufficiently **representative** of the **population** so as for it to be usable as a "stand in" in place of the **population**.  So in that case, we can get by with just *pretending* our **sample** is our **population**.  And that's **bootstrapping** folks. Clear as day. "We're gonna get the job done using just what you have available". And interestingly, it turns out that a "large enough" size of a sample for **bootstrapping** is often in fact not "very large" at all. I'm talking only five to ten samples might be enough, folks!

But that still only really gets us halfway there. Why does *pretending* the **sample** IS the **population** help us?

Well, if we have the "population", then we can **sample** from it.  But wait a minute. How do you sample from a sample? Well do you remember `df.sample(n=df.dim[0], replace=True)`?

Now the only questions are
- why `df.dim[0]`
- and why `replace=True`?

Let's see if you can answer this. Quiz yourself for a moment, and try to figure out why these are the right choices to allow us to "sample from the sample which we're pretending is the population so we're pretending we're sampling from the population when we actually sample from the sample"? The correct explanations of the answers to these questions are clarified in the next section down below, but if you're looking for a hint, consider the following code. And in particular, consider how it compares to the code above which created a **sampling distribution** for the **sample mean** of a **sample** from a **population** modelled by a **gamma distribution**. 

**STEP 1: "Population" (which is actually a Sample, and a first example sample from this "population")**

```python
country = 'Mexico' 

n_ = (df.origin==country).sum() # This is the correct choice for bootstrapping, but why?
replace_ = True  # This is the correct choice for bootstrapping, but why?

# We don't have a "poulation" any more... only the (original) "Sample"
x = df[df.origin==country].sample(n=n_, replace=replace_).points
# print(x)
print("The bootstrap sample mean for the current bootstrapped sample is", x.mean()) 
```

**STEP 2: "Sample" (compared to the "population" which is actually just the original Sample)**

```python
# what is this doing?

# np.random.seed(130)  # ?
x = df[df.origin==country].sample(n=n_, replace=replace_).points

dat = pd.DataFrame({'sample values': np.r_[df[df.origin==country].points.values, x],
                    'sample': np.r_[['Orginal Sample']*(df.origin==country).sum(),
                                    ['Bootstrap Sample']*n_]})             

fig = px.histogram(dat, x="sample values", color="sample", barmode="overlay")
fig.add_vline(x=x.mean(), line_dash="dot", annotation_text='Bootstrapped sample mean<br>'+str(x.mean()))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

**STEP 3: The BOOTSTRAPPED Sampling Distribution of the Sample Mean (is just like the Sampling Distribution of the Sample Mean)**

```python
# what is this doing?

number_of_simulations = 1000  # adjust and experiment with this
simulated_bootstrap_means = np.zeros(number_of_simulations)

# np.random.seed(130) # ?
n_ = 100 # adjust and experiment with this # but it should be 236  # why?
# for bootstrapping it should be (df.origin==country).sum()  # why?
# if it's not, it's not bootstrapping... it's something else...
# ...not sure what it would then be called... 

replace_ = True # ? ... what would happen if the bootstrap sample size n_ 
# was the same size as the origina sample size n, and this was `False`?

for i in range(number_of_simulations):
    simulated_bootstrap_means[i] = df[df.origin==country].sample(n=n_, replace=replace_).points.mean()

# The important stuff is up above: the stuff below is just for plotting things nicely
title1 = "The BOOTSTRAPPED sampling distribution of the sample mean"
title2 = str(number_of_simulations)+' means from '+str(number_of_simulations)+' simulated samples of size n = '+str(n_)
fig = px.histogram(pd.DataFrame({title2: simulated_bootstrap_means}), title=title1, x=title2,
                   histnorm='probability density')    

PRETEND_population_parameter_mu_μ_is_sample_mean = df[df.origin==country].points.mean()
PRETEND_population_parameter_sigma_σ_is_sample_std = df[df.origin==country].points.std()
support = PRETEND_population_parameter_mu_μ_is_sample_mean + \
          4*np.array([-1,1])*PRETEND_population_parameter_sigma_σ_is_sample_std/np.sqrt(n)
support = np.linspace(support[0], support[1], 100)
legend = 'The "sampling distribution<br>of the sample mean" is the<br>theoretical distribution of<br>"averages" of sample values'
fig.add_trace(go.Scatter(mode='lines', name=legend, x=support, 
                         y=stats.norm(loc=df[df.origin==country].points.mean(),
                                      scale=df[df.origin==country].points.std()/np.sqrt(n_)).pdf(support)))
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS    
```


### Not Bootstrapping

The code in the previous section can be used to demonstrate that the behavior we see from **bootstrapping** (by *pretending* the **sample** is the **population** and repeatedly **resampling bootstrap samples** from the **original actual sample**) works SO similarly to the (literally pretend "purely academic" and not at all real-world) situation where we act like we could really know the **population** that **sample** was drawn from (like a **normal** or **gamma distribution**), that we really can just *pretend* that we indeed do know the **population** (and its the **sample** we have). And this was exactly the premise promised by **bootstrapping** in the first place. But some open questions remain.

- Q: What makes repeatedly **resampling bootstrap samples** from the **original actual sample** analogous to repeatedly from a population, such as one represented by a **normal** or **gamma distribution**?
    - A: The correct answers to the following questions.
- Q: Should `n_` in `df['data'].sample(n=n_, replace=True)` be equal to `df.dim[0]` (or equivalently `len(df['data']`)?
    - A: Yes, `n` must be the length of our **actual observed sample** because we want to create the **bootstrapped sampling distribution** of our **statistic** of interest for the same sized **sample** as our **original actual sample**.  The **variability/uncertainty** of our **statistic** of interest relative to the size of our **original actual sample** is what we want to understand because that's what actually corresponds to the data we actually have. E.g., if we're making **inference** about the **population mean** μ using the **sample mean statistic** $\bar x$ then we want to know the **standard error** of $\bar x$ **for the sample size we have** (which means we want to know the **standard deviation** of the **bootstrapped sampling distribution** of the **sample mean** for the sample size we have). And this we can do because we can **simulate** the **bootstrapped sampling distribution** of the **sample mean** as in the code above so long as we follow the correct usage given by `df['data'].sample(n=n_, replace=True)`.
- Q: Should `replace=True` in `df['data'].sample(n=n_, replace=True)` really be `True`?
    - A: Yes, because we want to draw **independent** samples from our "population", which means whatever we've drawn for samples so far is not allowed to affect what we draw for samples next. If drawing a specific data point from a **population** means you can't draw it again in the future, then you don't have **independent samples**. And anyway, what would happen if we set `n=n_` correctly for **bootstrapping** but then used `replace=False` instead of `replace=True`? Do you see what this would produce? It would make every **bootstrap sampled** exactly equal to the **original actual sample**.

So all of this means if you're not setting `n_` to be the **actual original sample size** (`df.dim[0]` or equivalently `len(df['data']`) and `replace=True` in `df['data'].sample(n=n_, replace=True)` then you ain't **bootstrapping**, friend. **Bootstrapping is when  `n_` is the original sample size and `replace=True` and we repeatedly **resample bootstrap samples** from the **original actual sample** over and over to see the behavior of the **sample statistic** of interest relative to the amount of data we actually have. If `replace=False` then you ain't **bootstrapping**. If `n_` is not same as the **original sample size** then you ain't **bootstrapping**. If this is not yet making sense to you, then you need to consider carefully why we are only interested in understanding the **variability/uncertainty** of **sample statistic** for the **sample size** of the **actual original sample**.

> It is true that `n_` could be set to a different value than the **original sample size**, but doing so is once again a "purely academic" exercise. There's no point. It doesn't correspond to anything we care about in the real world. Same thing for using `replace=False`. Using this would only be another "purely academic" exercise. The only reason you'd experiment around with these different choices is if you were trying to get a sense of how things worked in a "purely academic" sense. And again, this would not correspond to anything in the real-world that we care about.  

**TLDR Summary (wrongly at the end because it's much better if you work to understand the actual details and rationale)**

**Bootstrapping** is a statistical approach to understanding **variability/uncertainty** of a **statistic**. It is based on **resampling with replacement** from a **sample** to create many **simulated bootstrap samples** (of the **original sample size**), and understanding the behavior of the **statistic** across these **bootstrapped samples**. **Bootstrapping** in this manner can be used to **simulate** the **sampling distribution** of a **statistic** (like the **sample mean**) which is what allows us to characterize **variability/uncertainty** of our **sample statistic estimators**. 

The steps in bootstrapping are as follows. 
1. Take an initial sample from the population.
2. **Resample with replacement** to create many **simulated bootstrapped samples** (of the same size as the **actual original sample**).
3. Calculate the **statistic** of interest (such as the **sample mean**) for each **bootstrapped sample**.
4. Create the **bootstrapped sampling distribution** of the **statistic** which indeed characterizes the **variability/uncertainty** of the **statistic** at the **sample size** of the **actual original sample**.
5. Use the **bootstrapped sampling distribution** to make statistical **inferences** about **population parameters**.


## LEC New Topics


### Confidence Intervals

A **statistic** from a **sample** will estimate** a corresponding **population parameter**. Do you have a couple examples of **statistics* and the corresponding **population parameter** that they **estimate**? 

The **bootstrapped sampling distribution** of a **statistic** is just another way (based on creating **simulations** using only the **original sample**) to understand the **sampling distribution** of the **statistic**. This means **bootstrapping** is just another way to understand the **variability/uncertainty** of the **statistic** compared to the **population parameter** it **estimates**. 

A **confidence interval** is a proposed range of values for the **population parameter** which we have determined (based on **independent sample**) according to a specific procedure that will have a certain chance (which we can choose called the **confidence level**) of successfully bounding  (or "capturing") the **actual true value of the population parameter**.  The most common choice for the **confidence level** is 95%, meaning that the chance that an **independent sample** will result in the construction of a **confidence interval** which "captures" the **actual true value of the population parameter** is 95%.  **Confidence intervals** either *do* or *do not* "capture" the **actual true value of the population parameter** once they are created, so (somewhat confusingly) we cannot correctly speak of the "probability" or "chance" that an explicit **confidence interval** has or has not "captured" the **actual true value of the population parameter**. Rather, the **confidence level**, such as a 95% **confidence level**, refers to the proportion of hypothetical **independent samples** for which the procedure of constructing a **confidence interval** will indeed result in the "capture" of the **actual true value of the population parameter**. 

So, the following are things we need to keep straight and clear in our minds. 
- The **confidence interval** is a proposed range of values for a **population parameter**. 
- The **confidence level** is the chance that a **confidence interval** construction procedure will "work" for an new **independent sample** 
    - where "work" means the construction procedure would indeed "capture" the the **actual true value of the population parameter** IF constructed
    - which can be thought of as rate that a **confidence interval** construction procedure will "work" in "the long run" over essentially infinitely many different **independent samples**
- There is not "probability" or "chance" that a **confidence interval** construction procedure has "worked" after it has been constructed
    - It either HAS or HAS NOT actually "captured" the **actual true value of the population parameter**
    - and unfortunately we don't know if it HAS or HAS NOT.


### Bootstrapped Confidence Intervals

So what is this fabled **confidence interval** construction procedure that we're referring to? It's so mysterious and undefined! 
Well, for the data we've been examining above, the following construction procedure creates a 95% **bootstrapped confidence interval**.

```python
np.quantile(simulated_bootstrap_means, [0.025, 0.975])  # 0.025 actually means the 2.5th percentile, which we write as 2.5%
                                                        # 0.975 actually means the 97.5th percentile, which we write as 97.5%
```

Well, this **confidence interval** construction procedure is now no longer undefined. But it sill does perhaps remain seemingly quite mysterious. Fortunately, it would certainly not be said to complicated. And it's really not that confusing, either, if you know what **percentiles** are. This just provides the 2.5% and 97.5% **percentiles** of the however many `number_of_simulations` there are of the `simulated_bootstrap_means`. So why does this have a 95% **confidence level**. Well, the difference of 97.5% - 2.5% = 95% probably has something to do with it. But there's really not much else to say about this.  Why so? Well, there's really only one way to see if this supposed 95% **bootstrapped confidence interval** construction procedure actually fulfills its promised 95% **confidence level**. 

And how could we determine this? **Simulation**, my friends. **Simulation**. Obviously. What else would it be in STA130? But to do what we need to do for our **simulation**, we're going to need to understand "double" `for` loops.

### "Double" `for` loops

Double `for` loops, also known as nested loops, involve placing one `for` loop inside another. They are useful when you need to perform repeated actions across multiple dimensions, like iterating over rows and columns in a 2D array or performing combinations of operations between two sets.

Here's an example:

```python
for i in range(3):       # Outer loop
    for j in range(2):   # Inner loop
        print(f"i={i}, j={j}")
# which will produce
# i=0, j=0
# i=0, j=1
# i=1, j=0
# i=1, j=1
# i=2, j=0
# i=2, j=1
```

1. **Outer Loop (`for i in range(3)`):** 
   - This loop runs 3 times (`i` takes values 0, 1, and 2).
   - For each iteration of `i`, the inner loop will execute fully.

2. **Inner Loop (`for j in range(2)`):** 
   - This loop runs 2 times (`j` takes values 0 and 1).
   - It repeats for each value of `i` from the outer loop.


#### Proving Bootstrapping


In **Statistics** we DO prove things with mathematics. It was only relatively recently that **Statistics Departments** (like the Department of Statistical Sciences here at UofT) rebelled and broke away from the **Mathematics Departments** saying, "sure, we do math too, but you math folks just aren't interested enough in what' we're working on -- where working on DATA -- and we're blowin' up and we gotta have our own department!". But hopefully you're getting the idea from this week that we often don't need to rely on mathematical proofs to understand what's happening with something.  Instead, we can just figure out by just *watching* how it works using **simulation**. So that's what we're going to do here to here to "prove" that **bootstrapped confidence intervals** such as `np.quantile(simulated_bootstrap_means, [0.025, 0.975])` ARE indeed what they claim to be.  Namely, a **confidence interval** construction procedure that promises their advertised **confidence level** (like 95% in the example of the last sentence). 

Here's the approach we'll take to complete our **simulation** "proof" using the "double" `for` loops introduced in the previous section. 

1. The **outer loop** will draw a **sample** from a KNOWN **population** (and it will do this `so_many_bootstrapped_confidence_intervals` times, defining the number of **outer loop iterations**)
    1. meaning, specifically, that we know the **parameters** of the **population** the **sample** is drawn from
    2. and before you complain, why yes, this is "purely academic"
    3. but obviously it is: we're doing a "proof" here "proving" that (and exactly how) the **bootstrapped confidence interval** construction procedure as a data analysis methodology "works" (so you can see EXACTLY what we are trying to say when we report a 95% **confidence interval**
    4. (and note that "works" above refers to the construction procedure as a data analysis methodology; whereas, previously we used the term "work" or "worked" to instead refer to the question of whether or not a created **confidence interval** did indeed "capture" the **actually true known parameter** of the **population** it is trying to "capture")  
2. The **inner loop** will create many **bootstrapped samples** for each of which we'll calculate the **sample statistic** of interest which **estimates** a corresponding **actually true known parameter** of the **population** the **sample** was drawn from
    1. and it will do this `so_many_bootstrap_samples` times, defining the number of **inner loop iterations**
    2. and it will collect all the computed **bootstrapped sample statistics** in `simulated_bootstrap_statistics_for_one_sample`
3. After the **inner loop** completes but before we start the next **iteration** of the **outer loop** we'll construct our **confidence interval** for our chosen **confidence level** using the now familiar procedure `np.quantile(simulated_bootstrap_statistics_for_one_sample, [lower_percentile, upper_percentile])` which will create a `upper_percentile-lower_percentile`% **confidence level confidence interval**
    1. which we will compare to the **actually true known parameter** of the **population** to see if this **confidence interval** indeed "worked" 
    2. by checking if the **confidence interval** did indeed "capture" the **actually true known parameter** of the **population** it is trying to "capture"
    3. and we'll keep a record of how many of the `so_many_bootstrapped_confidence_intervals` that we similarly construct (in the **inner loop** for each **iteration** of the **outer loop**) actually indeed "worked"
    4. which should be a probability very close to the advertised `upper_percentile-lower_percentile`% **confidence level**
    5. because if it is, then the **confidence interval** construction procedure as a data analysis methodology "works" by indeed delivering on its promise regarding the proportion of hypothetical **independent samples** for which the procedure of constructing a **confidence interval** will indeed result in the "capture" of the **actual true value** of the **population parameter**
    6. (and again be careful to not confuse "works" in 5 with "worked" in 3 since 5 refers to the data analysis methodology while 3 refers to whether or not a created **confidence interval** "works" which we only know in "purely academic" settings like our **simulation** demonstration here but which we would never know in the real world obviously because the whole point is we don't know the **population parameter** and we're trying to **estimate** it)

```python
import numpy as np
# np.random.seed(42)

# Parameters defining the population which we will be estimating
# Everything here is something that could be "experimented" with
population_mean_μ = 50
population_std_σ = 10
known_normal_distribution_just_suppose = stats.norm(loc=population_mean_μ, scale=population_std_σ)
# so we know population_mean_μ and population_std_σ in this "purely academic" exercise, obviously

# Details of the experiment [key specifications related to Step 1 above]
n = 300  # something that could be "experimented" with
so_many_bootstrapped_confidence_intervals = 1000  # what would increasing or decreasing this do?
all_my_samples_to_make_boostrap_confidence_intervals_for = \
    known_normal_distribution_just_suppose.rvs(size=(n, so_many_bootstrapped_confidence_intervals))  
    # rows are observations in a sample and the columns are all my samples
simulated_samples_df = pd.DataFrame(all_my_samples_to_make_boostrap_confidence_intervals_for)

# Details of bootstrapped confidence interval [key specifications related to Step 2 above]
so_many_bootstrap_samples = 500  # what would increasing or decreasing this do?
# `so_many_bootstrap_samples` is different than `so_many_bootstrapped_confidence_intervals`
# obviously, but what exactly is the difference between their simulation/experimental purpose?
confidence_level = 80  # something that could be "experimented" with
simulated_bootstrap_statistics_for_one_sample = np.zeros(so_many_bootstrap_samples)  # is this right? why?

# Initialize coverage counter [key specifications related to Step 3 above]
coverage_count = 0

# Simulation Experiment 

# loop over samples
for simulation_i in range(so_many_bootstrapped_confidence_intervals):

    sample_i = simulated_samples_df.iloc[:, simulation_i]

    # create bootstrap samples for current sample
    for bootstrap_sample_j in range(so_many_bootstrap_samples):

        bootstrap_sample = sample_i.sample(n=n, replace=True)
        simulated_bootstrap_statistics_for_one_sample[bootstrap_sample_j] = bootstrap_sample.mean()
        # Here this is the sample `.mean()` but could it be something else? Like, e.g., `.std()`? 
        # What about `np.quantile(bootstrap_sample, [some_percentile])`? What would that be estimating?

    # end of inner loop so now we're back into the outer loop
    
    # Calculate the 95% confidence interval for the bootstrap means
    confidence_interval = np.percentile(simulated_bootstrap_statistics_for_one_sample, 
                                        [(100-confidence_level)/2, 
                                         confidence_level + ((100-confidence_level)/2)])
                                      # Is the above "formula" defining the percentiles correct? Why?
    # `simulated_bootstrap_statistics_for_one_sample` will just be reused 
    # it gets overwritten when we do this again for the next sample
    
    # Check if the true population statistic falls within the constructed confidence interval
    # Did the procedure "work" this time?  Is `population_mean_μ` between the lower and upper ends of the range?
    if (confidence_interval[0] <= population_mean_μ) and (population_mean_μ <= confidence_interval[1]):
        coverage_count += 1

# Simulation "Proof"
coverage_rate = (coverage_count / so_many_bootstrapped_confidence_intervals) * 100
print(f"Coverage rate: {coverage_rate}%")
print("Is it near the intended ", confidence_level, '% confidence level? If so, it "works."', sep="")
print("The bootstrapped confidence interval construction produced as a data analysis methodology works as advertised")
```

The code as given above uses an 80% **confidence level**, constructs a **confidence interval** for the **sample mean** which **estimates** the **population mean** based on a **sample size** of $n = 300$. It **simulates** `so_many_bootstrapped_confidence_intervals=1000` **samples** from a **population** characterized by an **actual known true mean**, and for each of the it creates `so_many_bootstrap_samples=500` and calculates **bootstrapped sample means** for each of these, collecting them into `simulated_bootstrap_statistics_for_one_sample` and finally determining an 80% **bootstrap confidence interval** (with an 80%  **confidence level**), which is compared against **actual known true mean** `population_mean_μ`. 

```python
np.percentile(simulated_bootstrap_statistics_for_one_sample, 
              [(100-confidence_level)/2, confidence_level + ((100-confidence_level)/2)])
    if (confidence_interval[0] <= population_mean_μ) and (population_mean_μ <= confidence_interval[1]):
        coverage_count += 1
```

After all **simulations** have completed, the final proportion of the `so_many_bootstrapped_confidence_intervals=1000` which "work" and "capture" `population_mean_μ` is the empirically observed (so-called) **coverage rate**. If this proportion matches the promised 80%  **confidence level** then we have "proved" that the data analysis methodology of **bootstrapped confidence intervals** "works".

Well, for this experimental **simulation**, anyway. In actual fact, we might see the **coverage rate** for **bootstrapped confidence intervals** start to slip (and not meet their alleged guaranteed **coverage rates**) if the **sample sizes** are smaller and the **confidence level** is higher.  So, to be fair, the claims that we can always just "pretend" the **sample** is the **population** are a little bit "overconfident". This experiment indeed "proves" the **bootstrapped confidence intervals** data analysis methodology "works". But only for the exact specifications of our experimental simulation. That means, for this 80% **confidence level** for this `n=300` **sample size** and for this particular **population** modelled by this exact **normal distribution**. The more we're not in this exact context, the less this "proof" actually "proves" anything. If we're in a very different context, we need a very different **simulation** which accurately reflects the new situation that we're considering using **bootstrapped confidence intervals** for. 


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
# STA130 Homework 04 

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

1. Code and write all your answers (for both the "Prelecture" and "Postlecture" HW) in a python notebook (in code and markdown cells) 
    
    > It is *suggested but not mandatory* that you complete the "Prelecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
    
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
- [0.3 points]: Evaluation of correctness and effectiveness of written communication for Question "1"
- [0.3 points]: Correctness of understanding confirmed by code comments and relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] for Question "4"
- [0.3 points]: Evaluation of correctness and effectiveness of written communication for Question "6"
<!-- - [0.3 points]: Evaluation of submission for Question "9" -->

## "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]

**To prepare for this weeks lecture, first watch this video [introduction to bootstrapping](https://www.youtube.com/watch?v=Xz0x-8-cgaQ)**



```python
from IPython.display import YouTubeVideo
YouTubeVideo('Xz0x-8-cgaQ', width=800, height=500)
```

### 1. The "Pre-lecture" video (above) mentioned the "standard error of the mean" as being the "standard deviation" of the distribution bootstrapped means.  What is the difference between the "standard error of the mean" and the "standard deviation" of the original data? What distinct ideas do each of these capture? Explain this concisely in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _To answer this question, you could start a ChatBot session and try giving a ChatBot a shot at trying to explain this distinction to you. If you're not sure if you've been able to figure it out out this way, review [this ChatGPT session](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk4/GPT/SLS/00002_gpt3p5_SEM_vs_SD_Difference.md)._
> - _If you discuss this question with a ChatBot, don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)._
> 
> _Note that the "Pre-lecture" video (above) and the last *Question 5* of The **Week 04 TUT Communication Actvity #2** address the question of "What is bootstrapping?", but the question of "What is the difference between the "standard error of the mean" and the "standard deviation" of the original data?" does not really depend on what bootstrapping is._
> 
> _If you were to be interested in answering the question of "What is bootstrapping?", probably just asking a ChatBot directly would work. Or even something like "Explain variability of means, function of sample size, bootstrapping" or "How does the variability of means of simulated samples change as a function of sample size? Explain this to me in a simple way using bootstrapping!" would likely be pretty effective as prompts. ChatBots are not particularly picky about prompts when it comes to addressing very well understood topics (like bootstrapping). That said, the more concise context you provide in your prompt, the more you can guide the relevance and relatability of the responses of a ChatBot in a manner you desire. The "Further Guidance" under *Question 5* of **Communication Actvity #2** in TUT is a good example of this._
    
</details>

### 2. The "Pre-lecture" video (above) suggested that the "standard error of the mean" could be used to create a confidence interval, but didn't describe exactly how to do this.  How can we use the "standard error of the mean" to create a 95% confidence interval which "covers 95% of the bootstrapped sample means"? Explain this concisely in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Just describe the proceedure itself (probably as reported by a ChatBot), but explain the procedure in your own words in a way that makes the most sense to you. The point is not to understand or explain the theoretical justification as to why this procedure exists, it's just to recognize that it does indeed exist and to briefly describe it. This is because in this class we're going to instead focus on understanding and using 95% bootstrapped confidence intervals. So this "sample mean plus and minus about 2 times the standard error" really only provides some context against which to contrast and clarify bootstrapped confidence intervals_
>
> - _If you continue get help from a ChatBot for this question (as is intended and expected for this problem), don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)._
</details>

### 3. Creating the "sample mean plus and minus about 2 times the standard error" confidence interval addressed in the previous problem should indeed cover approximately 95% of the bootstrapped sample means. Alternatively, how do we create a 95% bootstrapped confidence interval using the bootstrapped means (without using their standard deviation to estimate the standard error of the mean)? Explain this concisely in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _A good explaination here would likely be based on explaining how (and why) to use the `np.quantile(...)` function on a collection of bootstrapped sample means. The "pre-lecture video" describes what this should be, just not in terms of`np.quantile(...)`, right before the "double bam"._
>
> _That said, there are many other questions about bootstrapping that you should be working on familiarizing yourself with as as you're thinking through th proceedure that answers this question._
> 
> - _If you had a_ ~theoretical distribution~ _histogram of bootstrapped sample means representing the variability/uncertianty of means (of "averages") that an observed sample of size n produces, how would you give a range estimating what the sample mean of a future sample of size n might be?_
>
> - _Unlike the "sample mean plus and minus about 2 times the standard error" approach which would only cover **approximately** 95% of the bootstrapped sample means, a 95% bootstrapped confidence interval would cover exactly 95% of the bootstrapped means._
>
> - _While the variability/uncertainty of sample mean statistics when sampling from a population is a function of the sample size (n) [how?], we would NEVER consider using a bootstrapped sample size that was different than the size of the original sample [why?]._
>
> - _Are bootstrapped samples different if they are the same size as the original sample and created by sampling **without replacement**?_

</details>

### 4. The "Pre-lecture" video (above) mentioned that bootstrap confidence intervals could apply to other statistics of the sample, such as the "median". Work with a ChatBot to create code to produce a 95% bootstrap confidence interval for a population mean based on a sample that you have and comment the code to demonstrate how the code can be changed to produce a 95% bootstrap confidence interval for different population parameter (other than the population mean, such as the population median).<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Hint: you can ask your ChatBot to create the code you need, and even make up a sample to use; but, you should work with your ChatBot to make sure you understand how the code works and what it's doing. Just having a ChatBot comment what the code does is not what this problem is asking you to do. This problem wants YOU to understand what the code does. To make sure you're indeed doing this, consider deleting the inline explanatory comments your ChatBot provides to you and write them again in your own words from scratch._
>
> - _Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)!_

</details>

<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Pre-lecture VS Post-lecture HW

Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: some of the "Post-lecture" HW questions continue to address the "Pre-lecture" video, so it's not particularly unreasonable to attempt to work ahead a little bit... 

- The very first question of the the "Post-lecture" HW addresses the previously emphasized topic of *parameters* versus *statistics*, and would again be a very good thing to be clear about in preparation for the upcoming lecture...
    
*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details>    

## "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]

### 5. The previous question addresses making a confidence interval for a population parameter based on a sample statistic. Why do we need to distinguish between the role of the popualation parameter and the sample sample statistic when it comes to confidence intervals? Explain this concisely in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _This question helps clarify the nature and relative roles of (population) parameters and (sample) statistics, which forms the fundamental conceptual relationship in statistics and data science; so, make sure you interact with a ChatBot (or search online or in the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki)) carefully and thoroughly to ensure that you understand the distinctions here in the context of confidence intervals._
>
> - _As always, don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)._

</details>

### 6. Provide written answers explaining the answers to the following questions in an informal manner of a conversation with a friend with little experience with statistics. <br>

1. What is the process of bootstrapping? 
2. What is the main purpose of bootstrapping? 
3. If you had a (hypothesized) guess about what the average of a population was, and you had a sample of size n from that population, how could you use bootstrapping to assess whether or not your (hypothesized) guess might be plausible?
   
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Your answers to the previous questions 3-5 above (and the "Further Guidance" comments in question 3) should be very helpful for answering this question; but, they are very likely be more technical than would be useful for explaining these ideas to your friends. Work to use descriptive and intuitive language in your explaination._

</details>


### 7. The "Pre-lecture" video (above) introduced hypothesis testing by saying that "the confidence interval covers zero, so we cannot reject the hypothesis that the drug is **[on average]** not doing anything".  This conclusion could be referred to as "failing to reject the null hypothesis", where the term "null" refers to the concept of "no effect **[on average]**".  Why does a confidence interval overlapping zero "fail to reject the null hypothesis" when the observed sample mean statistic itself is not zero? Alternatively, what would lead to the opposite conclusion in this context; namely, instead choosing "to reject the null hypothesis"? Explain the answers to these questions concisely in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _This question (which addresses a very similar content to the third question of the previous probelm) is really about characterizing and leveraging the behavior of the variability/uncertainty of sample means that we expect at a given sample size. Understanding why this characterization would explain the answer to this question is the key idea underlying statistics. In fact, this concept is the primary consideration in statistics and the essense of how statistical analysis works._
> 
> - In answering this question it is surely helpful to note the difference between the observed sample values in the sample $x_i$ (for $i = 1, \cdots, n$), the observed sample average $\bar x$, and the actual value of the parameter $\mu$ clearly. Hopefully the meanings and distinctions here are increasingly obvious, as they should be if you have a clear understanding of the answer to question "5" above. Related to this, the quotes above have been edited to include "**[on average]**" which more accurately clarifies the intended meaning of the statements from the video. It's very relevent (again related to Question "5" above) to understand why are we bothering with making an explicit distinction with this, and why is it slightly different to say that "the drug is on average not doing anything" as opposed to saying "the drug is not doing anything"._
> 
> Using a **null hypotheses** (and corresponding **alternative hypothesis**) will be addressed next week; but, to give a sneak peak preview of the **hypothesis testing** topic, the "null" and "alternative" are formally specified as 
>    
> $H_0: \mu=0 \quad \text{ and } \quad H_A: H_0 \text{ is false}$
>
> which means that our **null hypotheses** is that the average value $\mu$ of the population is $0$, while our **alternative hypothesis** is that the average value $\mu$ of the population is not $0$. 
> 
> **Statistical hypothesis testing** proceeds on the basis of the **scientific method** by defining the **null hypothesis** to be what we beleive until we have sufficient evidence to no longer believe it. As such, the **null hypotheses** is typically something that we *may not actually believe*; and, actually, the **null hypotheses** simply serves as a sort of "straw man" which we in fact really intend to give evidence against so as to no longer believe it (and hence move forward following the procedure of the **scientific method**).
</details>

### 8. Complete the following assignment. 


### Vaccine Data Analysis Assignment

**Overview**

The company AliTech has created a new vaccine that aims to improve the health of the people who take it. Your job is to use what you have learned in the course to give evidence for whether or not the vaccine is effective. 

**Data**
AliTech has released the following data.

```csv
PatientID,Age,Gender,InitialHealthScore,FinalHealthScore
1,45,M,84,86
2,34,F,78,86
3,29,M,83,80
4,52,F,81,86
5,37,M,81,84
6,41,F,80,86
7,33,M,79,86
8,48,F,85,82
9,26,M,76,83
10,39,F,83,84
```

**Deliverables**
While you can choose how to approach this project, the most obvious path would be to use bootstrapping, follow the analysis presented in the "Pre-lecture" HW video (above). Nonetheless, we are  primarily interested in evaluating your report relative to the following deliverables.

- A visual presentation giving some initial insight into the comparison of interest.
- A quantitative analysis of the data and an explanation of the method and purpose of this method.
- A conclusion regarding a null hypothesis of "no effect" after analyzing the data with your methodology.
- The clarity of your documentation, code, and written report. 

> Consider organizing your report within the following outline template.
> - Problem Introduction 
>     - An explaination of the meaning of a Null Hypothesis of "no effect" in this context
>     - Data Visualization (motivating and illustrating the comparison of interest)
> - Quantitative Analysis
>     - Methodology Code and Explanations
>     - Supporting Visualizations
> - Findings and Discussion
>     - Conclusion regarding a Null Hypothesis of "no effect"
>     - Further Considerations

**Further Instructions**
- When using random functions, you should make your analysis reproducible by using the `np.random.seed()` function
- Create a CSV file and read that file in with your code, but **do not** include the CSV file along with your submission


### 9. Have you reviewed the course wiki-textbook and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
>  Here is the link of [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) in case it gets lost among all the information you need to keep track of  : )
> 
> Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)

</details>

_**Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)!**_

## Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 

<br>
<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations
    
1. What is the difference between reporting a sample statistic (say, from the Canadian Social Connection Survey) as opposed to the a population parameter (chacterizing the population of the Canadians the Canadian Social Connection Survey samples)?
2. Why should bootsrapping (and confidence intervals in particular) be utilized when reporting sample statistics (say, from the Canadian Social Connection Survey)?
3. How does bootsrapping (and confidence intervals in particular) help us relate the data we have to all Canadians? 
4. Is the population that the Canadian Social Connection Survey samples really actually all Canadians? Or is it biased in some way? 
5. Why are the previous questions "Ethical" and "Professional" in nature?
6. If the Canadian Social Connection Survey samples Canadians in some sort of biased way, how could we begin considering if the results can generalize to all Canadians; or, perhaps, the degree to which the results could generalize to all Canadians?
</details>    

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>

### Current Course Project Capability Level
    
**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
        
At this point in the course you should be able to compute a bootstrap confidence interval for the (candian) population mean of a numeric variable of the sample of the Canadian Social Connection Survey. On the basis of only using the techniques we've encountered in the course so far, it would only be possible to assess a null hypothesis of "no effect" if we had "paired" (e.g., "before and after") measurements in our data; but, we could of course assess a hypothesized parameter value estimated by the bootstrapped confidence interval of a relevant sample statistic...
    
1. What are the different samples and populations that are part of the data related to the Canadian Social Connection Survey?
    
2. Consider whether or not we have "paired" (e.g., "before and after") measurements in our data which could be used to assess a null hypothesis of "no effect" (in the manner of the "Pre-lecture" HW video above); and, if such data is available, create a confidence interval for the average sample difference and use it to assess a null hypothesis of "no effect".
    
3. Pick a couple numeric variables from the Canadian Social Connection Survey with different amounts of non-missing data and create a 95% bootstrapped confidence intervals estimating population parameters for the variables.  
    1. You would not want to do this by hand [why?]; but, could you nonetheless describe how this process would be done if you were to do it by hand? 

    2. [For Advanced Students Only] There are two factors that go into the uncertainty of sample means: the standard deviation of the original sample and the size of the sample (and they create a standard error of the mean that is theoretically "the standard deviation of the original sample divided by the square root of n").  Compute the theoretical standard errors of the sample mean for the different variables you've considered; and, if they're different, confirm that they influence the (variance/uncertainty) bootstrapped sampling distribution of the mean as expected

</details>            
