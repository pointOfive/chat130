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


