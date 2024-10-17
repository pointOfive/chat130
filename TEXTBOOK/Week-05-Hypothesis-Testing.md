# P-values And How To Use And Not Use Them

**Tutorial/Homework: Topics**

1. [Null and Alternative Hypotheses](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Null-and-Alternative-Hypotheses)
2. [The Sampling Distribution of the Null Hypothesis](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Sampling-Distribution-of-the-Null-Hypothesis)
    1. [The role Sample Size n](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Role-of-Sample-Size-n) (re: [How n Drives Standard Error](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#How-n-Drives-Standard-Error))
    2. ["One sample" paired difference hypothesis tests with a "no effect" null](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#one-sample-paired-difference-hypothesis-tests-with-a-no-effect-null)
3. [p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#p-values)
    

**Tutorial/Homework: Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

3. [Using p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#using-p-values)
	1. [Using confidence intervals](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-relationship-between-p-values-and-confidence-intervals)
	2. [Misusing p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#misusing-p-values)
	3. [One- versus two-sided hypothesis tests](https://github.com/pointOfive/stat130chat130/wiki/Week-05-Hypothesis-Testing#one--versus-two-sided-hypothesis-tests)


**Lecture: New Topics**

1. [Type I and Type II Errors](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Type-I-and-Type-II-errors)
2. [The Reproducibility Crisis](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Reproducibility-Crisis)


## Tutorial/Homework: Topics


### Null and Alternative Hypotheses

|![](https://i.etsystatic.com/11565693/r/il/616685/5675811010/il_570xN.5675811010_se1e.jpg)|![](https://blog.coniferresearch.com/hs-fs/hubfs/20210715-con-frameworks-for-change-kuhn-cycle.png?width=468&name=20210715-con-frameworks-for-change-kuhn-cycle.png)|
|-|-|
| | |

**Statistical hypothesis testing** implements the scientific method of using data to falsify a theory. It follows a precise protocol of proposing an assumption about a population and then evaluating the evidence for or against this assumption. This evidence is based on **statistics** (from samples) to make inferences about the population. In other words, a hypothesis about a population parameter is made, and a statistic is used to infer something about that parameter. This allows us to consider the plausibility of the hypothesis regarding the population parameter, based on the sample data. In essence, hypothesis testing enables researchers to make probabilistic statements about [population parameters](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#statistics-estimate-parameters) without knowing the entire population.

The process of hypothesis testing starts with the formulation of the **null hypothesis**, $H_0$. Hypothesis testing works by assessing whether there is enough evidence against $H_0$, which may initially seem confusing because we "assume $H_0$ is true" while simultaneously looking for evidence to disprove it. However, the null hypothesis is only assumed to be true until there is sufficient evidence to suggest otherwise.

- We assume $H_0$ is true, but the goal is to gather evidence to reject or fail to reject $H_0$. So, when we "assume $H_0$ is true," we are secretly wondering whether "$H_0$ might actually be false."

The most common forms of $H_0$ tend to represent "default" or "uninteresting" beliefs about a situation, such as "there is no effect of the treatment" or "the coin is fair." If evidence against $H_0$ is found, we conclude that something **interesting** is happening, such as "there is an effect of the treatment!" or "the coin isn’t fair!"

Here are some examples of acceptable forms that a null hypothesis could take:

- $H_0:$ There is "no effect" of a treatment intervention on the average value in the population.
    - $H_0: \mu = 0$ (equivalent to the above if $\mu$ represents the change in the average value due to the treatment).
- $H_0: \mu = \mu_0$ (e.g., $\mu_0$ might be 0 or another hypothesized value).
    - $H_0: p = 0.5$ (e.g., "the chance a coin lands heads is 50/50").
    - $H_0:$ The coin is "fair."

The **alternative hypothesis**, $H_A$, is a simple statement that "$H_0$ is false." Hypothesis testing focuses primarily on assessing the null hypothesis, so the role of $H_A$ is to represent the possibility that $H_0$ is incorrect.

- If there is **strong evidence** against $H_0$, we "reject $H_0$."
- However, we avoid saying "we accept $H_A$" because hypothesis testing is only about giving evidence against $H_0$, not "proving" anything. It’s more accurate to say "we reject $H_0$."
- Similarly, we don’t say "we accept $H_0$" when there isn’t enough evidence to reject it. Instead, we say "we fail to reject $H_0$."

> These distinctions in terminology might seem pedantic, but using the correct language will help you better understand the principles and reasoning behind hypothesis testing, particularly the role of $H_0$.

### The Sampling Distribution of the Null Hypothesis

The **sampling distribution** under the null hypothesis characterizes the variability or uncertainty of the observed sample statistic, in the hypothetical situation where we are assuming the **null hypothesis** $H_0$ is true. Although the observed sample statistic has already been recorded, we can use **simulation** to repeatedly generate synthetic samples (and corresponding sample statistics) under the assumption that $H_0$ is true. This approach helps us understand the variability of the sample statistic in the hypothetical scenario where the null hypothesis is true.

For example, if we wanted to test whether a coin is fair, our hypotheses would be:

- **$H_0$:** The coin is fair (i.e., the probability of heads is 0.5).
- **$H_1$:** $H_0$ is false (i.e., the probability of heads is not 0.5).

When simulating the sampling distribution under $H_0$, three key factors come into play:

- **`n_coinflips_per_sample`:** The sample size (i.e., the number of coin flips used as evidence regarding the null hypothesis).
- **`n_sample_simulations`:** The number of samples to simulate (i.e., the number of times we flip a coin `n_coinflips_per_sample` times).
- **`H0_p = 0.5`:** The hypothesized value of the parameter under the null hypothesis, specifically $H_0: p = 0.5$, where $p$ represents the probability of the coin landing on heads.

```python
import numpy as np
from scipy import stats
import plotly.express as px

# Set the parameters
n_coinflips_per_sample = 100  # Number of coin flips per sample
n_sample_simulations = 10000  # Number of samples to simulate
simulated_proportion_coinflips_heads = np.zeros(n_sample_simulations)
H0_p = 0.5

# Simulate the samples
for i in range(n_sample_simulations):
    # Simulate the coin flips
    n_coinflips = np.random.choice(['heads','tails'], p=[H0_p,1-H0_p], size=n_coinflips_per_sample, replace=True)
    simulated_proportion_coinflips_heads[i] = (n_coinflips=='heads').mean()
    # or just use `proportion_coinflips_heads[i] = 
    #              np.random.choice([0,1], p=[H0_p,1-H0_p], size=n_coinflips_per_sample).mean()`

# or instead of the `for` loop, this could be done as
# simulated_coinflips = stats.binom.rvs(n=1, p=0.5, size=(n_sample_simulations, n_coinflips_per_sample))
# simulated_proportion_coinflips_heads = simulated_coinflips.mean(axis=1)

# Create the histogram using plotly.express
fig = px.histogram(simulated_proportion_coinflips_heads, nbins=30, 
                   title='Sampling Distribution of Proportion of Heads (Fair Coin)',
                   labels={'value': 'Proportion of Heads', 'count': 'Frequency'})
fig.update_traces(name='Simulated Proportions')
fig.update_xaxes(range=[0, 1])
fig.show()
```

#### The role of Sample Size n

The exercise of **creating the sampling distribution** of the proportion of heads when a coin is fair might initially seem strange or counterintuitive. Why run simulations at all? Why not just flip the coin a large number of times to determine if it's fair? In fact, the `stats.binom.rvs(n=1, p=0.5, size=(n_samples, n_coinflips_per_sample))` formulation shows that this is exactly what we're doing. We've simply organized the coin flips into simulations (with each row representing a simulation and each column representing an individual coin flip in that simulation).

Think of the "fair coin" example as a template for data collection. We gather some number of data points (with a sample size $n$) and then calculate a **statistic** from that sample. The key question is: what is the variability or uncertainty of that sample statistic? This variability, of course, depends on the sample size $n$. The statistic is used to infer the corresponding population parameter, and the strength of the evidence it provides depends on the variability of the statistic, which is directly influenced by the sample size $n$.

> This concept was previously discussed in the context of the [standard error versus the standard deviation](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#How-n-Drives-Standard-Error) in confidence intervals. The **standard deviation** of the sampling distribution (of a statistic) under the null hypothesis addresses the same issue: there is a **standard error** of a statistic under the assumption of the null hypothesis, which captures the variability or uncertainty of the statistic. This, in turn, influences the strength of the evidence that the data can provide regarding the null hypothesis $H_0$.


#### "One Sample" Paired Difference Hypothesis Tests with a "no effect" Null

In cases where two samples are involved and the observations are **paired differences** (such as in a "before versus after" treatment intervention), two possible null hypotheses can be considered:

- **$H_0: \mu = 0$** (there is "no effect" of the treatment intervention: the change in the population average, $\mu$, is 0).
- **$H_0: p = 0.5$** (changes due to the treatment intervention are "random": the chance of improvement, ppp, is 50/50).

The first null hypothesis results in what's known as a **paired t-test**. However, this test is beyond the scope of STA130, so we will not discuss it further. The second null hypothesis has a similar structure to the "coin flipping" hypothesis testing we covered earlier.

- The sample pairs are compared, and the **proportion "greater than"** is the observed statistic.
- The **sampling distribution** under the null is then simulated using a "fair" coin-flipping exercise.

> If you’re having trouble understanding how to use the "coin-flipping code" for a "paired difference" hypothesis test, seek help from a ChatBot, office hours, or study buddies to clarify why "coin flipping" and "paired difference" hypothesis tests are equivalent.


### p-values

A **p-value** is defined as **the probability that a statistic is as or more extreme than the observed statistic, assuming the null hypothesis is true**.

A p-value measures the **strength of the evidence** against $H_0$. The smaller the p-value, the stronger the evidence against the null hypothesis. But why is this so? A p-value is calculated by comparing the observed sample statistic to the **sampling distribution** of the statistic under the assumption that the null hypothesis is true. If the observed statistic seems implausible relative to this sampling distribution, it suggests that the evidence in the sample data does not align with the assumption that the null hypothesis is true. In other words, the sample data provides evidence **against** the null hypothesis.

To quantify how much the sample data deviates from the sampling distribution under the null hypothesis, we compute the p-value. This value represents the proportion of statistics sampled from the distribution under the null hypothesis that are "as or more extreme" than the observed statistic. The "as or more extreme" calculation involves comparing the relative differences between the observed statistic and the parameter specified by the null hypothesis, and similarly, the relative differences for the statistics sampled from the null hypothesis distribution.

For the "coin flipping" simulation example above, this would be done as follows.

```python
import plotly.graph_objects as go

observed_proportion_statistic = 0.6
print("If we observed a proportion of", observed_proportion_statistic, "heads out of", n_coinflips_per_sample, "coin flips")
print("The (p-value) chance that we would get a proportion (statistic)")
print('further from 50/50 ("as or more extreme") than the proportion of')
print("heads we saw (observed statistic) if the coin was fair (H0 is true)")
as_or_more_extreme = abs(simulated_proportion_coinflips_heads - H0_p) >= abs(observed_proportion_statistic - H0_p)
print("would be", as_or_more_extreme.sum()/n_sample_simulations, "which we estimated using", n_sample_simulations, "simulations of", n_coinflips_per_sample, "coin flips")
print("as illustrated below")

fig = px.histogram(simulated_proportion_coinflips_heads, nbins=30, 
                   title='Sampling Distribution of Proportion of Heads (Fair Coin)',
                   labels={'value': 'Proportion of Heads', 'count': 'Frequency'})
fig.update_traces(name='Simulated Proportions<br>under the assumption<br>that H0 is true<br>')

# Add vertical line for observed proportion
fig.add_trace(go.Scatter(x=[observed_proportion_statistic] * 2,
                         y=[0, max(np.histogram(simulated_proportion_coinflips_heads, bins=30)[0])],
              mode='lines', line=dict(color='red', width=2), name='Observed Proportion'))

# Mark the p-value area
x_as_or_more_extreme = simulated_proportion_coinflips_heads[as_or_more_extreme]
fig.add_trace(go.Scatter(x=x_as_or_more_extreme, y=0*x_as_or_more_extreme,
    mode='markers', marker=dict(color='red', size=10, symbol='x'),
    name='"as or more extreme" (symmetric)'))
fig.show()
```


## Tutorial/Homework: Lecture Extensions


### Using p-values

To ensure the correct and effective use of p-values, a _numerical_ p-value must be interpreted as a _qualitative_ result. Practically speaking, decisions need to be made in a clear and straightforward way. Specifically, we will either:

- **Fail to reject the null hypothesis based on the evidence at hand**, or
- **Reject the null hypothesis based on the strength of the available evidence.**

The following table characterizes the "strength of evidence" against a null hypothesis based on the p-value:

| p-value                | Evidence                                         |
| ---------------------- | ------------------------------------------------ |
| $$p > 0.1$$            | No evidence against the null hypothesis          |
| $$0.1 \ge p > 0.05$$   | Weak evidence against the null hypothesis        |
| $$0.05 \ge p > 0.01$$  | Moderate evidence against the null hypothesis    |
| $$0.01 \ge p > 0.001$$ | Strong evidence against the null hypothesis      |
| $$0.001 \ge p$$        | Very strong evidence against the null hypothesis |

A common approach to **formal hypothesis testing** involves selecting a **significance level** (often $\alpha = 0.05$) and then deciding to **reject the null hypothesis at the $\alpha$-significance level** if the p-value is less than the chosen $\alpha$ threshold.

- **IMPORTANTLY:** For this approach to be valid, the **$\alpha$-significance level must be chosen _before_** the p-value is computed.

The benefit of using this table is that you don't have to set a single $\alpha$-significance level. Instead, the table provides a collection of **$\alpha$-significance level thresholds** that we can compare the p-value to. Since this framework is agreed upon in advance, there's no risk of **"changing the $\alpha$-significance level"** after observing the data and calculating the p-value.

#### The Relationship Between p-values and Confidence Intervals

If you noticed that the numerical value of the **$\alpha=0.05$ significance level** is "one minus" the numerical value of a **95% confidence level** for a 95% confidence interval, good job. This is not a coincidence.  Recall the interpretation of a 95% confidence interval, and consider this against the interpretation , given below, of an $\alpha=0.05$ significance level. Let’s break down the relationship:

- A **95% confidence interval** claims that 95% of independently and identically distributed (i.i.d.) samples (with the same sample size as our observed sample) would result in a confidence interval that **captures** the true population parameter it estimates. Correspondingly, 5% of these i.i.d. samples would result in an interval that does **not capture** the true parameter.
    - However, once a confidence interval is constructed, it either **does** or **does not** capture the true population parameter. There is no longer an associated probability.
- **Rejecting a hypothesis** at the **$\alpha=0.05$ significance level** means that 5% of i.i.d. samples (with the same sample size as our observed sample) would incorrectly reject the null hypothesis when it is true.
    - In other words, hypothesis testing at the $\alpha=0.05$ level will **wrongly reject** a true null hypothesis in 5% of cases where the null hypothesis is actually true. This is equivalent to how 5% of i.i.d. samples using a 95% confidence interval construction will result in a "failing" interval that does not capture the true population parameter.

This is why **confidence intervals** can be used to perform hypothesis testing without explicitly using a p-value. Both procedures provide equivalent levels of **reliability** in the conclusions they draw about a null hypothesized parameter. With $\alpha=0.05$ hypothesis testing, 5% of samples will wrongly reject a true null hypothesis, just as 5% of samples using a 95% confidence interval will fail to capture the true population parameter.

- To simplify: use the **"strength of evidence" table** above when interpreting p-values if you’re making a decision about a null hypothesis. Alternatively, construct a confidence interval (e.g., 95%) and check if the hypothesized parameter value is **contained** within it to guide your decision. The confidence interval approach provides **coverage rate guarantees** based on the procedure used.
	
- The advantage of using confidence intervals is that they offer more than just a **point estimate** of the population parameter or evidence against a hypothesized value. They provide **inference** about the plausible range of values for the true population parameter based on the data.
    
- So, confidence intervals are quite useful! They provide **inference** about the true parameter and can be used for **decision-making** just like formal hypothesis testing.

#### Misusing p-values

Recalling that a **95% confidence level** indicates the reliability of our procedure (and does **not** refer to any specific confidence interval after it’s constructed), we must be careful about how we talk about confidence intervals. The following statements are **technically correct** and therefore allowed:

- This is a **95% confidence interval**.
- I have **95% confidence** that this constructed interval captures the true population parameter.
- I have used a **confidence interval procedure** that will "work" for 95% of hypothetical i.i.d. samples.
- There's a **95% chance** this confidence interval "worked" and captures the true population parameter.

The following statements, however, are **technically incorrect** and are **not allowed**:

- "There's a 95% chance the parameter is in this confidence interval." ← **NOPE**, parameters don’t have "chances" of being something.
- "There's a 95% probability the parameter is in this confidence interval." ← **NOPE**, parameters don’t behave probabilistically.

Sorry to be so strict about this, but using the wrong terminology could lead other statisticians to question your understanding, and we don’t want that!

---

But what does this have to do with **misusing p-values**? This section is about confidence interval terminology, right? Well, since we've seen the direct connection between hypothesis testing using $\alpha$-significance levels and confidence intervals, it should come as no surprise that p-values can also be misinterpreted. To avoid misusing p-values, here are three **correct ways** to use them:

1. **Remember the definition**: A p-value is the probability that a statistic is as or more extreme than the observed statistic, **assuming the null hypothesis is true**. **That’s it**. A p-value cannot mean anything else.
    
2. **Interpret p-values in terms of the "strength of evidence"**: Forget the exact definition and use the "strength of evidence" table from the [Using p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#using-p-values) section. This interpretation does not contradict the definition but simplifies it for practical use. It correctly interprets the meaning of the definition of a p-value in terms of the strength of evidence it indicates the data provides against the null hypothesis.
    
3. **Compare the p-value to the pre-determined $\alpha$-significance level**: If the p-value is smaller than the $\alpha$ threshold, reject the null hypothesis; if it’s larger, fail to reject it. **Never change the $\alpha$ threshold after seeing the data**, as this is considered "funny business" and compromises the integrity of your analysis.


#### One- versus Two-Sided Hypothesis Tests

Also known as **one-tailed hypothesis tests** (not to be confused with the distinction between "one-sample" or "two-sample" situations), a variation on the traditional **equality null hypothesis** (such as $H_0: p = 0.5$) allows us to address comparisons in an ordered manner by replacing the **two-sided** (or two-tailed) alternative hypothesis (such as $H_A: p \neq 0.5$) with **one-sided** (or one-tailed) alternatives. Namely:

$$\Large H_A: p > 0.5 \quad\textrm{ or }\quad H_A: p < 0.5 $$

which correspond to:

$$\Large \quad H_0: p \leq 0.5 \quad\textrm{ or }\quad H_0: p \geq 0.5$$

Naturally, the **$H_0$-centric form** remains preferred, so these correspond to:

$$\Large H_0: p \leq 0.5 \textrm{ and } H_A: H_0 \textrm{ is false} \quad\textrm{ or }\quad H_0: p \geq 0.5 \textrm{ and } H_A: H_0 \textrm{ is false}$$

Using a **one-sided (one-tailed) hypothesis test** allows us to address questions of **directional difference**, rather than simply testing for "no difference." For example, this lets us determine whether an unfair coin is biased in favor of heads or against heads, depending on our choice of null and alternative hypothesis. Depending on the research question, we may be interested in providing evidence against the null hypothesis in a way that considers a single comparative direction, rather than differences in either direction.

Additionally, by using a one-sided test that focuses on detecting a difference in one direction only, the **p-value** will necessarily become smaller, as "as or more extreme" is now considered in only **one tail** of the sampling distribution of the statistic.


## Lecture: New Topics


### Type I and Type II errors


**Type I** and **Type II errors** are key concepts in **formal (statistical) hypothesis testing**. But they have immediate generalizations to **machine learning** and **classification**-oriented **predicted modelling**, as we shall see later in the course. 

A **Type I Error** (or **False Positive**) occurs when we *reject* a null hypothesis that is actually *true*.

- In other words, a Type I error is when we conclude that there is an effect or a difference when, in reality, none exists.
- For example, if a medical test incorrectly shows that a patient has a disease when they actually don’t, this is a Type I error relative to the null hypothesis that a patient does not have a disease. 
- The probability of making a Type I error is denoted by "alpha" $\alpha$ corresponding to the $\alpha$-significance level threshold of the hypothesis test.
    - It is customary to use an $\alpha=0.05$ significance level threshold so that there's only a 5% (1 in 20) chance that we've made a Type I error when we reject a null hypothesis, meaning that we've wrongly rejected a null hypothesis that is in fact true. 

A **Type II Error** (or **False Negative**) occurs when we *fail to reject* a null hypothesis that is actually *false*.

- In other words, a Type II error is when we conclude that there is no effect or difference present when there actually is one.
- For example, if a medical test incorrectly shows that a patient does not have a disease when they actually do, this is a Type II error relative to the null hypothesis that a patient does not have a disease. 
- The probability of making a Type II error is denoted by "beta" $\beta$ (to associated it with "alpha" $\alpha$) and determining $\beta$  is the objective of so-called "sample size power analyses" (but these themselves are beyond the scope of STA130)
    - Even if "sample size power analyses" are beyond the scope of STA130, we should still be able to sensibly talk about and understand the notion of a "beta" $\beta$ the probability of making a Type II error 

| Decision       | Null Hypothesis is True   | Null Hypothesis is False |
|----------------|---------------------------|--------------------------|
| Reject Null    | **Type I Error (with chance α)**       | Correct Decision         |
| Fail to Reject | Correct Decision           | **Type II Error (with chance β)**     |

With this in mind, and recalling the previous initial conversation regarding [Null and Alternative Hypotheses](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Null-and-Alternative-Hypotheses), please abide by the following factual realities of the formal hypothesis testing framework based on null hypotheses.

- We use the data at hand to provide some degree of **strength of evidence** against the null hypothesis.

- Based on the **strength of evidence** shown by the data, we might choose to reject the null hypothesis. However, we could be wrong if the null hypothesis is actually true, even though the evidence suggests otherwise.
    - If this happens, we make a **Type I error** by incorrectly rejecting a true null hypothesis.
    
- If the data shows a **lack of strength of evidence** against the null hypothesis, we might choose to **fail to reject** it. However, we could still be wrong if the null hypothesis is actually false, despite the lack of strong evidence against it.
    - If this happens, we make a **Type II error** by failing to reject a false null hypothesis.
    
- We never prove the null hypothesis, and we never prove the alternative hypothesis. We only give evidence or fail to give evidence against the null hypothesis. 

- Therefore, we should never say "we accept" a null hypothesis or an alternative hypothesis. Instead, we either **reject the null hypothesis** based on the evidence or **fail to reject it**.


In the context of statistical hypothesis testing, we don't "prove $H_0$ is false"; instead, we provide evidence against $H_0$. For example, we might say, "We reject the null hypothesis with a p-value of X, meaning we have evidence against $H_0$." Similarly, we don’t "prove $H_0$ is true"—we simply lack evidence to reject it, so we might say, "We fail to reject the null hypothesis with a p-value of Y."

> Later on in the course, in the aforementioned context of **machine learning** and **classification**-oriented **predicted modelling**, we shall begin to examine the tradeoffs that we can crucial consider to in order to appropriately balance the risk of Type I and Type II errors (which we shall in that context begin calling *false positives* and *false negatives*). Indeed, it is generally the case that reducing the likelihood of one of these mistakes typically increases the likelihood of making the other of these mistakes. So the consequences of each type of mistake needs to be considered carefully to identify an optimal tradeoffs between the two that's appropriate for the context at hand. 

For now however, then, we would follow the following **Steps in Hypothesis Testing**.

1. **Formulate the Null Hypothesis (H₀):** This hypothesis often simply states that there is no effect or no difference.
2. **Collect Data and Perform a Test:** Use confidence intervals or p-values to make a decision regarding the null hypothesis.
3. **Make a Decision:** Communicate the nature of your decision based on a characterization of statistical evidence of your analysis.
4. **Alternative Hypotheses (H₁) are simple:** They're just "Not H₀" which is pretty boring and why confidence intervals are interesting.


### The Reproducibility Crisis 

It turns out using p-values and hypothesis testing appropriately is indeed apparently very challenging as judged by what's going on in scientific research these days. What's happening is many Type I (and Type II) errors seem to be getting made at a shockingly alarming rate in the era of modern research, leading to a lack of reproducibility of scientific "findings". Here's a figure from a 2016 study on this topic from the journal, Nature. And next to that is an image showing how the "strength of evidence" that's being reported looks suspiciously similar to "random noise" from a normal distribution. 

|![](https://bjoern.brembs.net/wp/wp-content/uploads/2017/03/reproducibility-graphic-online11.jpeg)|![](https://github.com/pointOfive/STA130_Week6_Slides/blob/main/images/repro2.png)|
|-|-|
| | |

So indeed there is quite the Contra-Versy (or is it Con-TROV-ersy?) around p-values. 

|![](https://505sanchez.com/wp-content/uploads/2018/08/Winning650pw-1.jpg)|![](https://i.imgflip.com/1mt8h2.jpg)|
|-|-|
| | |

On a general level it seems quite clear that p-values and hypothesis testing methodologies MUST play some ongoing contributing role in the so-called "replication crisis" rampantly afflicting mordern science; namely, "significant findings" made in scientific studies are not able to be reproduced by future studies at an alarming rate; and, this whole paradigm of "significant findings" is based on p-values and hypothesis testing... so, something's going on with this methodology in some way...
    
More specifically however, p-values are themselves quite problematic. To see this, just briefly consider the following article titles...

- [Why are p-values controversial?](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1277161) 
- [What a nerdy debate about p-values shows about science and how to fix it](https://www.vox.com/science-and-health/2017/7/31/16021654/p-values-statistical-significance-redefine-0005)
- [The reign of the p-value is over: what alternative analyses could we employ to fill the power vacuum?](https://royalsocietypublishing.org/doi/10.1098/rsbl.2019.0174)
- [Scientists rise up against statistical significance](https://www.nature.com/articles/d41586-019-00857-9)
- [Statistics experts urge scientists to rethink the p-value](https://www.spectrumnews.org/news/statistics-experts-urge-scientists-rethink-p-value)

While the issues here are relatively advanced and subtle (as introduced [here](https://www2.stat.duke.edu/~berger/p-values.html), presented [here](https://www.jarad.me/courses/stat587Eng/slides/Inference/I06-Pvalues/why_pvalues_dont_mean_what_you_think_they_mean.pdf), and demonstrated using simulation [here](https://jaradniemi.shinyapps.io/pvalue/)), the problem essentially comes down to the fact that most scientists (or just people) don't know how to really interpret the numeric value of a p-value. There are therefore two current proposed solutions to address this challenge.
    
1\. Just interpret p-values as describe in the previous "[misusing p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#misusing-p-values)" section of this weeks wiki-textbook. Basically, just interpret p-values using the following table (which really isn't that hard, so it's surprising that this solution isn't more broadly adopted).
    
|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
    
2\. Only do **hypothesis testing** on the basis of confidence intervals, not **p-values** (which might be the best solution wherever doing so is a realistic, convenient possibility).