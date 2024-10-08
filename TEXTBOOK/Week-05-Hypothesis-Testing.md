**Tutorial/Homework: Topics**

1. [Null and Alternative Hypotheses](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Null-and-Alternative-Hypotheses)
2. [The Sampling Distribution of the Null Hypothesis](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Sampling-Distribution-of-the-Null-Hypothesis)
    1. [The role Sample Size n](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Role-of-Sample-Size-n) (re: [How n Drives Standard Error](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#How-n-Drives-Standard-Error))
    2. ["one sample" paired difference hypothesis tests with a "no effect" null](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#one-sample-paired-difference-hypothesis-tests-with-a-no-effect-null)
3. [p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#p-values)
    

**Tutorial/Homework: Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

3\. [Using p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#using-p-values)\
___ i\. [Using confidence intervals](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-relationship-between-p-values-and-confidence-intervals)\
___ ii\. [Misusing p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#misusing-p-values)\
___ iii\. [one- versus two-sided hypothesis tests](https://github.com/pointOfive/stat130chat130/wiki/Week-05-Hypothesis-Testing#one--versus-two-sided-hypothesis-tests)


**Lecture: New Topics**

1. [Type I and Type II Errors](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Type-I-and-Type-II-errors)
2. [The Reproducibility Crisis](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#The-Reproducibility-Crisis)


## Tutorial/Homework: Topics


### Null and Alternative Hypotheses

|![](https://i.etsystatic.com/11565693/r/il/616685/5675811010/il_570xN.5675811010_se1e.jpg)|![](https://blog.coniferresearch.com/hs-fs/hubfs/20210715-con-frameworks-for-change-kuhn-cycle.png?width=468&name=20210715-con-frameworks-for-change-kuhn-cycle.png)|
|-|-|
| | |

Statistical hypothesis testing implements the scientific method of using data to falsify a theory. Specifically, it follows a very technically precise protocol of proposing an assumption about a population, and then giving (or failing to give) evidence against this assumption. The way this evidence (or lack of evidence) is created is by using statistics (of samples) to make inferences about the population. That is, a hypothesis about a parameter of the population is made, a statistic is then used to provide inference about this parameter, and this in turn allows us to consider the plausibility of the hypothesis regarding the parameter of the population (based on 
the sample). In essence, hypothesis testing allows researchers to make probabilistic statements about [population parameters](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#statistics-estimate-parameters) without knowing the entire population. 

The methodical approach of hypothesis testing to making decisions about a population based on observed sample data begins by formulating the null hypothesis, $H_0$. Hypothesis testing works by giving (or failing to give) evidence against $H_0$, which at first may appear confusing because we are "assuming $H_0$ is true" just to immediately turn around and attempt to "give evidence $H_0$ is not true". The thing is, the null hypothesis is only assumed to be true until there is sufficient evidence to suggest otherwise.

- We assume $H_0$ is true, but in fact hypothesis testing is based on giving (or failing to give) evidence against $H_0$ in order to reject $H_0$... so when we "assume $H_0$ is true", we're secretly really thinking "perhaps $H_0$ is actually really not true".

The most common forms of $H_0$ therefore tend to be the most default "uninteresting" beliefs about a situation, such as "there is no effect of the treatment" or "the coin in fair". Then, if we give evidence against $H_0$, that means we believe that indeed there is in fact something "interesting" going on, such as "there is an effect of the treatment!" or "the coin isn't fair -- there's cheating!" which is probably really what we're trying to use hypothesis testing to do anyway. 

Here are some acceptable forms that a null hypothesis could take.

- $H_0:$ There is "no affect" of a treatment intervention on the average value in the population
    - $H_0: \mu = 0$ [is an equivalent statement to $H_0$ above if $\mu$ is the parameter representing the change in average values in the population due to the treatment intervention]
- $H_0: \mu = \mu_0$ [for example, $\mu_0$ might be equal to $0$ or some other hypothesized value]
    - $H_0: p = 0.5$ [for example could refer to the statement that "the chance a coin is heads is 50/50"]
    - $H_0:$ The coin is "fair"

A somewhat curious circumstance is the alternative hypothesis $H_A$, which initially appears to be the same "kind" of thing as the null $H_0$.  But the alternative $H_A$ is only the very simple statement " $H_A: H_0 \textrm{ is false}$ ", which emphasizes that the only thing hypothesis testing really addresses in the null hypothesis $H_0$.

- If we have "strong evidence against $H_0$" then we will likely "choose to reject $H_0$".
- Curiously again, it is not recommended to therefore correspondingly say "we accept $H_A$"... this is because rather than "proving" anything, hypothesis testing is only concerned with "giving evidence against $H_0$"... so saying "we accept $H_A$" is just not in the spirit of hypothesis testing, whereas saying "we choose to reject $H_0$" is more appropriate language.
- In the same way, since we are never "proving" anything with hypothesis testing, we would never say "we accept $H_0$" (if we failed to provide sufficient evidence against $H_0$) when doing hypothesis testing... we would instead rather say "we fail to reject $H_0$".

> These "appropriate language usage" caveats may well seem like they are splitting hairs in an overly pedantic manner; but, adopting the correct use of terminology and nomenclature will all you to more quickly correctly orient your understanding of hypothesis testing around the use and meaning of $H_0$.


### The Sampling Distribution of the Null Hypothesis

The sampling distribution under the null characterizes the variability/uncertainty of the observed sample statistic in the hypothetical situation that the null hypothesis $H_0$ is true. Of course the observed sample statistic has already been observed, but we could nonetheless imagine using simulation to repeatedly create synthetic samples (and corresponding sample statistics) under the assumption of the null hypothesis $H_0$ to understand the variability/uncertainty sample statistic (in the hypothetical situation that the null hypothesis $H_0$ is true).

As a quick example, if we wanted to test the assumption that a coin is fair, our hypotheses would be

- $H_0$: The coin is fair (i.e., the probability of heads is 0.5).
- $H_1$: $H_0$ is false: The coin is not fair (i.e., the probability of heads is not 0.5).

Then there are three characteristics of an experiment simulating the sampling distribution under $H_0$.

- `n_coinflips_per_sample`: the sample size (i.e., the number of coin flips that we're considering to use as the evidence regarding the null hypothesis)
- `n_sample_simulations`: the number of samples to simulate (i.e., the number of times to flip a coin `n_coinflips_per_sample ` times)
- `H0_p = 0.5`: the hypothesized value of the parameter specified by the null hypothesis; namely, $H_0: p = 0.5$ (where $p$ denotes the probability of the coin landing on heads)

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
    n_coinflips = np.random.choice(['heads','tails'], p=[H0_p,1-H0_p], size=n_coinflips_per_sample)
    simulated_proportion_coinflips_heads[i] = (n_coinflips=='heads').mean()
    # or just use `proportion_coinflips_heads[i] = 
    #              np.random.choice([0,1], p=[H0_p,1-H0_p], size=n_coinflips_per_sample).mean()`

# or instead of the `for` loop, this could be done as
# simulated_coinflips = stats.binom(n=1, p=0.5, size=(n_samples, n_coinflips_per_sample))
# simulated_proportion_coinflips_heads = simulated_coinflips(axis=1)

# Create the histogram using plotly.express
fig = px.histogram(simulated_proportion_coinflips_heads, nbins=30, 
                   title='Sampling Distribution of Proportion of Heads (Fair Coin)',
                   labels={'value': 'Proportion of Heads', 'count': 'Frequency'})
fig.update_traces(name='Simulated Proportions')
fig.show()
```

#### The role of Sample Size n

The exercise "creating the sampling distribution of the proportion of heads when a coin is fair" above may at first seem somewhat strange and counter intuitive. Why are we running simulations at all? Why don't we just flip the coin a really large number of times to find out if it's fair or not? In fact, the `stats.binom(n=1, p=0.5, size=(n_samples, n_coinflips_per_sample))` formulation even shows that this is exactly what we're doing, and all that we've really done is to just organize the coin flips into simulations (on the rows) with individual coin flips (in the columns on a row)...

Think of the "fair coin" example as just being an example of the template that we have when we collect data. We collect some number of data points (with a sample sample size of n), and then we can calculate a statistic using that sample.  The question now is, what is the variability/uncertainty of that sample statistic? And that (of course) depends on the sample size n. The statistic can then be used to provide inference regarding the population parameter it correspondingly estimates, and the strength of the evidence it provides depends on the variability/uncertainty of the statistic (which in turn depends on the sample size n used to construct the statistic).

> This general concept was previously discussed in terms of the [standard error versus the standard deviation](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#How-n-Drives-Standard-Error) in the context of confidence intervals; and, the standard deviation of the sampling distribution (of a statistic) under the null exactly addresses the same topic; namely, there is a **standard error** of a statistic under the assumption of the null hypothesis which captures the variability/uncertainty of the statistic and therefore subsequently influences the strength of evidence that the data will be able to provide regarding the null hypothesis $H_0$.


#### "One Sample" Paired Difference Hypothesis Tests with a "no effect" Null

In contexts with two samples where the observations of interest are "paired difference" (such as in a "before versus after" treatment intervention) the following two null hypotheses can be considered.

- $H_0: \mu = 0$ [there is "no effect" of the treatment intervention: the change in the population average $\mu$ is $0$]
- $H_0: p = 0.5$ [changes due to treatment intervention are "random": the chance of improvement $p$ is 50/50"]

The former null hypothesis results in what's known as a "paired t-test" but this hypothesis test is beyond the scope of STA130 so we will not consider this null hypothesis further. The latter null hypothesis can be seen to have the same template format as the "coin flipping" hypothesis testing discussed above.

- The sample pairs are compared and the proportion "greater than" is the observed statistic
- The sampling distribution under the null is then simulated with a "fair" coin flipping exercise

> If you cannot complete the exercise of leveraging the "coin flipping code" to do a "paired difference" hypothesis test, you should immediately seek help from a ChatBot, office hours, or study buddies to understand why "coin flipping" and "paired difference" hypothesis test are "equivalent"


### p-values

A **p-value** is **the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true**.

A p-value is a measure of the strength of the evidence against $H_0$. The smaller the p-value the stronger the evidence is against the null hypothesis. But why is this so? A p-value is created by comparing the observed sample statistic against the sampling distribution of the statistic under the assumption that the null hypothesis is true.  If the observed statistic does not seem plausible relative to the sampling distribution of the statistic under the assumption that the null hypothesis is true, then the evidence in the sample data does not agree with the sampling distribution of the statistic under the assumption that the null hypothesis is true. Therefore, the sample data provides evidence against the sampling distribution of the statistic under the assumption that the null hypothesis is true. Therefore, the sample data provides evidence against the assumption that the null hypothesis is true.

To give a measure of how much the data does not agree with the sampling distribution of the statistic under the assumption that the null hypothesis is true, we compute the p-value. That is, we find out the proportion of the statistics sampled from the sampling distribution of the statistic under the assumption that the null hypothesis is true that look "as or more extreme" than the observed statistic calculated from the sample data. The "as or more extreme" consideration requires us to calculate the relative differences of the observed statistic with the parameter valued specified by the null hypothesis, and compare these with the analogously calculated relative differences of the statistics sampled from the sampling distribution of the statistic under the assumption that the null hypothesis is true (with the parameter valued specified by the null hypothesis).

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

To ensure the correct and effective use of p-values, a *numerical* p-value actually needs to be converted to a "*qualitative*" result. Practically speaking, decisions need to be made in a straightforward and clear manner. Namely, we will either 

- **fail to reject the null hypothesis based on the evidence at hand**
- **or reject the null hypothesis based on some strength of the available evidence**

The following table is how we can characterize the "strength of evidence" we have against a null hypothesis.

|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|

A common approach to **formal hypothesis testing** is closely related to this table. Namely, a certain **significance level** (commonly $\alpha = 0.05$) threshold is set, and we then **reject the null hypothesis at the $\alpha$-significance level** if the p-value is less than the chosen $\alpha$ threshold.

- **IMPORTANTLY, NOTE that** for this approach to be valid, the $\alpha$-significance level must be chosen *before* the p-value is computed. 

The benefit of this table is that you don't need to bother with setting an $\alpha$-significance level threshold to compare the p-value against. This table instead just sets a *collection* of $\alpha$-significance level thresholds that we'll eventually (always plan to) compare the p-value to. And since we've all agreed to this now, there's no "funny business" of potentially changing the $\alpha$-significance level AFTER we've observed the data and calculated a p-value. 


#### The relationship between p-values and confidence intervals

If you noticed that the numeric aspect of the $\alpha=0.05$-significance level is "one minus" the numeric aspect of a 95% confidence level of a 95% confidence interval, good job. This is not a coincidence.  Recall the (correct) interpretation of a 95% confidence interval, and consider this against the interpretation (correctly given below) of an $\alpha=0.05$-significance level.

- 95% confidence claims that 95% of independently and identically distributed (i.i.d.) samples (at the sample size or our actually observed sample) would result in a constructed confidence interval that would in fact "capture" the actual true population parameter value it estimates; and, relevantly, 5% of these i.i.d. samples would result in a constructed confidence interval that would in fact "not capture"  it.
    - Also remember that once the confidence interval is constructed, it not longer has any associated probability as it either simply DOES or DOES NOT "capture" the actually true population parameter.
- Rejecting a hypothesis at the $\alpha=0.05$-significance means that 5% of independently and identically distributed (i.i.d.) samples (at the sample size or our actually observed sample) would incorrectly reject the null hypothesis when in fact it was true.
    - This means that the formal hypothesis testing process of reject a null hypothesis using a $\alpha=0.05$-significance level threshold will wrongly reject a true null hypothesis in 5% of i.i.d. samples (for which the null hypothesis is in fact true); which, you'll now be able to appreciate, is equivalent to how 5% of i.i.d. samples for which a 95% confidence interval construction procedure is applied with result in a "failing" interval which "does not capture" the actual true population parameter value.

This is why we can use confidence intervals to perform hypothesis testing without using a p-value. Both procedures offer equivalent levels of "*reliability*" in the conclusions which they draw about a null hypothesized parameter.  With $\alpha=0.05$-significance level testing, 5% of the i.i.d. samples we draw will wrongly reject the null hypothesis at the $\alpha=0.05$ threshold. While with a 95% confidence interval, 5% of the i.i.d. samples we draw will result in a constructed confidence interval at the 95% confidence level which "fail to capture" the actual true population parameter value.  

- But let's make this easy. Just use the "strength of evidence" table above when interpreting p-values if you're making a decision about a null hypothesis specifying a hypothesized parameter value; or, just construct a (95%?) confidence interval, and see if the hypothesized parameter value is "contained" in the constructed interval and make a decision based on that (knowing the intended **coverage rate** guarantees of the interval construction procedure). 
- And note that the benefit of using the confidence interval approach is that it provides more than just a **point estimate** of the actual true population parameter value, or a degree of evidence against a hypothesized value of the population parameter; rather, it gives us **inference** about the plausible range of values for the actual true population parameter value for which there is evidence of in our data.
- So yeah... let's just use confidence intervals, okay? They provide both **inference** about what the actual true population parameter value might be based on the data at hand AND this can be immediately used for **decision making** just like formal hypothesis testing.


#### Misusing p-values

Recalling that a 95% confidence level indicates the reliability of our procedure AND DOES NOT refer to any specific confidence interval upon construction (which only either DOES or DOES NOT "capture" the actual true population parameter value), you will remember that we must take care with how we speak about confidence intervals. Therefore, the following statements are technically correct and hence allowed. 

- This is a 95% confidence interval.
- I have 95% confidence this constructed interval captures the actual true population parameter value.
- I have used a confidence interval procedure which will "work" for 95% of hypothetical i.i.d. samples.
- There's a 95% chance this confidence interval "worked" and does "capture" the actual true population parameter value.

The following therefore TECHNICALLY INCORRECT and ARE NOT allowed.

- There's a 95% chance the parameter is in this confidence interval. <-- NOPE, sounds too much like we're saying parameters have "chance", but parameters don't have "a chance" of being "this or that".
- There's a 95% probability the parameter is in this confidence interval. <-- NOPE, sounds too much like we're saying parameters have "probability", but parameters don't behave "probabilistically".

Sorry to be full grammar nazi on this point. But we've gotta. Otherwise other statisticians will talk badly about you behind your back. And we can't have that, now can we?

But what does this have to do with misusing p-values? This is just about confidence interval terminology and meaning, isn't it? Well, sure, but we've already seen the direct connection between hypothesis testing using $\alpha$-significance level thresholds, so it should not come as a surprise to you that, just as with confidence intervals, p-values can be misinterpreted. Now, the way to avoid misusing and incorrectly misinterpreting p-values is to simply use p-values in one of the three appropriate manners.

1. Remember that **a p-value** is the probability that a statistic is as or more extreme than the observed statistic if the null hypothesis is true**, and that's it. A p-value can NEVER EVER mean anything other than this. 
2. Forget the definition of p-values and just interpret them in terms of the "strength of evidence" they provide against the null hypothesis based on the ranges given by the table above in the section about [using p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#using-p-values). This interpretation does not contradict the definition of a p-value given above; rather, it correctly interprets the meaning of the definition of a p-value in terms of the "strength of evidence" it indicates the data provides against the null hypothesis.
3. Simply compare the p-value to the pre-determined $\alpha$-significance level threshold and either fail to reject the null hypothesis at this pre-determined $\alpha$-significance level if the p-value is larger than this threshold, or indeed, if the p-value is smaller than this threshold, reject the null hypothesis at the $\alpha$-significance level. Remember to NEVER EVER change the $\alpha$-significance level threshold under consideration. This must be pre-determined and chosen before the sample is observed and the p-value is computed.  Otherwise you're engaging in "funny business" which will lead to an incorrect assessment of the actual information contained in the numeric p-value with respect to the null hypothesis. That's why the "strength of evidence" table provided in the [using p-values](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#using-p-values) section above is so useful. It's always pre-defined and you can always just refer to it and use it once the sample has been observed and a p-value calculated.

So, with these correct ways to use a p-value stated, it seems like by just following these guidelines for using p-values, we should never have a problem using p-values correctly. This is true. However, here are mistakes that ARE STILL MADE even despite all these clear warnings and instructions... PLEASE, for the love of GOD and all that is holy and pure, DO NOT MAKE THESE ABSOLUTE ABOMINATION TRAVESTY BLUNDERS OF THE FOLLOWING FUBAR INTERPRETATIONS OF p-values.

1. A p-value is the probability the null hypothesis is true. <-- OMG. NO. THIS IS NOT TRUE.
2. A p-value is the chance we wrongly reject the null hypothesis. <-- What? NO. That's the $\alpha$-significance level. Why are you confusing this with a p-value??
3. A p-value is the probability that the hypothesized parameter value is correct. <-- Omg Jesus Christ kill me. We JUST finished talking about WHY we're so careful about the way we talk about confidence intervals; because, parameters don't have "chances" or "probabilities"...

So, anyway, please, please, please, please, please, I'm begging you, please do not make any of these absolute abomination travesty blunders of these UTTERLY FUBAR interpretations of p-values. Please. Why mess up like this? It's so clearly stated here why you do not need to mess this up. 


#### One- versus Two-Sided Hypothesis Tests

Also known as one-tailed hypothesis tests (and not to be confused with the distinction between "one sample" or "two sample" situations), a variation on the traditional "equality" null hypothesis (such as $H_0: p = 0.5$) which allows us to address comparisons in an ordered manner replaces the two-sided (or two-tailed) alternative hypothesis (such as $H_A: p \neq 0.5$) with one-sided (or one-tailed) alternatives; namely...

$$\Large H_A: p > 0.5 \quad\textrm{ or }\quad H_A: p < 0.5 \quad\textrm{ which then correspond to }\quad H_0: p \leq 0.5 \quad\textrm{ or }\quad H_0: p \geq 0.5$$

Naturally, the (now familiar) $H_0$-centric form remains preferred, so these correspond to 

$$\Large H_0: p \leq 0.5 \textrm{ and } H_A: H_0 \textrm{ is false} \quad\textrm{ or }\quad H_0: p \geq 0.5 \textrm{ and } H_A: H_0 \textrm{ is false}$$

Using a one-sided (one-tailed) hypothesis test allows us to address questions of directional difference as opposed to simply "no difference". For example, this allows us to address the question of whether an unfair coin is bias in favor "heads" or against heads, depending on our choice of null and alternative hypothesis pair.  Depending on the research question, we may indeed be interested in giving evidence against a null hypothesis in a manner that takes into affect a single comparative direction, as opposed to "a difference in either direction". 

In addition, by using a one-sided test that is specifically interested in detecting an difference in one direction only, the p-value will necessarily become smaller since "as or more extreme" is now only considered in a single tail of the sampling sampling distribution of the statistic. 


## Lecture: New Topics


### Type I and Type II errors


**Type I** and **Type II errors** are key concepts in **formal (statistical) hypothesis testing**. But they have immediate generalizations to **machine learning** and **classification**-oriented **predicted modelling**, as we shall see later in the course. 

A **Type I Error** (or **False Positive**) occurs when we *reject* a null hypothesis that is actually *true*.

- In other words, a Type I error is when we conclude that there is an effect or a difference when, in reality, none exists.
- For example, if a medical test incorrectly shows that a patient has a disease when they actually don’t, this is a Type I error relative to the (most naturally obvious) null hypothesis that a patient does not have a disease. 
- The probability of making a Type I error is denoted by "alpha" $\alpha$ corresponding to the $\alpha$-significance level threshold of the hypothesis test.
    - It is customary to use an $\alpha=0.05$-significance level threshold so that there's only a 5% (1 in 20) chance that we've made a Type I error when we reject a null hypothesis (meaning that we've wrongly rejected a null hypothesis that is in fact true). 

A **Type II Error** (or **False Negative**) occurs when we *fail to reject* a null hypothesis that is actually *false*.

- In other words, a Type II error is when we conclude that there is no effect or difference present when there actually is one.
- For example, if a medical test incorrectly shows that a patient does not have a disease when they actually do, this is a Type II error relative to the (most naturally obvious) null hypothesis that a patient does not have a disease. 
- The probability of making a Type II error is denoted by "beta" $\beta$ (to associated it with "alpha" $\alpha$) and determining $\beta$  is the objective of so-called "sample size power analyses" (but these themselves are beyond the scope of STA130)
    - Even if "sample size power analyses" are beyond the scope of STA130, we should still be able to sensibly talk about and understand the notion of a "beta" $\beta$ the probability of making a Type II error 

| Decision       | Null Hypothesis is True   | Null Hypothesis is False |
|----------------|---------------------------|--------------------------|
| Reject Null    | **Type I Error (with chance α)**       | Correct Decision         |
| Fail to Reject | Correct Decision           | **Type II Error (with chance β)**     |

With this in mind, and recalling the previous initial conversation regarding [Null and Alternative Hypotheses](https://github.com/pointOfive/stat130chat130/wiki/week-05-Hypothesis-Testing#Null-and-Alternative-Hypotheses), please abide by the following factual realities of the formal hypothesis testing framework based on null hypotheses.

- We only provide some degree of "strength of evidence" against the null hypothesis based on the data at hand. 
- On the basis of the available "strength of evidence" indicated by the data, we may choose to reject a null hypothesis, and we may do so wrongly if in fact (despite the "strength of evidence" against it) the null hypothesis is in fact true.
    - When this situation occurs we commit a Type I error by our incorrect choice to reject the null hypothesis.
- On the basis of a "lack of strength of evidence" against a null indicated by the available data, we may choose to refrain from rejecting a null hypothesis, and we may do so wrongly if in fact (despite the "lack of strength of evidence" against it) the null hypothesis is in fact false.
    - When this situation occurs we commit a Type II error by our incorrect choice to refrain from rejecting the null hypothesis.
- We never prove the null hypothesis, and we never prove the alternative hypothesis. We only give evidence or fail to give evidence against the null hypothesis. 
- We therefore additionally never say that "we accept" a null hypothesis or an alternative hypothesis.  We only may choose to reject a null hypothesis with a certain "strength of evidence" and therefore prefer the alternative hypothesis, or fail to do so on the basis of the evidence of the available data at hand. 


So in the context of statistical hypothesis testing, we do not "prove $H_0$ is false"; rather, we instead give evidence against the $H_0$. We can therefore say something like, "We reject the null hypothesis with a p-value of abc, meaning we have xyz evidence against the null hypothesis". And analogously, we do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$; rather, we would instead say something like, "We fail to reject the null hypothesis with a p-value of abc".

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