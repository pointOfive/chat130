
# Course Textbook: Week 05 Hypothesis Testing

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
    
2\. Only do **hypothesis testing** on the basis of confidence intervals, not **p-values** (which might be the best solution wherever doing so is a realistic, convenient possibility).# STA130 TUT 05 (Oct04)<br><br>🤔❓ <u>"Single Sample" Hypothesis Testing<u>


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
# (Statistical) Hypothesis Testing

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
# STA130 Homework 05 

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
    
1. Code and write all your answers (for both the "Pre-lecture" and "Post-lecture" HW) in a python notebook (in code and markdown cells) 
    
> It is *suggested but not mandatory* that you complete the "Pre-lecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
    
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
- [0.3 points]: Evaluation of written communication for "Question 2"
- [0.3 points]: Evaluation of written communication for "Question 4"
- [0.3 points]: Evalution of submission for "Question 8"


## "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]

### A. Watch this first pre-lecture video ("Hypothesis testing. Null vs alternative") addressing the question "What is a hypothesis?"<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _The video gives the example that the "Hillary versus Trump" U.S. presidential election campaign could not be turned into a hypothesis test assessing differences in performance between the two as U.S. presidents (because at the time of the election neither had been U.S. presidents). This is different than addressing "Obama versus Bush" within a hypothesis testing framework (because we have eight years worth of performance of both as U.S. presidents). A more contemporarily relevant comparison then would be the aborted election campaign efforts of "Biden versus Trump", which would have been a chimeric hybrid of the two comparisons mentioned above (because we have BOTH four years worth of DATA regarding the performance of both as U.S. presidents BUT we are also likely still [...or, were, prior to Biden dropping out of the presidential race...] interested in asking questions regarding their potential FUTURE performance of both as U.S. presidents for which we do not yet have any data). Anway, despite Biden dropping out of the election, we might still attempt to consider the record of the Biden presidency to be informative and predictive about the furture peformance of a potential Kamala Harris presidency._
> 
> _This hopefully (a) makes the examples of the video more contemporarily relevant, and (b) gives another example to further emphasize and contrast the distinction that's being made in the video._
>
> _Also, while these are relatively knit-picky, two technical issues that the video somewhat inaccurately introduces are:_
>
> - _the video states that "we accept the null hypothesis"; but, actually it would be more correct to say, "we fail to reject the null hypothesis"_
> - _the video specifies "less than" for the null hypothesis and "less than or equal" for the alternative hypothesis; but, actually, for mathematic reasons "less than or equal" version is the more technically correct choice for how the null hypothesis should be specified_
    
</details>


```python
from IPython.display import YouTubeVideo
# First pre-lecture video: 
# "Hypothesis testing. Null vs alternative
# https://www.youtube.com/watch?v=ZzeXCKd5a18
YouTubeVideo('ZzeXCKd5a18', width=800, height=500)
```

### B. Watch this second pre-lecture video ("What is a p-value") providing an intuitivie introduction to p-values <br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _This is intended to help you initially make more sense of the technical definition of a p-value, **the probability that a test statistic is as or more extreme than the observed test statistic if the null hypothesis was true**._
>
> _The thing is though, once you understand a p-value is, then you'll see how simple and elegant the above definition is... So, your objective in learning what a p-value is should be to be able to read and understand the definition of a p-value given above effortlessly... That way you can communicate with the language of statistical reasoning in 3.5 seconds rather than 3.5 minutes..._
    
</details>


```python
from IPython.display import YouTubeVideo
# Second pre-lecture video
# "What is a p-value"
# https://www.youtube.com/watch?v=9jW9G8MO4PQ
YouTubeVideo('9jW9G8MO4PQ', width=800, height=500)
```

### 1. The "first pre-lecture video" (above) describes hypothesis testing as addressing "an idea that can be tested", and the end of the video then discusses what our actual intended purpose in setting up a null hypothesis is. What is the key factor that makes the difference between ideas that can, and cannot be examined and tested statistically?  What would you describe is the key "criteria" defining what a good null hypothesis is? And what is the difference between a null hypothesis and an alternative hypothesis in the context of hypothesis testing? Answer these questions with concise explanations in your own words.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _After watching and understanding both of the videos above, you should be well equipped to answer this and the following few questions. But, you can also interact with your favourite ChatBot to see if you understand the concepts correctly and to clarify any open questions you might have about anything that still seems unclear._
>
> HOWEVER, as we increasingly tread into the more conceptual statistical concepts of STA130, "vanilla" ChatBots become less and less reilable.
> 1. First, "vanilla" ChatBots don't know the constraints and scope of the learning objectives of STA130, so in addition to their often verbose nature, they now present the possible risk of tangenting onto topics that do not concern (but may nonetheless potentially confuse and distract) us
> 2. Second, ChatBots are based on textual information online, and while much of this information is accurate and well articulated, there is also a not insignificant presense of confusion and misunderstanding of statistical concepts and topics online. The downside of this is that since ChatBots don't "reasons" but instead just actually "regurgitate" the freqently occuring patterns between words found in text, it's increasingly possible that responses ChatBots will in fact amount to only meaningless gibberish nonensense.
>
> **Therefore, it is recommended that students begin considering and exploring increasingly relying on the STA130 Custom NotebookLM (NBLM) ChatBot** rather than "vanilla" ChatBots when it comes to the specific and technical and conceptual statistical topics of STA130.**
>
> _Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT) if you are using a ChatBot interaction to help learn and understand something!_
>
> In the case of the custom NBLM ChatBot, you can't get a transcript of your conversation, unfortunately (since converational history records outside of an active NBLM ChatBot session are made available to you in the future...); but, that's perfectly fine regarding the requirement of the homework which is only that a summary of any ChatBot interactions is provided with the submission. 

</details>


### 2. Towards the end of the "first pre-lecture" video (above) it is stated that, "It is important to note that outcomes of tests refer to the population parameter, rather than the sample statistic! As such, the result that we get is for the population." In terms of the distinctions between the concepts of $x_i\!$'s, $\bar x$, $\mu$, and $\mu_0$, how would you describe what the sentence above means? Explain this concisely in your own words for a "non-statsitical" audience, defining the technical statistical terminology you use in your answer.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _A formal **null hypothesis** has the form $H_0: \mu=\mu_0$ which states that the average value $\mu$ of the population is $\mu_0$, while the **alternative hypothesis** would then be $H_A: H_0 \text{ is false}$ which states the average value $\mu$ of the population is not $\mu_0$. This question asks for a clear explanation of the distinguishing characteristics of the between the concepts of observed sample values $x_i$ (for $i = 1, \cdots, n$), the observed sample average $\bar x$, the actual value of $\mu$, and the value $\mu_0$ hypothesized under the null hypothesis relative to hypothesis testing._   
> 
> _This question extends "Question 7" from the Week 4 HW that you considered last week in a more formal manner in terms of hypothesis testing notation. It should be getting much easier to delineate the differences between parameters and populations, and samples and statistics; and, to understand how to interpret and apply these concepts in the context of new topics (such as hypothesis testing, as is done here)._ 
> 
> _As continually suggested and encouraged regarding the topics of parameters, populations, samples, and statistics, check with your notes or your favourite ChatBot to make sure you have a clear understanding of these terms. At this point in the course, you should be able to read and understand of the meaning of the termenologically dense sentence addressed in the prompt to this question!_ 
>
> Don't forget to ask for summaries of your ChatBot session(s) if you are using a ChatBot interaction to help learn and understand something! You only need to include link(s) to chat log histories if you're using ChatGPT, e.g., if you're using the custom STA130 NBLM ChatBot you can't get chat history logs, but you can get summaries, so just paste these into your homework notebook and indicate the both you're using if it can't provide links to chat log histories.
    
    

</details>


### 3. The second "Pre-lecture" video (above) explains that we "imagine a world where the null hypothesis is true" when calculating a p-value? Explain why this is in your own words in a way that makes the most sense to you.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Hint: your answer will likely be most efficiently correct and clear if it discusses the relavence of the sampling distribution of the test statistic under the null hypothesis._
    
</details>

### 4. The second "Pre-lecture" video (above) describes suggest that a smaller p-value makes the null hypothesis look more ridiculous. Explain why this is in your own words in a way that makes the most sense to you, clarifying the meaning of any technical statistical terminology you use in your answer.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Hint: your answer will likely be most efficiently correct and clear if it discusses how the observed test statistic relates to the sampling distribution of the test statistic under the null hypothesis._
    
</details> 

### 5. Güntürkün (2003) recorded how kissing couples tilt their heads. 80 out of 124 couples, or 64.5% tilted their heads to the right. Simulate a **p-value** using a "50/50 coin-flipping" model for the assumption of the **null hypothesis** $H_0$ that the population of humans don't have left or right head tilt tendencies when kissing, and use the table below to determine the level of evidence we have against $H_0$. <br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _The previous three "Questions 2-4" are highly relevant here. For this question, you need to first (along the lines of "Question 2") understand what the problem context describes to you in terms of something analogous to $x_i\!$'s, $\bar x$, $\mu$, and $\mu_0$. Then you need to (along the lines of "Question 3") figure out how to "imagine a world where the null hypothesis is true" so that you can go about computing a (**simulation** based) p-value calcuation for the null hypothesis under consideration relative to the available data. And finally, you need to make a determination about your potential decision to reject the null hypothesis on the strength of the data at hand on the basis of the "strength of evidence" table given below (which indeed supports the necessary interpretation required to provide an explanation answering "Question 4")._
>    
> _Regarding Güntürkün (2003) itself, click [here](https://www.nature.com/articles/news030210-7) if you want to know more!_    
    
</details> 


|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|

![Rodin's sculpture, "The Kiss"
](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Rodin_-_Le_Baiser_06.jpg/409px-Rodin_-_Le_Baiser_06.jpg)


<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Pre-lecture VS Post-lecture HW

Feel free to work on the \"Postlecture\" HW below if you're making good progress and want to continue: the next questions will continue addressing and building on the topics from the videos, so, it's just a choice whether or not you want to work a head a little bit...
    
*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details>    

## "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]

### 6. Can a smaller p-value definitively prove that the null hypothesis is false? Is it possible to definitively prove that Fido (from the "second pre-lecture video") is innocent using a p-value? Is it possible to difinitively prove that Fido is guilty using a p-value? How low or high does a p-value have to be to definitely prove one or the other? Explain this concisely in your own words.<br>

### 7. In the second half of the "first pre-lecture video" the concept of a "one sided" (or "one tailed") test is introduced in contrast to a "two sided" (or "two tailed") test. Work with a ChatBot to adjust the code from "Demo II of  the Week 5 TUT" (which revisits the "Vaccine Data Analysis Assignment" from Week 04 HW "Question 8") in order to compute a p-value for a "one sided" (or "one tailed") hypothesis test rather than the "two sided" (or "two tailed") version it provides. Describe (perhaps with the help of your ChatBot) what changed in the code; how this changes the interpretation of the hypothesis test; and whether or not we should indeed expect the p-value to be smaller in the "one tailed" versus "two tailed" analysis. <br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _[Demo II of the The Week 5 TUT](https://github.com/pointOfive/stat130chat130/blob/main/TUT/STA130F24_TUT05_Oct04.ipynb) revisiting the "[Vaccine Data Analysis Assignment](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW04_DueOct03.ipynb)" illustrates using simulation to estimate a two-sided (or "two tailed") p-value._
>
> _The notion of "one sided" or "two sided" tests is also referred to as "one tailed" or "two tailed" because (other than using "$\leq$" and "$>$" [or "$\geq$" and "$<$"] rather than "$=$" and "$\neq$" when specifying $H_0$ and $H_A$) the actual place where this distinction has a practical impact is in the calculation of p-values, which is done in the "tails" of the sampling distribution of the statistic of interest under the assumption that the null hypothesis is true._
>
> Don't forget to ask for summaries of your ChatBot session(s) if you are using a ChatBot interaction to help learn and understand something! You only need to include link(s) to chat log histories if you're using ChatGPT, e.g., if you're using the custom STA130 NBLM ChatBot you can't get chat history logs, but you can get summaries, so just paste these into your homework notebook and indicate the both you're using if it can't provide links to chat log histories.

</details>

### 8. Complete the following assignment. 

### Fisher's Tea Experiment

**Overview**

A most beloved piece of [statistical lore](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2012.00620.x) about the (most famous) statistician Ronald Fisher involves cups of tea with milk. Fisher and his friend and colleague, Dr. Muriel Bristol, worked at Cambridge in the 1920s and regularly had tea together. During one of their afternoon tea times, Bristol refused a cup of tea from Fisher because he put milk in first BEFORE pouring in the tea. Bristol said she could taste the difference, and much preferred the taste of tea when the milk was poured in afterward the tea. Fisher didn't think that there could be a difference and proposed a hypothesis test to examine the situation.

Fisher made 8 cups of tea, 4 with milk added in first and 4 with tea added in first, and gave them to Dr. Bristol without her seeing how they were made and she would say if she thought the tea or the milk was poured first. As it turned out, Bristol correctly identified if the tea or milk was poured first for all 8 of the cups. Fisher, being a skeptical statistician wanted to test if this could be happening by chance with Bristol just randomly guessing (or whether there was evidence against an assumption of Bristol just randomly guessing), and subsequently designed a statistical hypothesis test to do so.

Suppose you run an experiment like this with students in STA130. You get a random sample of 80 STA130 students to each taste one cup of tea and tell you whether they think the milk or tea was poured first. **Suppose 49 students are able to correctly state which was poured first.** Provide a statistical analysis of this experiment as guided through the following set of questions.

**Data**

49 out of a sample of 80 students are able to correctly state which was poured first.

**Deliverables**

While you can choose how to approach the project, we are interested in evaluating your report relative to the following deliverables: 
- Clarity of your documentation, code, and written report 
- Description of the population (and sample) and parameter of interest (and corresponding observed test statistic) 
- Formal null hypotheses $H_0$ 
    - Provide a formal version $H_0$ based on the population parameter 
    - Provide an informal interpretive statement explaining $H_0$ in more casual everyday common language
    - Alternative hypothesis $H_A$ in terms of $H_0$
- Quantitative analysis addressing the validity of $H_0$
    - Explanation of the method clearly articulating the purpose of the usage of statistic(s) to address $H_0$ the population parameter of interest 


**Comments**

- Regarding the population (and the sample), there is a clear difference between the experiment with STA130 students considered here and the original motivating experimental context of Fisher and Bristol.
    - the sample size is different.
    - but so too is the nature of the population. the parameter in question might be considered more personalized in the original experiment; whereas, the parameter in the context of STA130 students might be a more abstract concept
- The analysis here could be approached from the perspective of formal hypothesis testing.
    - which would likely involve the simulation of a sampling distribution under $H_0$ in order to estimate p-value with respect to the null hypothesis based on the observed test statistic (how?), concluding with the assement of $H_0$ based on an interpretation of the meaning of the p-value relative to $H_0$
    - but a confidence interval approach to considering the hypothesis could also be considered.

> Consider organizing your report within the following outline template.
> - Problem Introduction 
>     - Relationship between this experiment and the original with Fisher and Bristol
>     - Statements of the Null Hypothesis and Alternative hypothesis
> - Quantitative Analysis
>     - Methodology Code and Explanations
>     - *(if needed/optional)* Supporting Visualizations 
> - Findings and Discussion
>     - Conclusion regarding the Null Hypothesis

#### Further Instructions:
- When using random functions, you should make your analysis reproducible by using the `np.random.seed()` function


### 9. Have you reviewed the course wiki-textbook and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
 
>  _Here is the link of [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) in case it gets lost among all the information you need to keep track of_  : )
>    
> _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
    
</details>

_**Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)!**_ **But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!**

# Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 

<br>
<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>
    
### Ethical Professionalism Considerations
    
Using p-values and hypothesis testing appropriately is an important ethical and professional responsibility of anyone doing data analysis. Actually, there is quite the quiet Contra-Versy (or is it Con-TROV-ersy?) around p-values. First, on a general level, it seems quite clear that p-values and hypothesis testing methodologies MUST play some ongoing contributing role in the so-called "replication crisis" rampantly afflicting mordern science; namely, "significant findings" made in scientific studies are not able to be reproduced by future studies at an alarming rate; and, this whole paradigm of "significant findings" is based on p-values and hypothesis testing... so, something's going on with this methodology in some way...
    
More specifically however, p-values are themselves quite problematic. To see this, just briefly consider the following article titles...

- [Why are p-values controversial?](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1277161) 
- [What a nerdy debate about p-values shows about science and how to fix it](https://www.vox.com/science-and-health/2017/7/31/16021654/p-values-statistical-significance-redefine-0005)
- [The reign of the p-value is over: what alternative analyses could we employ to fill the power vacuum?](https://royalsocietypublishing.org/doi/10.1098/rsbl.2019.0174)
- [Scientists rise up against statistical significance](https://www.nature.com/articles/d41586-019-00857-9)
- [Statistics experts urge scientists to rethink the p-value](https://www.spectrumnews.org/news/statistics-experts-urge-scientists-rethink-p-value)

While the issues here are relatively advanced and subtle (as introduced [here](https://www2.stat.duke.edu/~berger/p-values.html), presented [here](https://www.jarad.me/courses/stat587Eng/slides/Inference/I06-Pvalues/why_pvalues_dont_mean_what_you_think_they_mean.pdf), and demonstrated using simulation [here](https://jaradniemi.shinyapps.io/pvalue/)), the problem essentially comes down to the fact that most scientists (or just people) don't know how to really interpret the numeric value of a p-value. There are therefore two current proposed solutions to address this challenge.
    
1. Just interpreting p-values using the follwing table (which really isn't that hard, so it's surprising that this solution isn't more broadly adopted...)
    
|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
    

2. Only do **hypothesis testing** on the basis of confidence intervals, not **p-values** (which might be the best solution wherever doing so is a realistic, convenient  possibility...)

With this quite broad introductory context in mind, what does your favorite ChatBot thinks about the following statements? 
    
1. Hypothesis testing is not a "mathematical proof"<br><br>

    1. We do not prove $H_0$ false, we instead give evidence against the $H_0$: "We reject the null hypothesis with a p-value of XYZ, meaning we have ABC evidence against the null hypothesis"
    2. We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$: "We fail to reject the null hypothesis with a p-value of XYZ"<br><br>

2. Implying that a "non-significant result" means there is "no effect" misleads an audience because this may in actual fact simply indicate that there was insufficient evidence to reject the null hypothesis. So this therefore overlooks the possibility of sample size limitations, or Type II errors (which means a test incorrectly concludes that there is no effect or difference when, in fact, there is one). 
    
> Similarly, analagously, a "significant result" used to reject the null hypothsis could alternatively be a Type I error (which means a test actually incorrectly rejected a null hypothesis when it was actually true)... we're only providing a measure of evidence against the null hypothesis... but the evidence could still incorrectly suggest the wrong conclusion... it really depends on how strong the evidence is...
>
> - all of which is why just interpreting p-values using the table above is a good idea...

3. The p-values used for hypothesis testing are contructed upone the assumptions of the null hypotheses they correspond to; but, null hypotheses are actually often presented in simple forms that routinely hide a lot of information that is implicitly used to construct the p-values. For example, distributional assumptions about the population, estimated "plug-in" values that can used to simplify the problem calculations, and the reliance upon "random sampling", etc...<br><br>
           
4. Drawing overly broad conclusions, or making recommendations based on findings that reject the null hypothesis in a specific context is fraught with the problematic risks of overgeneralization errors. Further exacerbating this issue, null hypotheses are typically so called "point null hypotheses" which is meant to emphasize that they are mathematically increadibly sharply specific; whereas, alternative hypotheses are usually very unspecific. An alternative hypothesis that "the null hypothesis is false" doesn't say much... we should wonder, "how, specfically, is the null false?"
    
As an example really giving a demonstrating this, consider rejecting a null hypothesis that there is no correlation between rain and pizza's delivered. Such a decision doesn't specify what the actual hypothetical correlation might be. In fact, it doesn't even indicate if there are more or less pizzas delivered when it rains... 

> which, actually, shows very clearly why statistical inference using hypothesis testing is inferior to statistical inference based on confidence intervals...
> 
> - a confidence interval provides a range of plausible values of what the parameter in question might be; whereas, ...
> - trying to more clearly address what the plausible values of the parameter in question might be on the basis of hypothesis testing would require conducting further experiements to continously reject increasingly detailed hypothesies to narrow down what the alternative hypothesis might actually include... which would indeed be an utterly vapid misuse of the intended purpose of hypothesis testing entrprise... 
    
</details>

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>
    
### Current Course Project Capability Level
    
**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
    
> ### NEW DEVELOPMENT<br>New Abilities Achieved and New Levels Unlocked!!!    
> **As noted, the Week 01 HW introduced the [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb) notebook.** _And there it instructed students to explore the notebook through the first 16 cells of the notebook._ The following cell in that notebook (there marked as "run cell 17") is preceded by an introductory section titled, "**Now for some comparisons...**", _**and all material from that point on provides an example to allow you to start applying what you're learning about Hypothesis Testing to the CSCS data**_ as now suggested next below.

    
At this point in the course there should be two kinds of hypothesis testing analyses you should be able to use to provide evidence against a null hypothesis (about some of the interesting columns from the Canadian Social Connection Survey data):
    
1. Any "before and after" data that can be made into differences can be used to test a null hypothesis of "no effect" of an intervention on the average change in the population (as illustrated through the example of the Week 5 TUT **Demo**)
    
2. Any binary data that could be approached analagously to the "Stella's Wheel of Destiny" example of the Week 5 TUT **Communication Activity** can be used to test a null hypothesis about the (population) chance of success `p` (using a `np.random.choice([0,1], p)` population to simulate the sampling distribution under the null)
    
    1. [For Advanced Students Only] And actually, hypothesis testing for other numerical data could be approached analagously to the method based on assuming a distibution for the population (such as `stats.norm(loc=mu0, scale=x.std)` in place of `np.random.choice([0,1], p)`... if you see what this means?)
    2. Or it could be based on seeing if a hypothesized parameter value was contained within a bootstrapped confidence interval...
    

1. How do hypothesis testing analyses correspond to bootstrapped confidence intervals? 
    
2. Create a **null hypothesis** about a population parameter than you can test using the Canadian Social Connection Survey data

3. Carry out the hypothesis test using simulation, and interpret the result of the estimated p-value relative to the null hypothesis
    
</details>    


```python

```
