# ~~Normal Distributions~~ Now REGRESSION'S gettin' jiggy wit it

**Tutorial/Homework: Topics**

1. [Multiple Linear Regression](Weekz-10-Multiple-Linear-Regression#Multiple-Linear-Regression)
    1. [Interactions](Weekz-10-Multiple-Linear-Regression#Interactions)
    2. [Categoricals](Weekz-10-Multiple-Linear-Regression#Categoricals)
2. [Model Fitting](Weekz-10-Multiple-Linear-Regression#Interactions)
    1. [Evidence-based Model Building](Weekz-10-Multiple-Linear-Regression#evidence-based-model-building)
    2. [Performance-based Model Building](Weekz-10-Multiple-Linear-Regression#performance-based-model-building)
    3. [Complexity, Multicollinearity, and Generalizability](Weekz-10-Multiple-Linear-Regression#complexity-multicollinearity-and-generalizability)

**Tutorial/Homework/Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

1. [Logistic Regression](Weekz-10-Multiple-Linear-Regression#logistic-regression)
    1. [Categorical to Binary Cat2Bin Variables](Weekz-10-Multiple-Linear-Regression#Categorical-to-Binary-Cat2Bin-Variables)
1. [And Beyond](Weekz-10-Multiple-Linear-Regression#and-beyond)

**Lecture: New Topics**

1. I'm planning to just show you how I work on this kind of data with a pretty interesting example...

**Out of scope:**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...the deep mathematical details condition numbers, variance inflation factors, K-Folds Cross-Validation...
4. ...the actual deep details of log odds, link functions, generalized linear models...

## Tutorial/Homework: Topics

### Multiple Linear Regression

With $\epsilon_i \sim \mathcal{N}(0, \sigma)$ **simple linear regression** is 

$$Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ or } \quad Y_i = \beta_0 + \beta_1 1_{[\textrm{``some group''}]}(k_i) + \epsilon_i$$

and allows us to example **linear associations** or **group comparisons** for **independent samples**.

**Multiple linear regression** generalizes this to analyze how **multiple predictors** together predict an **outcome variable**.

$$Y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \epsilon_i \quad \text{ or } \quad Y_i = \beta_0 + \beta_1 1_{[\textrm{some{ }group{ }1}]}(k_{1i}) + \beta_1 1_{[\textrm{some{ }group{ }2}]}(k_{2i}) + \cdots + \epsilon_i$$
or 
$$Y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \beta_1 1_{[\textrm{some{ }group{ }1}]}(k_{1i}) + \beta_1 1_{[\textrm{some{ }group{ }2}]}(k_{2i}) + \cdots + \epsilon_i$$

#### Interactions

In the most classic **interaction** the **slope** "rise over run" relationship between the **outcome** $Y_i$ and **continuous predictor** $x_i$ change across **different groups** specified by an **indicator variable**.

$$Y_i = \beta_0 + \beta_1 x_{i} + \beta_2 1_{[\textrm{some{ }group{ }1}]}(k_i}) + \beta_3 x_{i} \times \beta_3 1_{[\textrm{some{ }group{ }1}]}(k_{i}) + \epsilon_i$$

It is also possible to have **interactions** between two different **continuous predictor variables** 

$$Y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \beta_3 x_{2i}\times\beta_3 x_{2i} + \epsilon_i$$

or even for the *same*  **continuous predictor variable** with *itself*!

$$Y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{1i}^2 + \epsilon_i$$

This latter "quadratic" specification allows the model to fit (quadratic) parabolas rather than straight lines, and the interpretation of the former is also relatively straight forward in terms of the nature of the so-called "hyper-planes" defined by the **multiple linear regression** model. 

#### Categoricals

When a **predictor variable** is **categorical** it can be represented by $K-1$ **binary indicator variables**.

$$Y_i = \beta_0 + \beta_2 1_{[\textrm{some{ }group{ }2}]}(k_i}) + \beta_2 1_{[\textrm{some{ }group{ }3}]}(k_i}) + \cdots + \beta_K 1_{[\textrm{some{ }group{ }K}]}(k_i})  + \epsilon_i$$

The reason for only needing $K-1$ **binary indicator variables** is that the **coefficients** of **binary indicator variables** (as you will recall) define so-called **contrast** (differences) between two groups. And in the case of a **categorical predictor variable**, this **contrast** will be relative to the so-called **baseline** group.  In the specification above this has been chosen to be "some group 1", but obviously the group index names here mean arbitrary group could be the **baseline** group and then all other group **coefficients** will provide "offets" relative to that. 

With some reflection, this shows that while **binary indicator variables** provide a means for **group comparison** (with associated **p-values** for  "no difference" **hypothesis testing** as well as **confidence interval** construction for **inference**), in the case of **categorical variables** the **group comparisons** that are actually performed are only those of the groups defined by the **non-baseline** category levels relative to the **baseline group**. 

2. [Model Fitting](Weekz-10-Multiple-Linear-Regression#Interactions)
    1. [Evidence-based Model Building](Weekz-10-Multiple-Linear-Regression#Evidence-based)
    2. [Performance-based Model Building]

### Model Fitting

See the [_statsmodels_ "formula" documentation](https://www.statsmodels.org/dev/example_formulas.html) for the "classic R-style" formula process of model 
1. specification `model = smf.ols(linear_form, data)`
2. fitting `model.fit()` and  
3. analysis `model.fit().summary()`

The extension here in the context **multiple linear regression** relative to **simple linear regression** is perfectly straight-forward. All **hypothesis testing** and **inference** tasks can be considered on one **coefficient parameter** at a time. **Hypothesis testing** in the **multiple linear regression** now simply addresses 

$$H_0: \beta_k = 0 \textrm{ within the context of the full model}$$

#### Evidence-based Model Building

The **p-values** and **confidence intervals**, for **hypothesis testing** and **inference** respectively, in **multiple linear regression** relay on the **assumptions** of the model specification. We will not restate these here now since they are *exactly* the same as those made for the **simple linear regression** model, with only caveat now being that "the assumption that the linear form specification is *true*" simply involves a *more complex* linear form. 

Based on the belief that the assumptions are at least "reasonably approximately correct" the **multiple linear regression** model can now either be

1. "built up" by adding (**continuous**, **indicator**, and **interaction**) **predictor variables** to the **linear form specification** based on a series of **coefficient hypothesis tests** which justify the inclusion of the **predictor variables** in the model based on the  evidence present in the data against the assumption that the **variables** *have no association* with the **outcome variable** in the context of the model (as judged by the consideration that their corresponding **slope coefficient** has a value of $0$ within the context of the model). 
2. "pruned down" by removing **predictor variables** from the **linear form specification** based on a **coefficient hypothesis tests** which fail provide sufficient evidence against the **null hypothesis** that $H_0: \beta_k = 0 \textrm{ within the context of the full model}$. 

These *evidence-based* approaches to analyzing the data in order to determine what the appropriate form of **linear specification** of the model should be represent the *classical* statistical approach to the "model building problem" ad are respectively referred to as **forward selection** and **backward selection**. 


#### Performance-based Model Building

The "more modern **machine learning**" way to determine what the most appropriate model specification might be is to use a so-called **train-test** framework.  The idea behind this approach is very simple; namely, if your model specification is good, then it will perform well when it **predicts** the value of **outcome variables** on new data that *it has never seen before*.  That is, a good model should be able to make predictions on new data that were not used to fit the model in the first place just as well (or at least not too much worse) than it is able to make predictions about the "old data" that was used to fit the model in the first place. The differences we are describing here in this **train-test** framework is that of **out of sample performance** versus **in-sample performance**. We should *expect* a model to be able to perform well predictively on the same data that was used to fit it in the first place; but, we should also expect that a good model would *additionally* be able to predict similarly well on future data. The notion that a model can predict well on future data is called **model generalization**.  If a model can only perform well on the data it was fit based on but cannot **generalize** well **out of sample** when considered from the perspective of a **train-test** framework then we say the model is **overfit**.  A model could also be **underfit** which means its predictive performance is subpar relative to what might be optimally possible from a prediction model.  But this clarifies that we want a model to be *just right*.  We don't want a model to be **overfit** so it's future predictions are no longer as accurate as its predictions of the data which we already have (which we don't actually in fact need any predictions for of course since we already have it). But we similarly don't want a model that is **overfit** and not particularly predictively performative relative to what might be readily achievable through an alternative model form specification. 

It will no doubt take the careful and attentive reader several passes to parse, review, and being to make sense of the previous discussion. But in the final analysis the **train-test** framework is extremely simple to execute.  The `train_test_split` functionality of THE **machine learning** library `sklearn` (or `scikit-learn`) allows us to create a so-called **training dataset** and retain a separate **testing dataset** subset.  The analyst is then free to try as many different model specifications as they wish, eventually inevitably determining that the specification with the best balance of **out of sample performance** and **interpretability of the model specification** for ease of communication and explanation, and **model simplicity** to promote better **out of sample generalization**. 

The latter point that **model simplicity** promote better **out of sample generalization** is a result of the fact that the **more complex** a model is, the more flexibility it has to find **idiosyncratic spurious (false random chance) associations** in the data that may accidentally (in a **Type 1 error** manner) end up cause the model to be **overfit** to relationships that are actually only the result of the random sampling variability nature of the data.

#### Complexity, Multicollinearity, and Generalizability

As introduced in the previous section, **model complexity** and **model generalizability** are on the whole *inversely related*.  This does not mean that by making a model *more complex* that it will necessarily be **less generalizable**. Of course increasing the complexity of the model in a way that improves the predictive power of the model is how we avoid **model underfitting**.  But what this means is that if a model is **too complex** relative to the amount of information that is actually available in the data, then the model will be **overfit** to the **training dataset**. That is, the model fit will include **spurious (not actually real) associations** that are just randomly idiosyncratically present in the data but that will not **generalize** to the **training dataset** or other future **out of sample** predictions. Again, this is because just a result of the randomness of sampling variation it's always possible to observe a relationship that's present in the **training dataset** but which is not actually real and only seen accidentally "randomly by chance". But what this means is that the model is actually in fact *not very good* since it can only make good predictions for data that we already have (and don't really need to make predictions for actually, since we already have it).

In the context of **multiple linear regression** this really indeed does exactly all come down to finding a model that is appropriately perfectly correctly suited to the amount of information that is actually available in the data (relative to estimating the model in question). There are two ways to easily understand what it would mean for a **multiple linear regression** specification to be **too complex** relative to the amount of information that is actually available in the data (relative to that specification).

1. Every **coefficient** in the model is a **parameter** that must be **estimated** using the data at hand: the more data you have the more **parameters** you would expect you'd therefore be able to estimate; whereas, the less data you have, the fewer the number of **parameters** you might expect you'd be able to estimate. 
2. When trying to estimate the relationship of **multiple predictor variables** to the **outcome variable** of the model, if the **predictor variables** all have *similar* relationships with the **outcome** then it will be harder to determine which **predictor** the "relationship" should be assigned to: this is called **multicollinearity** and it causes *increased uncertainty* in the estimation of the **coefficient parameters** and it therefore *weakens* the *statistical power* of the data to provide *evidence* and *inference* about such affected **coefficient parameters**.

**Multicollinearity** and related **model complexity** topics (such as **condition number** and **variance inflation factors**) are addressed and discussed in this weeks homework (Questions "6" and "7" and "9" and related questions therein), so see the homework for further details; but, something very interesting to note and consider to arrive at a very *clear* understanding of **multicollinearity** is that **multicollinearity** is much more of a problem for statistical **evidence** and **inference** than it is for **predictive generalizability**. This is because if multiple **predictor** variables all perform the same "predictive role" relative to the **outcomes variable** then it doesn't really matter which I use or how I use them (i.e., the specific balance of **fitted coefficient parameters** that are used for the **predictor variables** in question) for making **predictions** of my outcome variability. Therefore, for the purposes of **generalizable prediction**, the presence of **multicollinearity** need not necessarily be prohibitive. But if understanding of the **coefficient parameters** is desired, say for the purposes of **model interpretability** or providing **evidence** of association or **inference** regarding "effects", then the presence of **multicollinearity** can rabidly become excessively problematic. 

One final exercise for your attention within the context of **model complexity** considerations is to notice how a model's (**in sample**) $R^2$ "proportion of variation explained" cannot ever be reduced by adding additional **predictor variables** to a **model specification**.  This demonstrates why (and how) **model complexity** and **model generalizability** are indeed *inversely related*. Intuitively, for data at hand, by adding a new **predictor** we have only "gained" information, and we have not "lost" any of the information we had. It would therefore not really be possible to "all of a sudden do worse than we were" just because we added more information. So the model's (**in sample**) $R^2$ will minimally not change but can only otherwise increase. If you have a slightly uneasy feeling about this, such as perhaps wondering something like, "what if I could put a bad, predictor variable in an intentionally adversarial manner to sabotage this?", then good job. Indeed, *you could*. But this would not affect the **in sample** $R^2$ of a model; rather, it would deteriorate an *out of sample** $R^2$ calculated on the **predictions** of the model relative to new data not yet observed by the model.  In the context of a fitted model, ALL the data is available and is used to predict itself. So "bad variables" are "not bad" for the **in sample** $R^2$. They're just used in **spurious** ways to idiosyncratically make the **in sample** predictions for the **training dataset** currently at hand. This is why they then don't **generalize** and will instead result in worse future **out of sample predictions**. The issue here is an interesting one though, because while this IS NOT **multicollinearity**, it's just **model overcomplexity** it will nonetheless also most likely result in the presence of **multicollinearity**. Essentially, the more **predictor variables** there are the more **multicollinearity** there will probably be.  This actually helps make diagnosing the problem of **model complexity** quite easy. Just look at the number of **predictor variables** being used and the degree of **multicollinearity** (as judged but the **condition number** of a **multiple linear regression model** as discussed in the Homework for this week) present in the data. And be weary if either of these seem to be "large". 


## Tutorial/Homework/Lecture Extensions

### Logistic Regression

**Logistic regression** models allow the **outcome variable** to be **binary** rather than **continuous** as in **multiple linear regression**. To do this something called a **link function** is used which (in the case of **logistic regression**) transforms the **linear form** of the model in manner that allows it to be used as the probability prediction of the **binary outcome**.  This transformation means that the **coefficients** of the **logistic model** must now be interpreted on a so-called **log odds** scale which is quite different compared to how they are simply understood as "rise over run" **slope** or **contrast** group difference **parameters** (or any of the additionally increasingly sophisticated interpretations of **coefficient parameters** connected to **interactions** of the **predictor variables**).  But, this "increased complexity" of understanding and interpreting "the meaning" of **coefficient parameters** of **logistic regression** (compared to **multiple linear regression**) aside, the deployment and use of **logistic regression** is indeed a *quite minor* extension (requiring *very little* generalization) relative to the workflow of **multiple linear regression**. This is demonstrated below for **logistic regression** for our favourite Poke the Man dataset. 

```python
# Here's an example of how you can do this
import pandas as pd
import statsmodels.formula.api as smf

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
pokeaman = pd.read_csv(url).fillna('None')

pokeaman['str8fyre'] = (pokeaman['Type 1']=='Fire').astype(int)
linear_model_specification_formula = \
'str8fyre ~ Attack*Legendary + Defense*I(Q("Type 2")=="None") + C(Generation)'
# The "link function" for logistic regression is called the "logit" and hence the name of this function
log_reg_fit = smf.logit(linear_model_specification_formula, data=pokeaman).fit()
log_reg_fit.summary()
```

### Categorical to Binary Cat2Bin Variables

The **outcome variable** in the `smf.logit` functionality must literally be **binary** ($0$ or $1$).

- That is why the `pokeaman['str8fyre'] = (pokeaman['Type 1']=='Fire').astype(int)` code is used
- but this could always be used to analogously produce whatever group **binary subcategory indicator** was desired for use

The so-called "R style formula" syntax `I(Q("Type 2")=="None")` allows us to create a group indicator as part of the "R style formula", which can be convenient. Unfortunately you can't do this "on the fly for the outcome". 

- In general, though, considering one of the groups of a **categorical variable** in it's **binary indicator variable** form is always an option when working with **categorical data*
- and as you will quickly realize offers a much simplified approach to addressing and working with **categorical variables** that should be taken advantage of whenever possible to avoid the full complication of trying to fully grapple with MANY subcategories of a variable simultaneously.
    - In the case of **predictor variables** do you really want or need EVERY level of the **categorical variable** as a **predictor**? Are they all EVEN *statistically significant* so as to suggest their relevance for inclusion into the model?
    - And in the case of **outcome variables** it can be quickly seen that considering **binary outcomes** is MUST simple, and when you consider **categorical outcomes** that's actually in fact all you're doing but you're just having to do it all at once simultaneously which can indeed get complicated.

### And Beyond

**Logistic regression** is what is known as a **generalized linear model**. 
The point of **generalized linear models** is to allow us to mode other **outcome variables** besides **continuous outcome variables**. 
Like, for example, how **logistic regression** allows us to model **binary outcome variables**.
If you'd like to see how to model *even more* kinds of **outcome variables**, for example, if you were interested in modeling **categorical (single draw multinomial) outcome variables** (as I gather quite some number of most of you would like to) for the purposes of your **course project**, then have a look [here](https://www.statsmodels.org/dev/api.html#discrete-and-count-models) and what's available to you. It looks to me like you're going to be interested in `statsmodels.discrete.discrete_model.MNLogit`. I also found something called [ordinal regression](https://www.statsmodels.org/dev/examples/notebooks/generated/ordinal_regression.html) that looks quite promising for the course project...


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
# STA130 Homework 07

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
- [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 4"
- [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 7"
- [0.3 points]: Well-communicated, clear demonstration of the "model building" process and techniques of "Question 9"


## "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]


### 1. Explain succinctly in your own words (but working with a ChatBot if needed)...<br>

1. the difference between **Simple Linear Regression** and **Multiple Linear Regression**; and the benefit the latter provides over the former


2. the difference between using a **continuous variable** and an **indicator variable** in **Simple Linear Regression**; and these two **linear forms**


3. the change that happens in the behavior of the model (i.e., the expected nature of the data it models) when a single **indicator variable** is introduced alongside a **continuous variable** to create a **Multiple Linear Regression**; and these two **linear forms** (i.e., the **Simple Linear Regression** versus the **Multiple Linear Regression**)


4. the effect of adding an **interaction** between a **continuous** and an **indicator variable** in **Multiple Linear Regression** models; and this **linear form**


5. the behavior of a **Multiple Linear Regression** model (i.e., the expected nature of the data it models) based only on **indicator variables** derived from a **non-binary categorical variable**; this **linear form**; and the necessarily resulting **binary variable encodings** it utilizes
       
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _The requested **linear forms** are **equations**, and answers must include **equations** along with the explanations that interpret the **linear forms**. Write you **linear forms** using a style something like_
> 
> - _"outcome = $\beta_0$ + $\beta_A$ predictorA + $\beta_B$ 1(predictorB)"_ 
> - _where the "1(.)" notation is for indicator variables_ 
> - _or feel free to use an similar alternative if a ChatBot provides you with another notation you think is clearer and like better if you prefer_
>
> _DO INCLUDE the **intercept** in your **linear forms**. You don't have to include notation related to the **error** term since this is essentially always assumed (and, actually, we usually don't even bother to include the **intercept** in such shorthand specifications either, for the same reason), but don't forget to include the **intercept** here this time (for practice). The modeling **assumptions** do not need to be addressed beyond this, but explanations will likely address the number of variables and the essential use-case (perhaps illustrated through examples) the different models imply._    
> 
> _Answers to the final question above should address the notion of a "baseline" group and it's role for **model interpretation**, why "number of categories minus one" **indicator variables** are used to represent the original **categorical variable**, and the relationship between the **binary** and **categorical variables** that are relevant for this model specification. An example use-case would likely be helpful for illustration here._ 
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
    
</details> 

### 2. Explain in your own words (but working with a ChatBot if needed) what the specific (outcome and predictor) variables are for the scenario below; whether or not any meaningful interactions might need to be taken into account when predicting the outcome; and provide the linear forms with and without the potential interactions that might need to be considered<br>

> Imagine a company that sells sports equipment. The company runs advertising campaigns on TV and online platforms. The effectiveness of the TV ad might depend on the amount spent on online advertising and vice versa, leading to an interaction effect between the two advertising mediums.    

1. Explain how to use these two formulas to make **predictions** of the **outcome**, and give a high level explaination in general terms of the difference between **predictions** from the models with and without the **interaction** 

2. Explain how to update and use the implied two formulas to make predictions of the outcome if, rather than considering two continuous predictor variables, we instead suppose the advertisement budgets are simply categorized as either "high" or "low" (binary variables)    
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _While working on this question, it's important to clearly understand the (**outcome** and **predictor**) **variables** under consideration, and they way they are being considered. Similarly to the previous (first) question of this homework assignment, this question requires the **equations** of the indicated **linear forms** and an explanation of their **interpretation** and use. What is different here is that the **interactions** being considered are between two **continuous variables** or two **binary variables** (for a total of four **equations** under consideration where two include the **interactions** and two do not)._
>
> _The way an **interaction** actually appears and works in the context of the linear form of a **multiple linear regression** model may not necessarily be immediately intuitive, as it is actually in fact somewhat subtle and tricky. Of course, an **interaction** is when the relationship of one **predictor variable** with the **outcome variable** depends on the value of another different **predictor variable**, so the impact of one **predictor variable** changes based on the presence or magnitude of another **predictor variable**. But are you sure you know what this means in the context of the **linear form** of a **multiple linear regression** model?_
>
> - _Imagine you're selling smoothies, the taste of your smoothie depends on the ingredients included in the smoothie, and there are two ingredients (bananas and strawberries) in the smoothie._
>     - _Adding more bananas into the smoothie will of course increase the "banana" flavor of the smoothie, and vice-versa for strawberries..._
>     - _But is this "banana" influence on the flavor always constant, or can it change depending on the absolute amount of strawberries in the smoothie?_ 
>     - _If the "banana" flavor influence is constant and does not depend on the  absolute amount of strawberries in the smoothie, then there is no **interaction** and the **linear form** of the model is $\beta_b b_i + \beta_s s_i$ and the model is said to be only **additive**._
>     - _But if the "banana" flavor influence does depend on the absolute amount of strawberries in the smoothie, then there IS an **interaction** and the **linear form** of the model is $\beta_b b_i + \beta_s s_i + \beta_{bs} (b_i\times s_i)$ and the model is said to be **synergistic**._
>         
> _These **linear forms** show that either bananas and strawberries do not have any **synergistic interaction** and contribute to the flavor independently; or, they do have a **synergistic interaction** and there is an interesting interplay between bananas and strawberries in the way they influence the taste of the smoothie._ 
> 
> - _So, if there is no **interaction**, then the effect of adding more bananas on the taste of the smoothie will always be the same, no matter how many strawberries you put in. So the effect of bananas on the smoothie is the same whether you add a lot of strawberries or just a few: $\beta_b b_i + \beta_s s_i$_
> - _Or, on the other hand, if there is an **interaction**, then the effect of adding bananas (on the smootie flavor) will be different depending on how many strawberries there currently are in the smoothie: $\beta_b b_i + \beta_s s_i + \beta_{bs} (b_i\times s_i)$_
> 
> _In this case, the right answer is probably that the **linear form** with the **interaction** is correct. This is because the flavor probably depends on the relative amount of bananas and strawberries in the smoothie; so, the effect of adding a fixed amount of bananas to the smoothie probalby depends on the absolute amount of strawberries that are in the smoothie._
> 
> _Again, because understanding **interactions** in the context of **linear forms** is somewhat subtle and tricky and indeed not necessarily obviously intuitive, let's think about this a bit more. And we can simplify the concept a little bit by considering how this **interaction** would actually technically work in a **linear form** if we just had **binary indicator variables**._
>         
> - _To consider the smootie example in terms of binary variables, suppose that if both fruits are added to the smootie, they will be added in the same amount. So the smoothie will be made with either just bananas, just strawberries, or both (or neither and you won't make a smoothie)._ 
>     - _The question regarding an **interaction** then is, is the influence of the ingredients on the taste of the smoothie **additive** or **synergistic**? That is, does the way bananas affects the flavor of the smoothie change depending on the inclusion or exclusion of strawberries in the smoothie?_
>     - _**Additive** $\beta_b 1_{[b_i=1]}(b_i) + \beta_s 1_{[s_i=1]}(s_i)$ means there are three different flavors but they are explained by just two **parameters**: banana $\beta_b$, strawberry $\beta_s$, and banana-strawberry $\beta_b+\beta_s$_
>     - _**Synergistic** $\beta_b 1_{[b_i=1]}(b_i) + \beta_s 1_{[s_i=1]}(s_i) + \beta_{bs}(1_{[b_i=1]}(b_i) \times 1_{[s_i=1]}(s_i))$ means there are of course again three different flavors, but this time they are explained by three **parameters**: banana $\beta_b$, strawberry $\beta_s$, and banana-strawberry $\beta_b+\beta_s + \beta_{bs}$, which indicates that the flavor is "more than just sum of its parts", meaning there is a **synergistic interaction** and there is an interesting interplay between bananas and strawberries in the way they influence the taste of the smoothie_
>     
> _As the **additive** and **synergistic** versions of the **linear form** of the two **binary indicator variables** context shows, we don't need an interaction to make different predictions for different combinations of things. Instead, what these show is that the prediction will either be **additive** and "just the sum of it's parts" or **synergistic** (**interactive**) and "more than just sum of its parts"._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
      
</details>


### 3. Use *smf* to fit *multiple linear regression* models to the course project dataset from the canadian social connection survey<br>

> **EDIT: No, you probably actually care about CATEGORICAL or BINARY outcomes rather than CONTINUOUS outcomes... so you'll probably not actually want to do _multiple linear regression_ and instead do _logistic regression_ or _multi-class classification_. Okay, I'll INSTEAD guide you through doing _logistic regression_.**

1. ~~for an **additive** specification for the **linear form** based on any combination of a couple **continuous**, **binary**, and/or **categorical variables** and a **CONTINUOUS OUTCOME varaible**~~ 
    1. This would have been easy to do following the instructions [here](https://www.statsmodels.org/dev/example_formulas.html). A good alternative analagous presentation for logistic regression I just found seems to be this one from a guy named [Andrew](https://www.andrewvillazon.com/logistic-regression-python-statsmodels/). He walks you through the `logit` alternative to `OLS` given [here](https://www.statsmodels.org/dev/api.html#discrete-and-count-models).
    2. Logistic is for a **binary outcome** so go see this [piazza post](https://piazza.com/class/m0584bs9t4thi/post/346_f1) describing how you can turn any **non-binary categorical variable** into a **binary variable**. 
    3. Then instead do this problem like this: **catogorical outcome** turned into a **binary outcome** for **logistic regression** and then use any **additive** combination of a couple of **continuous**, **binary**, and/or **categorical variables** as **predictor variables**. 


```python
# Here's an example of how you can do this
import pandas as pd
import statsmodels.formula.api as smf

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
pokeaman = pd.read_csv(url).fillna('None')

pokeaman['str8fyre'] = (pokeaman['Type 1']=='Fire').astype(int)
linear_model_specification_formula = \
'str8fyre ~ Attack*Legendary + Defense*I(Q("Type 2")=="None") + C(Generation)'
log_reg_fit = smf.logit(linear_model_specification_formula, data=pokeaman).fit()
log_reg_fit.summary()
```


2. ~~for a **synertistic interaction** specification for the **linear form** based on any combination of a couple **continuous**, **binary**, and/or **categorical variables**~~
    1. But go ahead and AGAIN do this for **logistic regression** like above.
    2. Things are going to be A LOT simpler if you restrict yourself to **continuous** and/or **binary predictor variables**.  But of course you could *use the same trick again* to treat any **categorical variable** as just a **binary variable** (in the manner of [that piazza post](https://piazza.com/class/m0584bs9t4thi/post/346_f1).
    

3. and **interpretively explain** your **linear forms** and how to use them to make **predictions**
    1. Look, intereting **logistic regression** *IS NOT* as simple as interpreting **multivariate linear regression**. This is because it requires you to understand so-called **log odds** and that's a bit tricky. 
    2. So, INSTEAD, **just intepret you logistic regression models** *AS IF* they were **multivariate linear regression model predictions**, okay?


4. and interpret the statistical evidence associated with the **predictor variables** for each of your model specifications 
    1. **Yeah, you're going to be able to do this based on the `.fit().summary()` table _just like with multiple linear regression_**... now you might be starting to see how AWESOME all of this stuff we're doing is going to be able to get...


5. and finally use `plotly` to visualize the data with corresponding "best fit lines" for a model with **continuous** plus **binary indicator** specification under both (a) **additive** and (b) **synergistic** specifications of the **linear form** (on separate figures), commenting on the apparent necessity (or lack thereof) of the **interaction** term for the data in question
    1. Aw, shit, you DEF not going to be able to do this if you're doing **logistic regression** because of that **log odds** thing I mentioned... hmm...
    2. OKAY! Just *pretend* it's **multivariate linear regression** (even if you're doing **logistic regression**) and *pretend* your **fitted coefficients** belong to a **continuous** and a **binary predictor variable**; then, draw the lines as requested, and simulate **random noise** for the values of your **predictor data** and plot your lines along with that data.
    

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _This [link](https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html) offers guidance on using `statsmodels.formula.api` (`smf`) to build statistical models in Python using formulas._
>
> _The "best fit lines" summarize the relationship between the **outcome** and **predictor variables** observed in the data as well as the **linear form** of the **multiple linear regression** allows. The statistical evidence for the these estimated realtionship characterizations of course depends on an evaluation of the **hypothesis testing** for the **coefficients** of the model. **Model building** is the process of exploring the evidence for observed relationships captured through the modeling of the data in order to arrive at reliable (**generalizable**) claims based on the data, and perhaps make predictions about the future based on these created beliefs and understandings (whose value of course depends on how trustworthy these created beliefs and understandings are)._
>
> _When we do not find sufficient sufficient evidence for supposed relationships that we'd like to leverage for understanding or prediction, attempting to move forward on the basis of such "findings" is certainly a dangerous errand..._
    
</details>


### 4. Explain the apparent contradiction between the factual statements regarding the fit below that "the model only explains 17.6% of the variability in the data" while at the same time "many of the *coefficients* are larger than 10 while having *strong* or *very strong evidence against* the *null hypothesis* of 'no effect'"<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _How do we simultaneously interpret **hypothesis testing** results regarding **coefficient estimates** based on **p-values** and **R-squared** "the proportion of variation in (outcome) $y$ explained by the model ($\hat y_i$)"? How can both be meaningfully understood at the same time? Do they address different aspects of a model?_
>    
> _As introduced in the previous homework, **R-squared** is_
>
> _$$R^2 = 1 - \frac{\sum_{i=1}^n(y_i-\hat y)^2}{\sum_{i=1}^n(y_i-\bar y)^2}$$_
>    
> _which describes the **explanatory power** of a model; whereas, **p-values** allow us to characterize **evidence against** a **null hypothesis**, and **coefficients** in a **multiple linear regression** context allow us to interpret the relationship between the **outcome** and a **predictor variable** "with all other **predictor variables** 'held constant'". Are these concepts thus contradictory or conflictual in some manner?_

|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
    
> _In `formula='HP ~ Q("Sp. Def") * C(Generation)'` the `Q` stands for "quote" and is needed to access column names when they have a "space" in their name, while the `C` indicates a **categorical** use of what is actually an **integer** valued column. Despite technically being **continuous** numbers, **integer** often simply indicate categories which should not necessarily be treated as an incremental **continuous predictor variable**. Remember, a model such as $\beta_0 + \beta_1 x$ means for each unit increase in $x$ the outcome increases "on average" by $\beta_1$; so, if $x$ takes on the values `1` through `6` as the `Generation` **predictor variable** here does, then this means the average value for "Generation 1" must be $\beta_0 + \beta_1$ while for "Generation 2" it must be $\beta_0 + 2\times \beta_1$ (and so on up to "Generation 6" which must be $\beta_0 + 6\times \beta_1$). This might be a very strange restriction to place on something that is really actually a **categorical predictor variable**. You can see in the given model fit below how this six-level **categorical predictor variable** is actually appropriately treated in the specification of the **linear form** using "Generation 1" for the "baseline" and **binary indicators** to model the "contrast" ("offsets") for the other "Generations"; and, how these are in turn used in the context of the **interaction** considered by the model specification._ 
    
</details>


```python
import pandas as pd

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# fail https://github.com/KeithGalli/pandas/blob/master/pokemon_data.csv
pokeaman = pd.read_csv(url) 
pokeaman
```


```python
import statsmodels.formula.api as smf

model1_spec = smf.ols(formula='HP ~ Q("Sp. Def") + C(Generation)', data=pokeaman)
model2_spec = smf.ols(formula='HP ~ Q("Sp. Def") + C(Generation) + Q("Sp. Def"):C(Generation)', data=pokeaman)
model2_spec = smf.ols(formula='HP ~ Q("Sp. Def") * C(Generation)', data=pokeaman)

model2_fit = model2_spec.fit()
model2_fit.summary()
```

<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Pre-lecture VS Post-lecture HW
    
Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: in this case the "Post-lecture" HW just builds on the "Post-lecture" HW, introducing and extending the considerations available in the **multiple linear regression context**. That said, as "question 3" above hopefully suggests and reminds you, the **course project** is well upon us, and prioritizing work on that (even over the homework) may very well be indicated at this point...

*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details>    


## "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]


### 5. Discuss the following (five cells of) code and results with a ChatBot and based on the understanding you arrive at in this conversation explain what the following (five cells of) are illustrating<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _Recall from the previous week's homework that the **R-squared** "the proportion of variation in (outcome) $y$ explained by the model ($\hat y_i$)" quantity (re-introduced in the previous problem) can be calculated as `np.corrcoef(y,fitted_model.fittedvalues)[0,1]**2` (as well as several other ways in the **simple linear regression** context). The **squared correlation** between the outcome $y$ and it's **fitted values** $\hat y$ is the most generally useful formulation of **R-squared** since this can be use in the **multiple linear regression** context._
> 
> _This question therefore thus addresses the question of model **generalizability** on the basis of "in sample" and "out of sample" **model performance** (measured by **R-squared**)._
> 
> - _The **squared correlation** between the **outcomes** $y$ and their **fitted values** $\hat y$ is an "in sample" **model performance** metric since the $\hat y$ "predictions" for the $y$ **outcomes** are based on using those already **observed outcomes** to fit the model to generate the $\hat y$._  
> 
> - _If we instead calculate **squared correlation** between **outcomes** $y$ that were not used to fit the model and their corresponding $\hat y$ **predictions** (which are indeed now actually **predictions** as opposed to **fitted values**), then we are now  calculating an "out of sample" **model performance** metric._
> 
> _When an "out of sample" metric performs more poorly than a comparitive "in sample" metric, then the **predictions** of the **fitted model** are not **generalizing** to data being the dataset the model is fit on. In this case we say the model is **overfit** (to the data its fit was based on). The purpose of using different **training** and **testing** datasets is to consider "in sample" versus "out of sample" **model performance** in order to try to confirm that the model is not **overfit** and that the **predictions** do indeed seem to **generalizable** beyond the dataset used for **model fitting**._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_
    
</details>


```python
import numpy as np
from sklearn.model_selection import train_test_split

fifty_fifty_split_size = int(pokeaman.shape[0]*0.5)

# Replace "NaN" (in the "Type 2" column with "None")
pokeaman.fillna('None', inplace=True)

np.random.seed(130)
pokeaman_train,pokeaman_test = \
  train_test_split(pokeaman, train_size=fifty_fifty_split_size)
pokeaman_train

```


```python
model_spec3 = smf.ols(formula='HP ~ Attack + Defense', 
                      data=pokeaman_train)
model3_fit = model_spec3.fit()
model3_fit.summary()
```


```python
yhat_model3 = model3_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model3_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model3)[0,1]**2)
```


```python
model4_linear_form = 'HP ~ Attack * Defense * Speed * Legendary'
model4_linear_form += ' * Q("Sp. Def") * Q("Sp. Atk")'
# DO NOT try adding '* C(Generation) * C(Q("Type 1")) * C(Q("Type 2"))'
# That's 6*18*19 = 6*18*19 possible interaction combinations...
# ...a huge number that will blow up your computer

model4_spec = smf.ols(formula=model4_linear_form, data=pokeaman_train)
model4_fit = model4_spec.fit()
model4_fit.summary()
```


```python
yhat_model4 = model4_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model4_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model4)[0,1]**2)
```

### 6. Work with a ChatBot to understand how the *model4_linear_form* (*linear form* specification of  *model4*) creates new *predictor variables* as the columns of the so-called "design matrix" *model4_spec.exog* (*model4_spec.exog.shape*) used to predict the *outcome variable*  *model4_spec.endog* and why the so-called *multicollinearity* in this "design matrix" (observed in *np.corrcoef(model4_spec.exog)*) contribues to the lack of "out of sample" *generalization* of *predictions* from *model4_fit*; then, explain this consisely in your own works<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _The **overfitting** observed in the previous problem is a question of **model complexity** relative to the amount of information or evidence available in a given dataset (or we could just say "the amount of data in a dataset"). The **model fit** for  `model4` resulted in an **overfit model** because the specification of its **linear form** was overly complex (relative to the the amount of available data). Indeed, `model4` is rediculously complex as can be seen from `model4_fit.summary()`. This in turn allowed the **model fit** to "detect" idiosyncratic associations spuriously present specifically in the **training** dataset but which did not **generalize** to the **testing** dataset. If a model is too **complex** then it will find and model "patterns" in a **training** dataset which are actually just accidental "noise" from the random sampling nature of the dataset. The simpler (or more parsimoneous) `model3` on the other hand was able to identify **predictive associations** in the **training** dataset which did **generalize** to the **testing** dataset. This is because `model3` only sought to understand simpler **predictive association** for which there was enough data in the **training** dataset to reliably detect and obviously identify. And these patterns were indeed sufficiently "real" in the sense that they were present and **generalized** into the **testing** dataset as well as the **training** dataset. So they could be "found" in the **training** and then used in (**generalized** to) the **testing** dataset._
> 
> _This question therefore addresses the topic of the **evidence** a given dataset provides for the **predictive associations** detected by a **fitted model**. It should be increasingly clear at this point that evidence for a model can be addressed using **coefficent hypothesis testing** in the context of **multiple linear regression**, but that examinations of "in sample" versus "out of sample" **model performance** metrics are what in fact are directly designed to address this question of **generalizability**. That said, this question introduces another consideration of **multicollinearity** as something that affects the **generalizability** of **model fits** in the **multiple linear regression** context. A good question that a ChatBot could help you understand is (a) "why is **generalizability** more uncertain if two **predictor variables** are highly **correlated**?" and (b) "why is **generalizability** more uncertain if multiple **predictor variables** are highly **multicollinear**?"_
>
> _The four code cells below are not necessary for answering this question; however, they introduce two very practical helpful tools for the **multiple linear regression** context that are immediately relevant for this question. The first is the so-called **condition number** (of a "design matrix") which provides a very simple diagnostic which can serve as a measure the degree of **multicollinearity** that is present in a **model fit**. If this number is "very large" then there is a large degree of **multicollinearity** and suggests room for doubt regarding the **generalizability** of the **fitted model**. The second tool(s) are the `center` and `scale` functions. It is best practice to "center and scale" **continuous predictor variables** (but not **indicator variables**) in the **multiple linear regression** context as is done below. While "centering and scaling" does make interpreting the predictions on the original scale of the data slighly more complicated, it also must be done in order to get a "true" evaluation of the degree of **multicollinearity** present in a **model fit** using the **condition number** of the model ("design matrix"). The examples below show that the **condition number** reported by a **fitted model** are "artificially inflacted" if "centering and scaling" is not used. Specically, they show that the **condition number** of `model3_fit` is really `1.66` (as opposed to the "very large" `343` which is reported without "centering and scaling"); whereas, the **condition number** for `model4_fit` is "very (VERY) large" irrespective of "centering and scaling", showing that the overwheling presense of **multicollinearity** in `model4_fit` is in fact a very real thing.  Indeed, we have already seen that `model4_fit` is grossly **overfit** and does not remotely **generalize** beyond its **training** dataset. Without knowing this, however, the comically large **condition number** for `model4_fit` (after "centering and scaling") makes it abundantly clear that we should have great doubts about the likely **generalizability** of `model4_fit` (even without examining specific aspects of **multicollinearity** directly or examining "in sample" versus "out of sample" **model performance** comparisions)._
>
> - _The "specific aspects of **multicollinearity**" reference above refer to understanding and attributing the detrmimental affects of specific **predictor variables** towards **multicollinearity**. This can be done using so-called **variance inflation factors**, but this is beyond the scope of STA130. We should be aware that the presence of excessive **multicollinearity** as indicated by a large **condition number** for a ("centered and scaled") **fitted model** raises grave concerns regarding the potential **generalizability** of the model._
>
> _The `np.corrcoef(model4_spec.exog)` examination of the **correlations** of a "design matrix" considered in ths problems prompt is analogous to the examination of the **correlations** present in a dataset that might considered when initially examining the **predictor variables** of a dataset, such as `pokeaman.iloc[:,4:12].corr()`. Indeed, such an examination is often the first step in examining the potential presence of **multicollinearity** among the **predictor variables** of a dataset. However, these are consideration of **pairwise correlation**, whereas **multicollinearity** generalizes this notion to the full collection of **predictor variables** together. A **condition number** for a "centered and scale" version of a **fit model** can therefore be viewed as serving the analogous purposes of a multivariate generalization of **pairwise correlation**._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
    
</details>



```python
# "Cond. No." WAS 343.0 WITHOUT to centering and scaling
model3_fit.summary() 
```


```python
from patsy import center, scale

model3_linear_form_center_scale = \
  'HP ~ scale(center(Attack)) + scale(center(Defense))' 
model_spec3_center_scale = smf.ols(formula=model3_linear_form_center_scale,
                                   data=pokeaman_train)
model3_center_scale_fit = model_spec3_center_scale.fit()
model3_center_scale_fit.summary()
# "Cond. No." is NOW 1.66 due to centering and scaling
```


```python
model4_linear_form_CS = 'HP ~ scale(center(Attack)) * scale(center(Defense))'
model4_linear_form_CS += ' * scale(center(Speed)) * Legendary' 
model4_linear_form_CS += ' * scale(center(Q("Sp. Def"))) * scale(center(Q("Sp. Atk")))'
# Legendary is an indicator, so we don't center and scale that

model4_CS_spec = smf.ols(formula=model4_linear_form_CS, data=pokeaman_train)
model4_CS_fit = model4_CS_spec.fit()
model4_CS_fit.summary().tables[-1]  # Cond. No. is 2,250,000,000,000,000

# The condition number is still bad even after centering and scaling
```


```python
# Just as the condition number was very bad to start with
model4_fit.summary().tables[-1]  # Cond. No. is 12,000,000,000,000,000

```

### 7. Discuss with a ChatBot the rationale and principles by which *model5_linear_form* is  extended and developed from *model3_fit* and *model4_fit*; *model6_linear_form* is  extended and developed from *model5_linear_form*; and *model7_linear_form* is  extended and developed from *model6_linear_form*; then, explain this breifly and consisely in your own words<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _We again include the **condition number** for the "centered and scaled" version of `model7_fit` to show that **multicollinearity** does not appear to be a major concern for this model (and the same would be true regarding `model6_fit` if the analogous "centered and scaled" version of the model was considered). While it is true that the **condition number** of `15.4` observed for `model7_fit` is perhaps "large", this would not be considered "vary large"._
>
> - _Regarding **condition numbers**, a ChatBot gave me cutoffs of `<30` not a big problem, up to `<300` maybe an issue, up to `<1000` definitely **multicollinearity**, and beyond that is pretty much likely to be "serious" problems with **multicollinearity**. Personally, cutoffs around `10`, `100`, and `1000` seem about right to me._
>
> _This question addresses the **model building** exercise using both an **evidence** based approach using **coefficient hypothesis testing** as well as examinations of **generalizability** using comparisions of "in sample" versus "out of sample" **model performance** metrics. Through these tools, different models were considered, extended, and developed, finally arriving at `model7_fit`. When we feel we can improve the **model performance** in a **generalizable** manner, then all relatively underperforming models are said to be **underfit**, meaning that they do not leverage all the **predictive associations** available to improve **predictions**._
> 
> _While the previous "Question 6" above introduced and explored the impact of **multicollinearity** in the **multiple linear regression** context_ 
>     
> - _(whereby "the effects" of multiple **predictor variables** are "tangled up" and therefore do not allow the model to reliably determine contribution attributions between the **predictor variables**, which potentially leads to poor **estimation** of their "effects" in the model, which in turn is the problematic state of affairs which leads to a lack of **generalizability** in such high **multicollinearity** settings)_
> 
> _there is still the (actually even more important) consideration of the actual **evidence** of **predictive associations**. The question is whether or not there is sufficient **evidence** in the data backing up the **estimated** fit of the **linear form** specification. Quantifying the **evidence** for a **estimated** model is a separate question from the problem of **multicollinearity**, the assessment of which is actually the primary purpose of **multiple linear regression** methodology._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot); but, if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
    
</details>    


```python
# Here's something a little more reasonable...
model5_linear_form = 'HP ~ Attack + Defense + Speed + Legendary'
model5_linear_form += ' + Q("Sp. Def") + Q("Sp. Atk")'
model5_linear_form += ' + C(Generation) + C(Q("Type 1")) + C(Q("Type 2"))'

model5_spec = smf.ols(formula=model5_linear_form, data=pokeaman_train)
model5_fit = model5_spec.fit()
model5_fit.summary()
```


```python
yhat_model5 = model5_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model5_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model5)[0,1]**2)
```


```python
# Here's something a little more reasonable...
model6_linear_form = 'HP ~ Attack + Speed + Q("Sp. Def") + Q("Sp. Atk")'
# And here we'll add the significant indicators from the previous model
# https://chatgpt.com/share/81ab88df-4f07-49f9-a44a-de0cfd89c67c
model6_linear_form += ' + I(Q("Type 1")=="Normal")'
model6_linear_form += ' + I(Q("Type 1")=="Water")'
model6_linear_form += ' + I(Generation==2)'
model6_linear_form += ' + I(Generation==5)'

model6_spec = smf.ols(formula=model6_linear_form, data=pokeaman_train)
model6_fit = model6_spec.fit()
model6_fit.summary()
```


```python
yhat_model6 = model6_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model6_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2)
```


```python
# And here's a slight change that seems to perhaps improve prediction...
model7_linear_form = 'HP ~ Attack * Speed * Q("Sp. Def") * Q("Sp. Atk")'
model7_linear_form += ' + I(Q("Type 1")=="Normal")'
model7_linear_form += ' + I(Q("Type 1")=="Water")'
model7_linear_form += ' + I(Generation==2)'
model7_linear_form += ' + I(Generation==5)'

model7_spec = smf.ols(formula=model7_linear_form, data=pokeaman_train)
model7_fit = model7_spec.fit()
model7_fit.summary()
```


```python
yhat_model7 = model7_fit.predict(pokeaman_test)
y = pokeaman_test.HP
print("'In sample' R-squared:    ", model7_fit.rsquared)
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2)
```


```python
# And here's a slight change that seems to perhas improve prediction...
model7_linear_form_CS = 'HP ~ scale(center(Attack)) * scale(center(Speed))'
model7_linear_form_CS += ' * scale(center(Q("Sp. Def"))) * scale(center(Q("Sp. Atk")))'
# We DO NOT center and scale indicator variables
model7_linear_form_CS += ' + I(Q("Type 1")=="Normal")'
model7_linear_form_CS += ' + I(Q("Type 1")=="Water")'
model7_linear_form_CS += ' + I(Generation==2)'
model7_linear_form_CS += ' + I(Generation==5)'

model7_CS_spec = smf.ols(formula=model7_linear_form_CS, data=pokeaman_train)
model7_CS_fit = model7_CS_spec.fit()
model7_CS_fit.summary().tables[-1] 
# "Cond. No." is NOW 15.4 due to centering and scaling
```


```python
# "Cond. No." WAS 2,340,000,000 WITHOUT to centering and scaling
model7_fit.summary().tables[-1]
```

### 8. Work with a ChatBot to write a *for* loop to create, collect, and visualize many different paired "in sample" and "out of sample" *model performance* metric actualizations (by not using *np.random.seed(130)* within each loop iteration); and explain in your own words the meaning of your results and purpose of this demonstration<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _The following code could likely be slightly edited and repurposed to match the 50-50 **train-test split** analysis and data above (in the `train_test_split` method of "Question 5").  Considering the behavior of the `model3_fit` **linear form** specification would be the suggested way to start, but it might also be interesting and/or helpful to consider the different available **linear form** specifications in the manner of this problem..._
>    
> ```python
> import plotly.express as px  # etc.
>
> songs_training_data,songs_testing_data = train_test_split(songs, train_size=31)
> linear_form = 'danceability ~ energy * loudness + energy * mode'
>    
> reps = 100
> in_sample_Rsquared = np.array([0.0]*reps)
> out_of_sample_Rsquared = np.array([0.0]*reps)
> for i in range(reps):
>     songs_training_data,songs_testing_data = \
>       train_test_split(songs, train_size=31)
>     final_model_fit = smf.ols(formula=linear_form, 
>                               data=songs_training_data).fit()
>     in_sample_Rsquared[i] = final_model_fit.rsquared
>     out_of_sample_Rsquared[i] = \
>       np.corrcoef(songs_testing_data.danceability, 
>                   final_model_fit.predict(songs_testing_data))[0,1]**2
>     
> df = pd.DataFrame({"In Sample Performance (Rsquared)": in_sample_Rsquared,
>                    "Out of Sample Performance (Rsquared)": out_of_sample_Rsquared})   >  
> fig = px.scatter(df, x="In Sample Performance (Rsquared)", 
>                      y="Out of Sample Performance (Rsquared)")
> fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="y=x", line_shape='linear'))  
> ```
>
> _When you first look at this question, you might be unsure about the specific issue that the code is addressing. Take a moment to think about why the code repeatedly randomly re-splits the data, fits the model, and compares the "in sample" versus "out of sample" **R-squared** values (over and over). Of course, if a **fit model** performs well on the **training** dataset but doesn't do as well on the **testing** dataset then we might be observing the affects of **overfitting**. But why might it sometimes be the opposite situation (which we actually encountered right away for `model3_fit` when the **train-test split** was based on  `np.random.seed(130)` and resulted in a better "out of sample" **R-squared** of about `0.21` vereses the 'in-sample" **R-squared** of about `0.15`)? If you're thinking that this should therefore vice-versa intuitively mean **underfitting**, actually that's not right because **underfitting** is when the **generalizability** of a different model **linear form** specification that provides improved **model performance** is **validated**. What were seeing here, the variable, is something else..._
>        
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_ 
    
</details>
    

### 9. Work with a ChatBot to understand the meaning of the illustration below; and, explain this in your own words<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _While we had seemed to **validate** the **generalizability** of `model7_fit` in **model building** exercise of the previous "Question 7" above, as well as the improved **model performance** of `model7_fit` comapred to `model6_fit`, the `model7_fit` model was always nonetheless more complex than `model6_fit` model (as seen by comparing their `.summary()` methods). This complexity, despite the minimal concerns regarding **multicollinearity**, should always have suggested some room for caution. This is because, as previously discussed in "Question 6" above, a complex **linear form** specification can allow a "**model fit** to 'detect' idiosyncratic associations spuriously present specifically in the **training** dataset but which did not **generalize** to the **testing** dataset." Indeed, a close look at the **p-values** in `model7_fit.summary()` will show that the **evidence** (in the data) for many of the **estimated coefficients** of `model7_fit` is in fact not very strong. In comparision, the **evidence** (in the data) for many of the **estimated coefficients** of `model6_fit.summary()` is consistently stronger._
>
> _As discussed towards the end of the commentary in the previous "Question 7" above, the primary purpose of **multiple linear regression** methodology is to allow us to assess the **evidence** (in the data) for a given **linear form** specification based on **coefficient hypothesis testing**. In this regard, then, `model6_fit` might be preferred over `model7_fit` despite the better "out of sample" **model performance** of `model7_fit` over `model6_fit`. This may not be enough to convince everyone however, so an additional consideration that might be made here is that the more simpler (more parsimoneous) nature of `model6_fit` should be preferred over `model7_fit` from the perspective of **model interpretability**. Indeed, it is quite unclear how exactly one should think about and understand a four-way **interaction** variable such as `Attack:Speed:Q("Sp. Def"):Q("Sp. Atk")` in conjunction with the whole host of the additional lower order interations. From a **model interpretability** perspective, understanding the meaning of the complex specification of `model7_fit` is "challenging" and "complicated" to say the least._
>
> - _There are also often circumstances where **model interpretability** can be MORE IMPORTANT than raw **model performance** in "out of sample" **prediction**._
> - _This is ESPECIALLY true if **predictive model performance** is relatively comparable between models of two different complexity levels. In such cases, the benefits of better **model interpretability** might provide a clear argument for the simpler (more parsimoneous) model, not to mention the additional potential benefit of more consistent improved **generalizability** over the the more complex model this might offer._
>
> _This question drives home the point that a simpler (more parsimoneous) model always offers the potential benefit of more consistent **generalizability**, not to mention **interpretability**, over more complex models. We should *ONLY* use increasingly complex models that without questions outperfrm simler models. The code below illustrates this by further additionally raising the consideration that the random **train-test** approach used above is actually not the most natural one available for our dataset, which has different "Generations". In fact, if we were actually using this model to make **predictions**, we would increasingly acquire more data over time which we would use to make **precictions** about future data which we haven't yet seen, which is what the code demonstrates. And low and behold, this exposes **generalizability** concerns that we missed when we used the dataset in an idealized way and not actually how we would use such a dataset in practice in the real world (where data would arrive sequentially, and current data is used to predict future data). These **generalizability** concerns do affect both models, but the appear to be more problematic for `model7_fit` than `model6_fit`, which is certainly a result of the increased complexity of `model7_fit` which always opens up the possibility of model **overfitting**._

<details>    


```python
model7_gen1_predict_future = smf.ols(formula=model7_linear_form,
                                   data=pokeaman[pokeaman.Generation==1])
model7_gen1_predict_future_fit = model7_gen1_predict_future.fit()
print("'In sample' R-squared:    ", model7_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model7_gen1_predict_future_fit.rsquared, "(gen1_predict_future)")
y = pokeaman[pokeaman.Generation!=1].HP
yhat = model7_gen1_predict_future_fit.predict(pokeaman[pokeaman.Generation!=1])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1_predict_future)")
```


```python
model7_gen1to5_predict_future = smf.ols(formula=model7_linear_form,
                                   data=pokeaman[pokeaman.Generation!=6])
model7_gen1to5_predict_future_fit = model7_gen1to5_predict_future.fit()
print("'In sample' R-squared:    ", model7_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model7)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model7_gen1to5_predict_future_fit.rsquared, "(gen1to5_predict_future)")
y = pokeaman[pokeaman.Generation==6].HP
yhat = model7_gen1to5_predict_future_fit.predict(pokeaman[pokeaman.Generation==6])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1to5_predict_future)")
```


```python
model6_gen1_predict_future = smf.ols(formula=model6_linear_form,
                                   data=pokeaman[pokeaman.Generation==1])
model6_gen1_predict_future_fit = model6_gen1_predict_future.fit()
print("'In sample' R-squared:    ", model6_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model6_gen1_predict_future_fit.rsquared, "(gen1_predict_future)")
y = pokeaman[pokeaman.Generation!=1].HP
yhat = model6_gen1_predict_future_fit.predict(pokeaman[pokeaman.Generation!=1])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1_predict_future)")
```


```python
model6_gen1to5_predict_future = smf.ols(formula=model6_linear_form,
                                   data=pokeaman[pokeaman.Generation!=6])
model6_gen1to5_predict_future_fit = model6_gen1to5_predict_future.fit()
print("'In sample' R-squared:    ", model6_fit.rsquared, "(original)")
y = pokeaman_test.HP
print("'Out of sample' R-squared:", np.corrcoef(y,yhat_model6)[0,1]**2, "(original)")
print("'In sample' R-squared:    ", model6_gen1to5_predict_future_fit.rsquared, "(gen1to5_predict_future)")
y = pokeaman[pokeaman.Generation==6].HP
yhat = model6_gen1to5_predict_future_fit.predict(pokeaman[pokeaman.Generation==6])
print("'Out of sample' R-squared:", np.corrcoef(y,yhat)[0,1]**2, "(gen1to5_predict_future)")
```

## Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course

<br>
<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations

This week addresses **multiple linear regression**, perhaps best exemplified through the consideration on **interactions** and their impact on the **model interpretation**, **evidence** and **validity* of models using **coefficient hypothesis testing** and "in sample" versus "out of sample" **model performance** comparision. Exactly, as in **simple linear regression**, the correctness of **p-values** used to give **evidence** for **predictive associations** that are **estimated** from a dataset depends on the (at least approximate) "truth" of the assumptions of the **multiple linear regression**, which are the same as those of the **simple linear regression** with the exception that the specification **linear form** can now model a much richer set of relationships between **predictor** and **outcome variables** based on **predictive associations** observed and **evidenced** in the data. With all this in mind, and reflecting back on the **Ethical Professionalism Considerations** from the previous week concerning **simple linear regression**...

> - Which of the methods used for diagnostically assessing the assumptions of a **simple linear regression** specification could be used analogously generalized to the **multiple linear regression** context for the same purpose? 
> 
> - Examining the assumption of the **linear form** is more challenging in **multiple linear context**, but can be done using so-called **partial regression** (or **added variable**) **plot**. Is a ChatBot able to provide code to perform this diagnostic and instructions regarding its purpose, interpretation, and appropriate usage?
>     
> - Are there other diagnostic analyses that a ChatBot might suggest for you to help you evaluate the appropriateness of the assumptions of **fitted multiple linear regression model** you are considering using for **interpretation** or **prediction**? And if so, s the ChatBot able to provide code to perform these additional diagnostic and instructions regarding their purpose, interpretation, and appropriate usages?
>     
> - What do you think your ethical and professional responsibilites are when it comes to using and leveraging **multiple linear regression** methodology (and associated assumptions therein) in your work? To illustrate and demonstrate your thoughts on these considerations, can you give any specific examples of decisions that might be made during your process of executing a **multiple linear regression** that could have ethical and professional implications, risks, or consequences? What do you think are the simplest steps can you take to ensure that the conclusions of your work are both valid and reliable? What steps do you think are the most challenging from a practical perspective? 

</details>

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>

**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
    
> ### NEW DEVELOPMENT<br>New Abilities Achieved and New Levels Unlocked!!!    
> 
> "Question 3" as described below only addresses **multiple linear regression**... but you'll also perhaps notice that "Question 3" as described above extends this to address addresses **logistic regression**.
    
### Current Course Project Capability Level 
    
This homework's "Question 3" introduced the idea of performing some **multiple linear regression** analyses on dataset from the Canadian Social Connection Survey. While other questions of this homework focussed on other datasets, the general analyses and principles they introduce are no doubt informative and applicable to this the dataset for our course project. Ideally, this should put you in a position to quite proficiently perform **multiple linear regression** analyses for the course project if you so desire and find appropriate for the objectives of you course project submission. Thus, the following (and more) should be possible at this stage... 

1. Select multiple **predictors predictor** from the Canadian Social Connection Survey data and examine how they jointly influence an outcome variable, paying special attention to the inclusion and interpretation of **categorical** and **indicator variables** and **interactions** (in terms of "baseline" reference groups and "contrast" or "offsets").

2. Visualize different kinds of **predictive association** relationships, including **interactions** and relationship between **predictor** and the **outcome** variables that change across different levels of other **categorical** or **indicator predictor variables**, using tools like `plotly.express`.

3. Use **coefficient hypothesis testing** and "in sample" versus "out of sample" **model performance** evaluation to perform **model building** and examine **generalizability** of **fitted models**.
       
4. Assess the presence of **multicollinearity** by considering the **condition numbers** of **fitted models** (with "centering and scaling") and their subsequent potential implications on **generalizability** of **fitted models**; and, perhaps even examine **pairwise correlation** and/or **variance inflation factors** for each **predictor variable** if you're feeling extra ambitious and want to go well "above and beyond" (in which case you could also consider the relationship between **multicollinearity** and why one level of a **categorical** variable is always omitted).

5. Compare and contrast such analyses and their benefits with previous methodologies introduced and considered in the course.
    
6. Explore using model diagnostic to check assess the assumptions of your **multiple linear regression** analyses, and reflect on how failurse of these assumptions might impact the reliability of your findings and conlusions derived from your **fitted model**.

</details>

