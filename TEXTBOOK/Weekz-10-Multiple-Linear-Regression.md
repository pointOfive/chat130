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


