## 00 Tools

This is the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki/). The other primary tools and resources of the course are as follows.

1. [UofT Jupyterhub](https://datatools.utoronto.ca) (classic notebook, jupyterhub, or [google colab](https://colab.research.google.com/) are all fine) `.ipynb` notebook files
2. [ChatGPT](https://chat.openai.com/) (or [Copilot](https://copilot.microsoft.com/)) "vanilla" ChatBots
3. [STA130 custom NBLM ChatBot](https://github.com/pointOfive/stat130chat130/wiki)
4. Course [GitHub](https://github.com/pointOfive/stat130chat130/blob/main/README.md)
5. Course [Quercus Homepage](https://q.utoronto.ca/courses/354091/)
6. Course [Piazza Discussion Board](https://piazza.com/utoronto.ca/fall2024/sta130)


## Week 01 Data Summarization

> Simple exploratory data analysis (EDA)  
> and AI ChatBots are Very Good (at some things)

**Tutorial/Homework: Topics**

1. importing libraries... like [_pandas_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#import)
2. loading data... with [_pd.read_csv()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#read_csv)
3. counting missing values... with [_df.isna().sum()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Missingness-I)
4. observations (rows) and variables (columns)... [_df.shape_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Variables-and-Observations) and [_df.columns_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Variables-and-Observations)
5. numeric versus non-numeric... [_df.describe()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Types-I) and [_df.value_counts()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Types-I)
6. removing missing data... with [_df.dropna()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Missingness-II) and [_del df['col']_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Missingness-II)
7. grouping and aggregation.... with [_df.groupby("col1")["col2"].describe()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Grouping-and-Aggregation)

**Tutorial/Homework: Lecture Extensions**

> Topic numbers below correspond to extensions of topic items above.

2\. [function/method arguments](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#functionmethod-arguments) (like `encoding`, `dropna`, `inplace`, and return vs side-effect)\
3\. [boolean values and coercion](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#Boolean-Values-and-Coercion)\
4\. _ i. [_.dtypes_ and _.astype()_](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#pandas-column-data-types)\
___ ii. [statistic calculation functions](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#some-statistics-calculations) 

**Lecture: New Topics**

1. [sorting and (0-based) indexing](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#sorting-and-iloc-indexing)
2. [subsetting via conditionals and boolean selection](https://github.com/pointOfive/stat130chat130/wiki/week-01-Data-Summarization#logical-conditionals-boolean-selectionsubsetting-and-loc-indexing-v2)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as how to handle missing values using more advanced techniques that don't just "ignore" or "remove" them (for example by filling or imputing the missing values and the assumptions required when doing so...)
4. ...further "data wrangling topics" such as "joining" and "merging"; "pivoting", "wide to long", and "tidy" data formats; etc.


## Week 02 Coding and Probability

> Chance is intuitive and use AI ChatBots<br> to make coding and understanding code easier

**Tutorial/Homework: Topic**

1. python object types... [_tuple_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#types), [_list_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#types), [_dict_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#types)
2. another key data type... [_np.array_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#nparary) (and `np.random.choice`)
3. for loops... [_for i in range(n):_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#for-loops)
    1. [_print()_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#for-loops)
    2. [_for x in some_list:_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-for-loops)
    3. [_for i,x in enumerate(some_list):_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-for-loops)
    4. ~`for key,val in dictionary.items()` and `dictionary.keys()` and `dictionary.values()`~
4. logical flow control... [_if_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#logical-flow-control), [_elif_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#logical-flow-control), [_else_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#logical-flow-control)
    1. ~[_try-except_ blocks](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#logical-flow-control)~

**Tutorial/Homework: Lecture Extensions**

1. more object types... [_type()_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-types) 
    1. [more indexing for "lists"](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-indexing)
    2. [more _np.array_ with _.dtype_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-nparray) 
    3. [more "list" behavior with _str_ and _.split()_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-list-behavior-with-str-and-split)
        1. ~text manipulation with `.apply(lambda x: ...)`, `.replace()`, and `re`~
    4. [operator overloading](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#operator-overloading)
2. [What are _pandas DataFrame objects_?](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#what-are-pddataframe-objects)
3. [_for word_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding/week-02-Coding#More-for-Loops) _in_ [_sentence.split():_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#more-list-behavior-with-str-and-split)

**Lecture: New Topics**

1. [_from scipy import stats_, _stats.multinomial_, and probability](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#scipystats) (and `np.random.choice`)
    1. [conditional probability Pr(A|B) and independence Pr(A|B)=Pr(A)](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding-and-Probability#conditional-probability-and-independence)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as modular code design (with `def` based functions or `classes`)
4. ...such as dictionary iteration (which has been removed from the above material)
5. ...such as text manipulation with `.apply(lambda x: ...)`, `.replace()`, `re` (which are introduced but are generally out of scope for STA130)


## Week 03 Data Visualization

> Populations and Sampling and more interesting EDA  
> by making figures with AI ChatBots 

**Tutorial/Homework: Topics**

1. [More Precise Data Types (As Opposed to Object Types)](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#continuous-discrete-nominal-and-ordinal-categorical-and-binary): continuous, discrete, nominal and ordinal categorical, and binary
2. [Bar Plots and Modes](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#Bar-plots-and-modes)
3. [Histograms](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#Histograms)
4. [Box Plots, Range, IQR, and Outliers](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#Box-plots-and-spread)
5. [Skew and Multimodality](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#skew-and-multimodality)
    1. [Mean versus Median](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#skew-and-multimodality)
    2. [Normality and Standard Deviations](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#normal-distributions)
    3. [Characteristics of a Normal Distribution](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#characteristics-of-a-Normal-Distribution)

**Tutorial/Homework: Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

> Topic numbers below correspond to extensions of topic items above.

2\. [Plotting: Plotly, Seaborn, Matplotlib, Pandas, and other visualization tools.](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#modern-plotting)\
___ i. [Legends, annotations, figure panels, etc.](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#legends-annotations-figure-panels-etc)\
3\. [Kernel Density Estimation using Violin Plots](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#smoothed-histograms)\
5\. [Log Transformations](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#log-transformations)

**Lecture: New Topics**

This section introduces new concepts that are not covered in the tutorial/homework topics.

1. [Populations](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#populations-and-distributions) _from scipy import stats_ 
	1. `stats.multinomial` and `np.random.choice()` 
	2. `stats.norm`, `stats.gamma`, and `stats.poisson`
2. [Samples](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#Sampling) versus populations (distributions)
3. [Statistical Inference](https://github.com/pointOfive/stat130chat130/wiki/week-03-Data-Visualization#Statistics-Estimate-Parameters)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above
	1. Expectation, moments, integration, heavy tailed distributions
	2. Kernel functions for kernel density estimation
3. bokeh, shiny, d3, etc...


## Week 04 Bootstrapping

> Confidence Intervals and Statistical Inference  
> (as opposed to just Estimation) using Sampling Distributions

**Tutorial/Homework: Topic**

1. [Simulation](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Simulation) (with `for` loops and `from scipy import stats`)
2. [Sampling Distribution of the Sample Mean](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#VariabilityUncertainty-of-the-Sample-Mean)
3. [Standard Deviation versus Standard Error](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Standard-Deviation-versus-Standard-Error)
4. [How n Drives Standard Error](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#how-n-drives-standard-error)

**Tutorial/Homework: Lecture Extensions**

1. [Independent Sampling](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Independent-Samples) functions like `df.sample([n=n/frac=1], replace=False)`
    1. [Are Sampling Distributions Skewed?](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Are-Sampling-Distributions-Skewed)
    2. [Bootstrapping](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#bootstrapping)
    3. [Not Bootstrapping](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#not-bootstrapping)

**Lecture: New Topics**

1. [Confidence Intervals](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Confidence-Intervals)
2. [Bootstrapped Confidence Intervals](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Bootstrapped-Confidence-Intervals)
3. ["Double" _for_ loops](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#double-for-loops)
    1. [Proving Bootstrapped Confidence Intervals using Simulation](https://github.com/pointOfive/stat130chat130/wiki/week-04-Bootstrapping#Proving-Bootstrapping)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as the **Central Limit Theorem (CLT)**, **Law of Large Numbers (LLN)**, and theoretical "x-bar plus/minus about 2 standard errors" confidence intervals (based on the so-called "pivot" form)
4. ... the alternative sampling function `np.random.choice(list_of_options, p, replace=True)` which will be introduced for different purposes later


## Week 05 Hypothesis Testing

> P-values And How To Use And Not Use Them

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

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. Simulation versus theoretical hypothesis testing frameworks, z-tests and t-tests, parametric versus nonparametric hypothesis testing frameworks, other tests such as Fisher Exam or Chi-squared or F-tests, etc...
4. Well, these above are indeed **out of scope** for the the STA130 **final exam** but it looks like they're going to be DEFINITELY *NOT out of scope* for the *course project**...


## Week 7ate9 Simple Linear Regression

> Normal Distributions gettin' jiggy wit it

**LEC 1 New Topics**

1. [Correlation Association (IS NOT Causation)](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#Correlation-Association-IS-NOT-Causation)
    1. [DO NOT USE Correlation to Measure ANYTHING EXCEPT "Straight Line" Linear Association](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#DO-NOT-USE-Correlation-to-Measure-ANYTHING-EXCEPT-Straight-Line-Linear-Association)
    2. [Correlation is just for Y = mx + b](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#Correlation-is-just-for-y--mx--b)
2. [Simple Linear Regression is Just a Normal Distribution](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#simple-linear-regression-is-just-a-normal-distribution)
    1. [Terminology: predictor, outcome, intercept and slope coefficients, and error terms](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#Terminology-predictor-outcome-intercept-and-slope-coefficients-and-error-terms)

**TUT/HW Topics**

1. [_import statsmodels.formula.api as smf_](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#statsmodels)
2. [_smf.ols_](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#smfols)
    1. ["R-style" formulas I](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#r-style-formulas-i)
    2. ["quoting" non-standard columns](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#quoting)
3. [_smf.ols("y~x", data=df).fit()_ and _.params_](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#fitting-models) $\hat \beta_k$ versus $\beta_k$
    1. [_.fittedvalues_](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#fittedvalues)
    2. [_.rsquared_ "variation proportion explained"](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#rsquared-variation-proportion-explained)
    3. [_.resid_ residuals and assumption diagnostics](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#resid-residuals-and-assumption-diagnostics)
4. [_smf.ols("y~x", data=df).fit().summary()_ and _.tables[1]_ for Testing "On Average" Linear Association](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#testing-on-average-linear-association)

**LEC 2 New Topics / Extensions**

1. [Two(2) unpaired samples group comparisons](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#two2-unpaired-samples-group-comparisons)
2. [Two(2) unpaired sample permutation tests](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#two2-unpaired-sample-permutation-tests)
3. [Two(2) unpaired sample bootstrapping](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#two2-unpaired-sample-bootstrapping)
4. [Indicator variables and contrasts linear regression](https://github.com/pointOfive/stat130chat130/wiki/week-7ate9-Simple-Linear-Regression#indicator-variables-and-contrasts-linear-regression)

**Out of scope:**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as all the stuff around multi/bivariate normal distribution and their covariance matrices, ellipses and their math and visual weirdness outside of a 1:1 aspect ratio, and eigenvectors and eigenvalues and major axis lines, etc...
4. ...such as the mathematical formulas correlation, but just noting that they sort of just look like formulas for variance...


## Weeks 10 Multiple Linear Regression

> ~~Normal Distributions~~ Now REGRESSION'S gettin' jiggy wit it

**Tutorial/Homework: Topics**

1. [Multiple Linear Regression](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#multiple-linear-regression)
    1. [Interactions](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#interactions)
    2. [Categoricals](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#categoricals)
2. [Model Fitting](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#model-fitting)
    1. [Evidence-based Model Building](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#evidence-based-model-building)
    2. [Performance-based Model Building](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#performance-based-model-building)
    3. [Complexity, Multicollinearity, and Generalizability](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#complexity-multicollinearity-and-generalizability)

**Tutorial/Homework/Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

1. [Logistic Regression](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#logistic-regression)
    1. [Categorical to Binary Cat2Bin Variables](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#categorical-to-binary-cat2bin-variables)
1. [And Beyond](https://github.com/pointOfive/stat130chat130/wiki/Weekz-10-Multiple-Linear-Regression#and-beyond)

**Lecture: New Topics**

1. I'm planning to just show you how I work on this kind of data with a pretty interesting example...

**Out of scope:**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...the deep mathematical details condition numbers, variance inflation factors, K-Folds Cross-Validation...
4. ...the actual deep details of log odds, link functions, generalized linear models...

Weekz 11 Classification Decision Trees

> Machine Learning

**Tutorial/Homework: Topics**

1. [Classification Decision Trees](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#classification-decision-trees)
    1. [Classification versus Regression](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#classification-versus-regression) 
2. [`scikit-learn` versus `statsmodels`](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#scikit-learn-versus-statsmodels)
    1. [Feature Importances](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#feature-importances)
3. [Confusion Matrices](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#confusion-matrices)
    1. [Metrics](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#metrics)
4. [In Sample versus Out of Sample](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#in-sample-versus-out-of-sample)

**Tutorial/Homework Extensions/New Topics for Lecture**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

1. [Model Fitting: Decision Trees Construction](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#model-fitting-decision-trees-construction) 
2. [Model Complexity and Machine Learning](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#model-complexity-and-machine-learning)
3. [Prediction](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#prediction)
    1. [ROC curves](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#roc-curves)
    2. [Partial Dependency Plots](https://github.com/pointOfive/stat130chat130/wiki/weekz-11-classification-decision-trees#partial-dependency-plots)


**Out of scope:**

1. Additional **classification metrics** and additional considerations around **confusion matrices** beyond those discussed above [previously, "Material covered in future weeks"]
2. Deeper details of **Decision Tree** and **Random Forest** construction (**model fitting**) processes [previously, "Anything not substantively addressed above"]
3. ...the actual deep details of log odds, link functions, generalized linear models, and now **multi-class classification** since we can instead just use `.predict()` and we now know about `predict_proba()`...
4. ...other **Machine Learning** models and the rest of `scikit-learn`, e.g., K-Folds Cross-Validation and **model complexity regularization tuning** with `sklearn.model_selection.GridSearchCV`, etc.
