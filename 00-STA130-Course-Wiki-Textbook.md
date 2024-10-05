## 00 Tools

This is the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki/). The other primary tools and resources of the course are as follows.

1. [UofT Jupyterhub](https://datatools.utoronto.ca) (classic notebook, jupyterhub, or [google colab](https://colab.research.google.com/) are all fine) `.ipynb` notebook files
2. [ChatGPT](https://chat.openai.com/) (or [Copilot](https://copilot.microsoft.com/)) "vanilla" ChatBots
3. [STA130 custom NBLM ChatBot](https://github.com/pointOfive/stat130chat130/wiki)
4. Course [GitHub](https://github.com/pointOfive/stat130chat130/blob/main/README.md)
5. Course [Quercus Homepage](https://q.utoronto.ca/courses/354091/)
6. Course [Piazza Discussion Board](https://piazza.com/utoronto.ca/fall2024/sta130)


## Week 01 Data Summarization

**Tutorial/Homework: Topics**

1. importing libraries... like [_pandas_](week-01-Data-Summarization#import)
2. loading data... with [_pd.read_csv()_](week-01-Data-Summarization#read_csv)
3. counting missing values... with [_df.isna().sum()_](week-01-Data-Summarization#Missingness-I)
4. observations (rows) and variables (columns)... [_df.shape_](week-01-Data-Summarization#Variables-and-Observations) and [_df.columns_](week-01-Data-Summarization#Variables-and-Observations)
5. numeric versus non-numeric... [_df.describe()_](week-01-Data-Summarization#Types-I) and [_df.value_counts()_](week-01-Data-Summarization#Types-I)
6. removing missing data... with [_df.dropna()_](week-01-Data-Summarization#Missingness-II) and [_del df['col']_](week-01-Data-Summarization#Missingness-II)
7. grouping and aggregation.... with [_df.groupby("col1")["col2"].describe()_](week-01-Data-Summarization#Grouping-and-Aggregation)

**Tutorial/Homework: Lecture Extensions**

> Topic numbers below correspond to extensions of topic items above.

2\. [function/method arguments](week-01-Data-Summarization#functionmethod-arguments) (like `encoding`, `dropna`, `inplace`, and return vs side-effect)\
3\. [boolean values and coercion](week-01-Data-Summarization#Boolean-Values-and-Coercion)\
4\. _ i. [_.dtypes_ and _.astype()_](week-01-Data-Summarization#pandas-column-data-types)\
___ ii. [statistic calculation functions](week-01-Data-Summarization#some-statistics-calculations) 

**Lecture: New Topics**

1. [sorting and (0-based) indexing](week-01-Data-Summarization#sorting-and-iloc-indexing)
2. [subsetting via conditionals and boolean selection](week-01-Data-Summarization#logical-conditionals-boolean-selectionsubsetting-and-loc-indexing-v2)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as how to handle missing values using more advanced techniques that don't just "ignore" or "remove" them (for example by filling or imputing the missing values and the assumptions required when doing so...)
4. ...further "data wrangling topics" such as "joining" and "merging"; "pivoting", "wide to long", and "tidy" data formats; etc.


## Week 02 Coding and Probability

**Tutorial/Homework: Topic**

1. python object types... [_tuple_](week-02-Coding#Types), [_list_](week-02-Coding#types), [_dict_](week-02-Coding#Types)
2. another key data type... [_np.array_](week-02-Coding#np-array) (and `np.random.choice`)
3. for loops... [_for i in range(n):_](week-02-Coding#for-loops)
    1. [_print()_](week-02-Coding#for-loops)
    2. [_for x in some_list:_](week-02-Coding#More-for-Loops)
    3. [_for i,x in enumerate(some_list):_](week-02-Coding#More-for-Loops)
    4. ~`for key,val in dictionary.items()` and `dictionary.keys()` and `dictionary.values()`~
4. logical flow control... [_if_](week-02-Coding#Logical-Flow-Control), [_elif_](week-02-Coding#Logical-Flow-Control), [_else_](week-02-Coding#Logical-Flow-Control)
    1. ~[_try-except_ blocks](week-02-Coding#Logical-Flow-Control)~

**Tutorial/Homework: Lecture Extensions**

1. more object types... [_type()_](week-02-Coding#more-types) 
    1. [more indexing for "lists"](week-02-Coding#more-indexing)
    2. [more _np.array_ with _.dtype_](week-02-Coding#more-nparray) 
    3. [more "list" behavior with _str_ and _.split()_](week-02-Coding#more-list-behavior-with-str-and-split)
        1. ~text manipulation with `.apply(lambda x: ...)`, `.replace()`, and `re`~
    4. [operator overloading](week-02-Coding#operator-overloading)
2. [What are _pandas DataFrame objects_?](week-02-Coding#what-are-pandas-dataframe-objects)
3. [_for word_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding/week-02-Coding#More-for-Loops) _in_ [_sentence.split():_](week-02-Coding#more-list-behavior-with-str-and-split)

**Lecture: New Topics**

1. [_from scipy import stats_, _stats.multinomial_, and probability](week-02-Coding#scipystats) (and `np.random.choice`)
    1. [conditional probability Pr(A|B) and independence Pr(A|B)=Pr(A)](week-02-Coding#conditional-probability-and-independence)

**Out of scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as modular code design (with `def` based functions or `classes`)
4. ...such as dictionary iteration (which has been removed from the above material)
5. ...such as text manipulation with `.apply(lambda x: ...)`, `.replace()`, `re` (which are introduced but are generally out of scope for STA130)


## Week 03 Data Visualization

**Tutorial/Homework: Topics**

1. [More Precise Data Types (As Opposed to Object Types)](week-03-Data-Visualization#continuous-discrete-nominal-and-ordinal-categorical-and-binary): continuous, discrete, nominal and ordinal categorical, and binary
2. [Bar Plots and Modes](week-03-Data-Visualization#Bar-plots-and-modes)
3. [Histograms](week-03-Data-Visualization#Histograms)
4. [Box Plots, Range, IQR, and Outliers](week-03-Data-Visualization#Box-plots-and-spread)
5. [Skew and Multimodality](week-03-Data-Visualization#skew-and-multimodality)
    1. [Mean versus Median](week-03-Data-Visualization#skew-and-multimodality)
    2. [Normality and Standard Deviations](week-03-Data-Visualization#normal-distributions)
    3. [Characteristics of a Normal Distribution](week-03-Data-Visualization#characteristics-of-a-Normal-Distribution)

**Tutorial/Homework: Lecture Extensions**

These are topics introduced in the lecture that build upon the tutorial/homework topics discussed above

> Topic numbers below correspond to extensions of topic items above.

2\. [Plotting: Plotly, Seaborn, Matplotlib, Pandas, and other visualization tools.](week-03-Data-Visualization#modern-plotting)\
___ i. [Legends, annotations, figure panels, etc.](week-03-Data-Visualization#legends-annotations-figure-panels-etc)\
3\. [Kernel Density Estimation using Violin Plots](week-03-Data-Visualization#smoothed-histograms)\
5\. [Log Transformations](week-03-Data-Visualization#log-transformations)

**Lecture: New Topics**

This section introduces new concepts that are not covered in the tutorial/homework topics.

1. [Populations](week-03-Data-Visualization#Populations) _from scipy import stats_ 
	1. `stats.multinomial` and `np.random.choice()` 
	2. `stats.norm`, `stats.gamma`, and `stats.poisson`
2. [Samples](week-03-Data-Visualization#Sampling) versus populations (distributions)
3. [Statistical Inference](week-03-Data-Visualization#Statistics-Estimate-Parameters)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above
	1. Expectation, moments, integration, heavy tailed distributions
	2. Kernel functions for kernel density estimation
3. bokeh, shiny, d3, etc...


## Week 04 Bootstrapping

**Tutorial/Homework: Topic**

1. [Simulation](week-04-Bootstrapping#Simulation) (with `for` loops and `from scipy import stats`)
2. [Sampling Distribution of the Sample Mean](week-04-Bootstrapping#VariabilityUncertainty-of-the-Sample-Mean)
3. [Standard Deviation versus Standard Error](week-04-Bootstrapping#Standard-Deviation-versus-Standard-Error)
4. [How n Drives Standard Error](week-04-Bootstrapping#How-n-drives-Standard-Error)

**Tutorial/Homework: Lecture Extensions**

1. [Independent Sampling](week-04-Bootstrapping#Independent-Samples) functions like `df.sample([n=n/frac=1], replace=False)`
    1. [Are Sampling Distributions Skewed?](week-04-Bootstrapping#Are-Sampling-Distributions-Skewed)
    2. [Bootstrapping](week-04-Bootstrapping#bootstrapping)
    3. [Not Bootstrapping](week-04-Bootstrapping#not-bootstrapping)

**Lecture: New Topics**

1. [Confidence Intervals](week-04-Bootstrapping#Confidence-Intervals)
2. [Bootstrapped Confidence Intervals](week-04-Bootstrapping#Bootstrapped-Confidence-Intervals)
3. ["Double" _for_ loops](week-04-Bootstrapping#double-for-loops)
    1. [Proving Bootstrapped Confidence Intervals using Simulation](week-04-Bootstrapping#Proving-Bootstrapping)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as the **Central Limit Theorem (CLT)**, **Law of Large Numbers (LLN)**, and theoretical "x-bar plus/minus about 2 standard errors" confidence intervals (based on the so-called "pivot" form)
4. ... the alternative sampling function `np.random.choice(list_of_options, p, replace=True)` which will be introduced for different purposes later


## Week 05 Hypothesis Testing

**Tutorial/Homework: Topic**

1. [null and alternative hypotheses](https://github.com/pointOfive/STA130_ChatGPT/wiki/week-05-Hypothesis-Testing#Null-and-Alternative-Hypotheses)
2. the sampling distribution under the null
    1. one sample "difference" hypothesis tests with a "no effect" null
3. [p-values](https://github.com/pointOfive/STA130_ChatGPT/wiki/week-05-Hypothesis-Testing#p-values)
4. [one- or two-sided hypothesis tests](https://github.com/pointOfive/STA130_ChatGPT/wiki/week-05-Hypothesis-Testing#One-or-Two-Sided-Hypothesis-Tests)
    
**LEC Extensions / New Topics**

> Topic numbers below correspond to related extensions and developments building from the topic items above.

2\. independent "random" samples\
___ i\. one sample hypothesized parameter value tests\
3\. using p-values\
___ i\. Type I and Type II errors

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. Simulation versus theoretical hypothesis testing frameworks, z-tests and t-tests, parametric versus nonparametric hypothesis testing frameworks, other tests such as Chi-squared or F-tests, etc...


## Week 06 Simple Linear Regression

**LEC 1 New Topics**

1. correlation association (is not causation)
2. y = ax + b
3. predictor, outcome, intercept and slope coefficients, and error terms
4. simple linear regression is a normal distribution

**TUT/HW Topics**

1. `import statsmodels.formula.api as smf`
2. "R-style" formulas and `smf.ols(y~x, data=df)`
3. using `smf.ols(y~x, data=df).fit().summary()`
    1. `.tables[1]`, `.params`, `.fittedvalues`, `.rsquared`
    2. $\hat \beta_k$ versus $\beta_k$
    3. hypothesis testing no linear association "on average"

**LEC 2 New Topics / Extensions**

1. indicator variables
2. two sample group comparisons
3. normality assumption diagnostic
4. one, paired, and two sample tests
5. two sample permutation tests
6. two sample bootstrapping
