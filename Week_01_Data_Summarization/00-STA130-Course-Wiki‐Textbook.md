# 00 Tools

1. [GitHub](https://github.com/pointOfive/STA130_ChatGPT/blob/main/README.md)
2. [UofT Jupyterhub](https://datatools.utoronto.ca) (or[google collab](https://colab.research.google.com/)) `.ipynb` notebook files
3. [Copilot](https://copilot.microsoft.com/) (or [ChatGPT](https://chat.openai.com/))
4. Course [wiki-textbook](https://github.com/pointOfive/STA130_ChatGPT/wiki/)


# 01 Data Summarization

**TUT/HW Topics**

1. importing libraries... like [`pandas`](01-Data-Summarization#import)
2. loading data... with [`pd.read_csv()`](01-Data-Summarization#read_csv)
3. counting missing values... with [`df.isna.sum()`](01-Data-Summarization#Missingness-I)
4. observations (rows) and variables (columns)... [`df.shape`](01-Data-Summarization#Variables-and-Observations) and [`df.columns`](01-Data-Summarization#Variables-and-Observations)
5. numeric versus non-numeric... [`df.describe()`](01-Data-Summarization#Types-I) and [`df.value_counts()`](01-Data-Summarization#Types-I)
6. removing missing data... with [`df.dropna()`](01-Data-Summarization#Missingness-II) and [`del df['col']`](01-Data-Summarization#Missingness-II)
7. grouping and aggregation.... with [`df.groupby("col1")["col2"].describe()`](01-Data-Summarization#Grouping-and-Aggregation)

**LEC Extensions**

2. [function/method arguments](01-Data-Summarization#functionmethod-arguments)
    1. ~function side-effects~
3. [boolean values and coercion](01-Data-Summarization#Boolean-Values-and-Coercion)
5. 
    1. [`.dtypes` and `.astype()`](01-Data-Summarization#pandas-column-data-types)
    2. dictionary `dict()` objects
    3. statistic calculation functions

**LEC New Topics**

1. sorting and (0-based) indexing
2. subsetting and boolean selection

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as how to handle missing values using more advanced techniques that don't just "ignore" or "remove" them (for example by filling or imputing the missing values and assumption required when doing so)
4. ...further "data wrangling topics" such as "joining" and "merging"; "pivoting", "wide to long", and "tidy" data formats; etc.


# 02 Coding

**TUT/HW Topics**

1. first data types... [`tuple`](02-Coding#Types-II), [`list`](02-Coding#Types-II), [`dict`](02-Coding#Types-II)
2. another key data type... [`np.array`](02-Coding#np.array) [and `np.random.choice`]
3. loops... [`for i in range(n)`](02-Coding#for-loops) and [`print()`](02-Coding#for-loops)
4. logical flow control... with [`if`](02-Coding#Logical-Flow-Control)/[`else`](02-Coding#Logical-Flow-Control)
    1. [`try-except` blocks](02-Coding#Logical-Flow-Control)

**LEC Extensions / New Topics**

1. 
    1. "object" `type()` [as opposed to "data type" (like numerical or categorical)]
    2. `str` (and `sentence.split()`) as opposed to `int` versus `float` versus `bool`
    3. operator overloading polymorphism with `+` and `.sum()`
2. `from scipy import stats`, `stats.multinomial`, and probability [and `np.random.choice`]
    1. conditional probability Pr(Y=y|X=x)
3.
    1. [`for x in lst`](02-Coding#More-Loops) and [`for word in sentence.split()`](02-Coding#More-Loops) and 
        1. ~text manipulation with `.apply(lambda x: ...)`, `.replace()`, `re`~
    2. more loops... such as [`for i,x in enumerate(a_list)`](02-Coding#More-Loops)
        1. ~`for key,val in dictionary.items()` and `dictionary.keys()` and `dictionary.values()`~
4. [`elif`](02-Coding#Logical-Flow-Control)

**Out of scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as modular code design (with `def` based functions or `classes`)
4. ...such as dictionary iteration (which has been removed from the above material)
5. ...such as text manipulation with `.apply(lambda x: ...)`, `.replace()`, `re` (which are introduced but are generally out of scope for STA130)


# 03 Data Visualization

**TUT/HW Topics**

1. ["types of data"](03-Data-Visualization#Types-III)... continuous, discrete, nominal and ordinal categorical, and binary
2. [bar plots](03-Data-Visualization#Bar-plots-and-modes) and the [mode](03-Data-Visualization#Bar-plots-and-modes)    
3. [histograms](03-Data-Visualization#Histograms)
4. [box plots](03-Data-Visualization#Box-plots-and-spread), [range](03-Data-Visualization#Box-plots-and-spread), [IQR](03-Data-Visualization#Box-plots-and-spread) and [outliers](03-Data-Visualization#Box-plots-and-spread)
5. [skew](03-Data-Visualization#skew-and-multimodality) and [multimodality](03-Data-Visualization#skew-and-multimodality) 
    1. [mean versus median](03-Data-Visualization#skew-and-multimodality)
    2. [normality and standard deviations](03-Data-Visualization#skew-and-multimodality)
    
**LEC Extensions**

2. plotting... plotly, seaborn, matplotlib, pandas
3. kernel density estimation "violin plots"
4. legends, annotations, figure panels
5. log transformations

**LEC New Topics**

1. populations [`from scipy import stats`](03-Data-Visualization#Populations) (re: `stats.multinomial` and `np.random.choice()`) with `stats.norm` and `stats.poisson`
2. [samples](03-Data-Visualization#Sampling) from populations (distributions) 
3. [statistics estimate parameters](03-Data-Visualization#Statistics-Estimate-Parameters)


**Out of scope**
1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as expectation, moments, integration, heavy tailed distributions...
4. ...such as kernel functions for kernel density estimation
5. ...bokeh, shiny, d3, ...

# 04 Bootstrapping

**TUT/HW Topics**

1. [simulation](04-Bootstrapping#Simulation) (with `for` loops and `from scipy import stats`)
2. [the sampling distribution of the mean](04-Bootstrapping#Variability/Uncertainty-of-the-Sample-Mean)
3. [standard deviation versus standard error](04-Bootstrapping#Standard-Deviation-versus-Standard-Error)
4. [how standard error is driven by n](04-Bootstrapping#How-n-Drives-Standard-Error)

**LEC Extensions**

1. `df.sample(n=n_, frac=1, replace=False)`
    1. skewed distributions
    2. bootstrapping
    3. not bootstrapping

**LEC New Topics**

1. confidence (and confidence levels)
    1. "double" `for` loops
2. bootstrapped confidence intervals

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as the **Central Limit Theorem (CLT)**, **Law of Large Numbers (LLN)**, and theoretical "x-bar plus/minus about 2 standard errors" confidence intervals (based on the so-called "pivot" form)
4. ... the alternative sampling function `np.random.choice(size, p, replace=True)` which will be introduced for different purposes later


# 05 Hypothesis Testing

**TUT/HW Topics**

1. [null and alternative hypotheses](https://github.com/pointOfive/STA130_ChatGPT/wiki/05-Hypothesis-Testing#Null-and-Alternative-Hypotheses)
2. the sampling distribution under the null 
    1. one sample "difference" hypothesis tests with a "no effect" null
3. [p-values](https://github.com/pointOfive/STA130_ChatGPT/wiki/05-Hypothesis-Testing#p-values)
4. [one- or two-sided hypothesis tests](https://github.com/pointOfive/STA130_ChatGPT/wiki/05-Hypothesis-Testing#One-or-Two-Sided Hypothesis-Tests)
    
**LEC Extensions / New Topics**

2. independent "random" samples
    1. one sample hypothesized parameter value tests
3. using p-values
    1. Type I and Type II errors

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. Simulation versus theoretical hypothesis testing frameworks, z-tests and t-tests, parametric versus nonparametric hypothesis testing frameworks, other tests such as Chi-squared or F-tests, etc...


# 06 Simple Linear Regression

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
4. two sample permutation tests
5. two sample bootstrapping