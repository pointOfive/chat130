
# Course Textbook: Week 01 Data Summarization

# Simple numerical summaries and Chat is pretty great

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


## Tutorial/Homework: Topics

### import

```python
import pandas as pd # aliasing usage is `pd.read_csv()`
import pandas # original name usaged is `pandas.read_csv()`
from pandas import read_csv # funtion only import usage is `read_csv()`
```

### read_csv

```python
import pandas as pd

# read a local file
pd.read_csv("./path/to/filename.csv") # "." denotes the "current directory"
# the "csv" file "filename.csv" is in subfolders of the "current directory"

# read an online file
pd.read_csv("https://www.url.com/path/to/filename.csv")
```

**Using URL links**

When accessing an online file, you must link to an actual "raw" `.csv` file.
- This link "looks like" an actual `.csv` file, **but it is not**: [https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)
    - If you follow the link you'll see it's some sort of webpage that visualizes a `.csv` file, but it is not the actual "raw" `.csv` file.
- Here is the link to the actual "raw" `.csv` file that can be found on the above page: [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv)
   - If you follow the link you'll see it's now the actual "raw" `.csv` file.

```python
# will not work
failing_url = "https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv"
titanic_df = pd.read_csv(failing_url)
# does not access an actual "raw" `.csv` file

# will work
working_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
titanic_df = pd.read_csv(working_url)
# links directly to an actual "raw" `.csv` file
```

See the [Function/Method Arguments](#functionmethod-arguments) section below for further details!
 

### Missingness I

In Python we generally use the `pandas` library to count missing values.
Specifically, we use the `.isna()` (or equivalently `isnull()` since it is just an **alias** for `.isna()`) followed by `.sum()`.
 
> Sometimes we may additionally use `axis=1` and `.any()`.

Here’s a quick example of how this is done:

```python
import pandas as pd
import numpy as np

# Create a DataFrame with some missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, np.nan, np.nan, 4]
})

# Count the number of missing values in each column
missing_values_per_column = df.isna().sum()

# Count the total number of missing values in the DataFrame
total_missing_values = df.isna().sum().sum()

# Count the number of rows with at least one missing value
rows_with_missing_values = df.isna().any(axis=1).sum()

# Count the number of missing values in each row
missing_values_per_row = df.isna().sum(axis=1)

print(missing_values_per_column)
print("Total missing values:", total_missing_values)
print("Number of rows with at least one missing value:", rows_with_missing_values)
print(missing_values_per_row)
```

For more details regarding "boolean values and coercion", see [Boolean Values and Coercion](#Boolean-Values-and-Coercion).

### Variables and Observations

```python
import pandas as pd

titanic_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')

titanic_df.shape # number of rows and columns
# this is an "attribute" not a "method" so it does not have end with parenthesis `()`

titanic_df.columns # a list of the names of the columns
# also an attribute...

# You can rename columns as desired...
# This would make columns named "a" and "b" uppercase
df.rename(columns={'a': 'A', 'b': 'B'}, inplace=True)

# The `{'a': 'A', 'b': 'B'}` is a "dictionary" `dict()` data type object.
# In dictionary parlance, the lowercase letters in the above example are "keys"
# and the uppercase letters are the "values" which correspond the to the "keys"

# In this case, the code specifies the columns to be renamed (the keys)
# and what their new names should be (the values).
```

**Observations** are usually organized as rows in a dataset. Each observation represents a single entity upon which data has been measured and recorded. For example, if you’re analyzing a dataset of patients in a hospital, each patient would be an observation.

**Variables** are the different things that can be measured and recorded for each entity, and thus usually correspond to the columns in a dataset. So, the **observation** is comprised of all the values in the columns (or **variables**) of a dataset.

> These concepts are discussed in more detail, [here](https://www.statology.org/observation-in-statistics/).

We're likely to intuitively think of an "observation" as a single value, and we often analyze the values of a single column of data which tends to further bolsters the concept that an "observation" can be thought of as a single value. Since an "observation" refers to whatever set of variables we're considering, there is not a problem with this simplified view of things at the moment.

Variables can be [numerical (quantitative) or categorical (qualitative)](https://uniskills.library.curtin.edu.au/numeracy/statistics/data-variable-types/). For instance, a [patient dataset](http://www.statistics4u.info/fundstat_eng/cc_variables.html) might include the variables of age, weight, blood type, etc.

> Missing values in datasets need to be handled carefully during analysis because they can affect the results. Different statistical analyses and tools have their own ways of dealing with missing values, either by ignoring, removing them, or filling them in with estimated values. These techniques are beyond the scope of STA130 so we will not introduce or consider them here.

### Types I

The `.describe()` **method** provides descriptive statistics that summarize **numerical data** in terms of its location (or position) and scale (or spread). Its provides mean, standard deviation, median (50th percentile), quartiles (25th and 75th percentile), and minimum and maximum values.

> The statistic calculations are based on the non-missing values in the data set, and the number of such non-missing values used to calculate the statistics is given by the "count" value returned from `.describe()`.

The `df[column_name].value_counts()` **method** counts the number of each unique value in a column (named `column_name`). This **method** is used for **categorical data** to understand the distribution of categories within a feature. It does not count missing values by default, but it can include them in the counts by instead using `df[column_name].value_counts(dropna=False)`.

Here’s a demonstration using the Titanic dataset.

```python
import pandas as pd

# Load the Titanic dataset
titanic_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')

# Use df.describe() to get descriptive statistics for numerical features
numerical_stats = titanic_df.describe()

# Use df.value_counts() on a categorical column, for example, 'Embarked'
embarked_counts = titanic_df['embarked'].value_counts()
# embarked_counts_withNaN = titanic_df['embarked'].value_counts(dropna=False)

print(numerical_stats)
print(embarked_counts)
```

### Missingness II

```python
# Assuming 'df' is your DataFrame

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop rows that have all missing values
df.dropna(how='all', inplace=True)

# Keep rows with at least 2 non-missing values
df.dropna(thresh=2, inplace=True)

# Remove an entire column named 'col' from df
del df['col']
```

The order in which you remove rows or columns with missing values to some degree determines the number of non-missing values that are "thrown away" when rows and columns are removed... so proceed intentionally and cautiously to when removing data so you don't "unnecessarily" through away data when you're removing rows and columns from a dataset.

> The `del df['col']` expression is a somewhat unusual looking line of `python` code which is the result of the `python dict type` structure underlying `pandas DataFrame objects` which will be addressed in Week 02, and in the "[What are pandas DataFrame objects?](week-02-Coding#what-are-pandas-dataframe-objects)" section of the Week 02 course wiki-textbook.


### Grouping and Aggregation

Grouping and aggregation are powerful concepts in data analysis, particularly with pandas in Python. They allow you to organize data into groups and then perform operations on those groups to extract insights.

**Grouping** refers to the process of organizing data into groups based on some criteria. In pandas, this is done using the `.groupby()` **method**. When you group data, you’re essentially splitting the DataFrame into smaller chunks based on unique values of a specified key column or columns. For example, `df.groupby("col1")` will create a group for each unique value in `"col1"`.

**Aggregation** refers to computing summaries of each of the groups once they're separated. Some examples of aggregation functions are the `.sum()`, `.mean()`, `.min()`, `.max()`, and `.count()` **methods**. When you use `df.groupby("col1")["col2"].describe()` you're doing all of these at once (as well as `np.quantile([25,50,75])`).

> After `df.groupby("col1")` groups the data by unique values in `"col1"`, the subsequent `["col2"]` selects the `"col2"` column from the data, and then for each group the concluding `.describe()` computes the summary statistics for `"col2"` within each group. Namely, the count, mean, standard deviation, minimum, 25% (first quartile), 50% (median), 75% (third quartile), and maximum values for `"col2"` within each group.

Missing values in the grouping column (`"col1"`) will result in a separate group if there are any, while the `.describe()` **method** automatically excludes missing values when calculating descriptive statistics for `"col2"`.


## Tutorial/Homework: Lecture Extensions


### Function/Method Arguments

The `pandas.read_csv` `python` (`pandas`) **function** is used to read a CSV (Comma Separated Values) file as a `pandas DataFrame object` (that can be assigned into a `python` variable). Running the following in a `jupyter notebook cell` will show you that the `pandas.read_csv` **function** can be controlled with a huge number of **arguments**.

```python
import pandas as pd
pd.read_csv? # add ? to the end of a function to see the so-called
# *signature* of the function... that is, all the possible *arguments* of a function
```

> In statistics, the term **parameter** refers to a characterization about a **population** (as will be discussed later in the course); but, in programming, the term **parameters** refers to the kinds of input that can be used to control the behavior of a **function**; whereas, the actual values given to the **parameters** of a **function** are called the **arguments** of the **function**. We will therefore use the term **arguments** to refer to the inputs of a **function**; but, technically, the **arguments** are the values assigned to the **parameters** of a **function** when it is called.

In `python`, **function arguments** are named and can be optional if they have default values. The `filepath_or_buffer` **argument** of the `pd.read_csv` **function** is required (and not optional so it does not have default value). The `encoding` **argument** is optional, and has a default value of `None`, which means that the function uses the system's default character encoding system when reading the file. A common character encoding system is [UTF-8](https://en.wikipedia.org/wiki/UTF-8#:~:text=UTF%2D8%20is%20the%20dominant,8%20encodings%20on%20the%20web), and you could force `pd.read_csv` to expect this encoding by including the **argument** `encoding="utf-8"` into the `pd.read_csv` **function** when it is called. Another (sometimes useful) alternative `encoding="ISO-8859-1"` is demonstrated below.

```python
trickily_encoded_file = "https://raw.githubusercontent.com/pointOfive/STA130_F23/main/Data/amazonbooks.csv"
# remember, "https://github.com/pointOfive/STA130_F23/blob/main/Data/amazonbooks.csv" won't work
# because that's not actually a link to a real CSV file [as you can see if you go to that github page]...

pd.read_csv(trickily_encoded_file, encoding='UTF-8') # fails
#pd.read_csv(trickily_encoded_file) # fails, because it defaults to UTF-8
#pd.read_csv(trickily_encoded_file, encoding="ISO-8859-1")# works!
```

There are likely many `pandas` **arguments** that you will find useful and helpful. For `pd.read_csv` some **arguments** address some relatively common special cases are

- `sep` for different file types
- `skiprows` and `names` to control column names
- and see more [here](https://note.nkmk.me/en/python-pandas-read-csv-tsv/)\
  (because `pd.read_csv?` will probably be more confusing that helpful the first few times you look at it...)

Moving beyond `pd.read_csv`, we've already seen many useful **arguments**

- `df[column_name].value_counts(dropna=False)` included a count of the number of missing observations along with the counts of the unique values in `column_name`
- `df.rename(columns={'a': 'A', 'b': 'B'}, inplace=True)` allowed us to rename the columns of an existing data frame
    - where `inplace=True` updated `df` without requiring the reassingment `df = df.rename(columns={'a': 'A', 'b': 'B'})`, and this behavior is also present in `df.dropna(inplace=True)`
- and `df.isna().any(axis=1).sum()`, `df.dropna(how='all', inplace=True)`, and `df.dropna(thresh=2, inplace=True)` all provided useful control over how data missing data was removed from a `pandas DataFrame object`

```python
df = pd.read_csv(tricky_file, encoding="ISO-8859-1")
# df = df.dropna() # instead of this, just use
df.dropna(inplace=True) # where the so-called "side-effect" of this *method*
# is to transform the `df` object into an updated form without missing values

# We typically think of "functions" as "returning values or object"
# as in the case of `df = df.dropna()`; but, `df.dropna(inplace=True)`
# demonstrates that functions can operate in terms of "side-effects" on objects as well...
```

> Technically, `pd.read_csv` is a **function** while `df.value_counts(...)`, `df.rename(...), `df.dropna(...)`, `df.isna()...`, etc., are all **methods**. The reason for the difference is that a **method** is a "function" that belongs to the `df` `pandas DataFrame object`. You can think of a **method** like `.dropna(...)` as a "function" who's first (default) **argument** is the `df` `pandas DataFrame object`.
>
> We have already used the term **method** above (without explicitly defining it) in sections
> - [Variables and Observations](week-01-Data-Summarization#Variables-and-Observations)
> - [Types I](week-01-Data-Summarization#Types-I)
> - [Missingness II](week-01-Data-Summarization#Missingness-II)
> - [Grouping and Aggregation](week-01-Data-Summarization#Grouping-and-Aggregation)


### Boolean Values and Coercion

In pandas, **boolean** values are represented as `True` or `False`. When you use the `.isna()` method on a DataFrame, it returns a DataFrame of boolean values, where each value is `True` if the corresponding element is missing and `False` otherwise.

**Coercion** refers to the conversion of one data type into another. In the case of `df.isna().sum()`, coercion happens implicitly when the `.sum()` method is called. The `.sum()` method treats `True` as `1` and `False` as `0` and sums these integer values. This is a form of coercion where boolean values are coerced into integers for the purpose of performing arithmetic operations.

Here’s an example to illustrate this:

```python
import pandas as pd
import numpy as np

# Create a DataFrame with some missing values
df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4, 5, np.nan]
})

# Apply isna() to get a DataFrame of boolean values
bool_df = df.isna()

# Coerce boolean values to integers and sum them
missing_values_sum = bool_df.sum()

print(bool_df)
print(missing_values_sum)
```

In the output, `bool_df` shows that in data in the DataFrame is boolean after applying `.isna()`. The `missing_values_sum` shows the sum of missing values per column, where `True` values have been coerced to `1` and summed up.

> For more details regarding "counting missing values", see [Missingness I](week-01-Data-Summarization#Missingness-I) and [Missingness II](week-01-Data-Summarization#Missingness-II). For additional examples creating boolean values using logical conditionals, see [Logical Conditionals and Boolean Selection/Subsetting](week-01-Data-Summarization#logical-conditionals-and-boolean-selection-subsetting) below.


### _pandas_ column data _types_

As demonstrated below

- the `.dtypes` **attribute** defines the `type` of data that is stored in a `pandas DataFrameObject` column
- while `.astype()` **method** is used to convert data types to specific formats more specifically suite the nature of the data column

The `.dtypes` **attribute** of a `pandas DataFrame object` provides the data type of each column. This is useful for identifying whether columns are numerical (e.g., `int64`, `float64`) or categorical (e.g., `object`, `category`).

```python
import pandas as pd

# Sample DataFrame
data = {
    'age': [25, 32, 47, 51],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'income': [50000, 60000, 70000, 80000],
    'has_pet': ['yes', 'no', 'no', 'yes']
}
# Here, `age` and `income` are numerical (integers), while `name` and `has_pet` are given the `type` of `object`
#   (which in the case of `has_pet` could be interpreted as **categorical** data; whereas
#    `name` is probably better interpreted as an identifier rather than a "cateogory")

df = pd.DataFrame(data)
df.dtypes
```

The `.astype()` **method** is used to convert the data type of a column to another type. For instance, you might want to convert a column from `object` to `category` (a more memory-efficient way to store **categorical data**) or convert a **numerical** column to `float64` if you need to include decimal points.

```python
# Convert the type of 'has_pet' to "categorical" and the type of 'income' to "float"
df['has_pet'] = df['has_pet'].astype('category') # `has_pet` is now of type `category`
df['income'] = df['income'].astype('float64') # `income` has been converted from `int64` to `float64`, allowing for decimal points.
# Alternatively, this could be done using `new_types = {'has_pet': 'category', 'income': 'float64'}`
# df = df.astype(new_types)  # just like how `.rename()` can be used to change column names
df.dtypes
```

Tying these "data" types back to the `.describe()` and `...value_counts()` **methods** addressed in the [Types I](week-01-Data-Summarization#Types-I) section above

- `df['income'].astype('float64').describe()` is appropriate since this is a numeric data type
- `df['has_pet'].astype('category')` is appropriate since this is a non-numeric (**categorical**) data type

> Something that you might like to do here is use `inplace`, e.g., `df['has_pet'].astype('category', inplace=True)`, but **this will not work**(!) because the `.astype()` does not have an `inplace` parameter because it returns a new `pandas DataFrame` or `Series object` with the converted data type. So, the typical usage is to reassign the result back to the original column, as shown in the examples above.
>
> For methods that do support `inplace`, such as `drop()`, `fillna()`, or `replace()`, the `inplace=True` parameter modifies the original DataFrame without creating a new one. Since `.astype()` doesn't support `inplace`, you need to explicitly assign the result to the column you want to change.

  
### Some Statistics Calculations

The `.describe()` method in `pandas` provides a quick statistical summary of numerical columns in a `pandas DataFrame object`. It computes several key statistics, which are especially useful for understanding the nature of the distribution of the data (such as its usual values and the general spread of values around this usual value, as will be discussed later in Week 03). Here's an explanation of the statistical functions computed by `.describe()` and their corresponding programatic or mathematical notations as applicable.

- **Count**: `df['col'].notna().sum()`

  The number of non-missing entries in each column, generally referenced mathematically as $n$



- **Sample Mean**: `df['x'].mean()`

  The average value of the entries, generally notated and computed (where $i$ "indexes" the observations) as

  $$\bar x = \frac{1}{n} \sum_{i=1}^{n} x_i$$



- **Sample Standard Deviation**: `df['x'].std()`

  A measure of the spread (or dispersion) of the values which is "the ([geometric mean](https://en.wikipedia.org/wiki/Geometric_mean)) average distance of points away from the sample mean $\bar x$" defined by the formula (where $i$ "indexes" the observations)

  $$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

  > where $n-1$ is used rather than $n$ for "technical purposes" related to so-called "estimator bias" which is a topic to be addressed in a more advanced statistics course and is beyond the scope of STA130.
  >
  > The **sample variance** is the **squared standard deviation**
  >
  > $$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$



- **Minimum**: `df['x'].min()`

  The smallest value in the column, notated mathematically (where $i$ "indexes" the observations) as

  $$\min_{i} x_i = \min(x_1, x_2, \ldots, x_n)$$



- **25th Percentile (25%)**: `df['x'].quantile(0.25)`

  The value below which 25% of the data falls, often notated mathematically as $Q_1$



- **Median / 50th Percentile (50%)**: `df['x'].quantile(0.5)`

  The middle value in the data set, dividing the data into two equal halves such that 50% of the data falls below this value, usually referred to as the **median** (rather than $Q_2$)


- **75th Percentile (75%)**: `df['x'].quantile(0.75)`

  The value below which 25% of the data falls, often notated mathematically as $Q_3$



- **Minimum**: `df['x'].min()`

  The largest value in the column, notated mathematically (where $i$ "indexes" the observations) as 

  $$\max_{i} x_i = \max(x_1, x_2, \ldots, x_n)$$

These are **statistics** for **numeric** data; whereas, the `df['x'].value_counts()` **method** returns the count of each unique value in the data and so is contrastingly appropriate when column `x` contains non-numeric (**categorical**) data. Using `df['x'].value_counts(dropna=False)` will additionally includes the number of missing values in the column in the returned counts; whereas, to determine this for **numeric** variables in the context of `df.describe()` would require a relative comparison to `df.shape` or `df['x'].size`.


## Lecture: New Topics


### Sorting and _.iloc_ Indexing

We can look into `pandas DataFrame object` datasets, such as the one introduced in LEC

```python
import pandas as pd
url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
df = pd.read_csv(url)  # Load the data
df.head()  # Display the first few rows
```

by **sorting** by the values within a column using `df.sort_values(by=['col1','col2])`

```python
df_Name_sorted = df.sort_values(by='Name')  # or
df_Type12_sorted = df.sort_values(by=['Type 1','Type 2']) 
df_Type12_sorted
```

and then **selecting** specific rows (and/or columns) in the data by indexing into the `.iloc` ("index location") **attribute**

```python
start_row = 50
end_row_plus_1 = start_row + 10  # this example will select 10 rows
# This takes the row from index `start_row` up to (but not including) `end_row_plus_1`
df_Name_sorted.iloc[start_row:end_row_plus_1, :]  # and ":" in the second position means "all columns"
# df_Name_sorted.iloc[:, 1:3]  # "all rows" but columns 2 and 3... wait, what?
```

Python is `0`-indexed, which means the first row is in index position `0` (for the rows), and similarly the first column is in index position `0` (for the columns). So `1:3` means take the 2nd and 3rd index position (of either the rows or columns, depending on which position it is in the square brackets `[rows, cols]`, so `[:, 1:3]` above references the columns).

Now look again at the output of `df_Name_sorted.iloc[start_row:end_row_plus_1, :]`
- There is a "column" without a name on the far left of the printout (which doesn't quite match the column named `#`) that can be accessed through the `.index` **attribute**\
 `df_Name_sorted.iloc[start_row:end_row_plus_1, :].index`

- But notice that the code in question doesn't correspond to the numbers in the `.index` **attribute**...\
  This is because the `.iloc` **attribute** is based on the actual (`0`-indexed) row numbers of the `pandas DataFrame object` as it currently exists, not the numbers in the `.index`. Here, the sorting of `df_Name_sorted` has resulted in the "shuffling" of the `.index` **attribute** relative to its original order (which can be seen by looking at the initial `df` object).

It's important to keep remember that the `.iloc` and `.index` **attributes** don't refer to the same thing, especially since they often initially appear to (and seem to be named in way that suggests they "should and would").

```python
# At first `df.index` is 0, 1, 2, 3, 4, ... so
df.iloc[0:5, :].index  # is also still (0, 1, 2, 3, 4)
# But now `df.dropna().index` is 0, 1, 2, 3, 6(!), ... so now
df.dropna().iloc[0:5, :].index  # is actually (0, 1, 2, 3, 6) instead of "indexes" (0, 1, 2, 3, 4) corresponding to "0:5"
```

### Logical Conditionals, Boolean Selection/Subsetting, and _.loc_ indexing V2

Sorting alphabetically (or even numerically) and then subsetting by actually row index numbers is going to end up feeling pretty tedious. Fortunately, we can use **logical conditionals** to **subset** to only the parts of a dataset we are interested in by using so-called **boolean selection**.

```python
# Inequality comaprison options are `>=`, `>`, `<`, and `<=`
df_100plusHP = df[ df.HP >= 100 ]  # creates boolean selection with a logical conditional; or,
df_Legendary = df[df.Legendary]  # already a boolean selection so no logical conditional needed; or
# Opposite of a boolean uses `~`; so, `df_NotLegendary = df[~df.Legendary]`
df_Fire = df[ df['Type 1'] == 'Fire' ]
# Opposite of `==` uses "!-"; so, df_NotFire = df[  df['Type 1'] == 'Fire' ]
df_Fire
```

If you want to subset to just some specific columns as well, you'll need to use the `.loc` **attribute** (as opposed to the `.iloc` **attribute**).

```python
df.loc[ df['Type 1'] == 'Fire', ["Name", "Attack", "Defense"] ]
# df.iloc[df['Type 1'] == 'Fire', ["Name", "Attack", "Defense"]] won't work(!)
# because `i` in `iloc` really needs index numbers, not "names" or "boolean selections"
```

And more complex **logical conditionals** are available through the "and" `&` and "or" `|` operations

```python
df.loc[ (df['Type 1'] == 'Fire') & (df.HP >= 100) , ["Name", "Attack", "Defense"] ]
df.loc[ (df['Type 1'] == 'Fire') & ((df.HP >= 100) | df.Legendary) , ["Name", "Attack", "Defense", "Legendary"] ]
df.loc[ ~(df['Type 1'] == 'Fire') & ((df.HP >= 100) & df.Legendary) , ["Name", "Type 1", "Attack", "Defense", "Legendary"] ]
```


