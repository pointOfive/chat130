
# Course Textbook: Week 01 Data Summarization

**TUT/HW Topics**

1. importing libraries... like [_pandas_](#import)
2. loading data... with [_pd.read_csv()_](#read_csv)
3. counting missing values... with [_df.isna().sum()_](#Missingness-I)
4. observations (rows) and variables (columns)... [_df.shape_](#Variables-and-Observations) and [_df.columns_](#Variables-and-Observations)
5. numeric versus non-numeric... [_df.describe()_](#Types-I) and [_df.value_counts()_](#Types-I)
6. removing missing data... with [_df.dropna()_](#Missingness-II) and [_del df['col']_](n#Missingness-II)
7. grouping and aggregation.... with [_df.groupby("col1")["col2"].describe()_](#Grouping-and-Aggregation)

**LEC Extensions**

> Topic numbers below correspond to extensions of topic items above.

2\. [function/method arguments](#functionmethod-arguments) (like `encoding`, `dropna`, `inplace`, and return vs side-effect)\
3\. [boolean values and coercion](#Boolean-Values-and-Coercion)\
5\. _ i. [_.dtypes_ and _.astype()_](#pandas-column-data-types)\
___ ii. [statistic calculation functions](#some-statistics-calculations) 

**LEC New Topics**

1. [sorting and (0-based) indexing](#sorting-and-iloc-indexing)
2. [subsetting via conditionals and boolean selection](#logical-conditionals-boolean-selectionsubsetting-and-loc-indexing-v2)

**Out of Scope**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as how to handle missing values using more advanced techniques that don't just "ignore" or "remove" them (for example by filling or imputing the missing values and the assumptions required when doing so...)
4. ...further "data wrangling topics" such as "joining" and "merging"; "pivoting", "wide to long", and "tidy" data formats; etc.

## TUT/HW Topics

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


## LEC Extensions


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

# Course Tutorial: Week 01 TUT

## STA130 TUT 01 (Sep06)<br><br> 🏃🏻‍♀️ 🏃🏻 <u> Hitting the ground running... <u>


### 🚧 🏗️ (Using notebooks and ChatBots) Demo [45 minutes]  
      
#### 1. *[About 8 of the 45 minutes]* Demonstrate going to the course [Quercus homepage](https://q.utoronto.ca/courses/354091); accessing the [Course GitHub Repo](https://github.com/pointOfive/STA130_ChatGPT); opening new and uploaded notebooks on [UofT Jupyterhub](https://datatools.utoronto.ca) (classic jupyter notebook, or jupyterhub is fine, or students may use [google collab](https://colab.research.google.com/)); and using Jupyter notebooks as a "`Python` calculator" and editing ["Markdown cells"](https://www.markdownguide.org/cheat-sheet/)<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> This is all simple and pretty obvious intended to be that way so students get that they can do this on their own.

</details>
    
#### 2. *[About 30 of the 45 minutes]* Demonstrate using [ChatGPT](https://chat.openai.com/) (or [Copilot](https://copilot.microsoft.com/) if conversation privacy is desired) to

1. find an (a) "amusing, funny, or otherwise interesting dataset" which (b) has missing values and is (c) available online through a URL link to a csv file;
2. load the data into the notebook with `pandas` and get "missing data counts" for the dataset;
3. prompt the ChatBot to "Please provide a summary of our interaction for submitting as part of the requirements of an assignment"; and, to "Please provide me with the final working verson of the code that we created"<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> The intention here is to demonstrate that [at the current GPT4.0-ish level] the ChatBot (with probably about an 80% chance) **cannot fullfil all the requests of the inquiry (of (a) "funny or amusing" nature, (b) the presence of missingness, and (c) working url links)** *but will otherwise produce working code*<br><br>
> 
> 1. ChatBots have a notoriously "short term memory"; so, be ready for them to "forget" specific details of your prompting requests 
> 2. ChatBots often cannot pivot away substantially from initial answers; so, be ready for your efforts at follow up and correction with the ChatBot to prove frustratingly futile (which, may in this case actually have a lot to do with the following fact, that...)
> 3. ChatBots don't seem to be very aware of the contents of datasets that are avalable online (or even working url links where datasets are); so, ChatBot are not currently a substitue for exploring dataset repository such as [TidyTuesday](https://github.com/rfordatascience/tidytuesday) (or other data repositiory resources) and reviewing data yourself (although, ChatBot interactions can nonetheless be help with brainstorm dataset ideas and provide a way to "search for content", perhaps especially when referencing a specific website in the conversation)<br><br>
> 
> Examples of this task going pretty well are available [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00006_copilot_funnyamusingNAdatasetV3.md), [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00007_copilot_funnyamusingNAdatasetV4.md), and [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00001_gpt3p5_villagersdata.md); while, examples of this going poorly are available [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00002_copilot_funnyamusingNAdataset.md) and [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00002_gpt3p5_funnyasusingNAdataset.md). Successes and failures are found within the Microsoft Copilot and ChatGPT ChatBots both, suggesting the quality of the results likely has to do more with "randomness" and perhaps the nature of the prompting and engagement as opposed to the actual ChatBot version being used...
    
</details>

#### 3. *[About 7 of the 45 minutes]* Demonstrate saving your python jupyter notebook in your own account and "repo" on [github.com](https://github.com), and sharing (a) notebook links, (b), ChatBot transcript log links, (c) ChatBot summaries through a piazza post and a Quercus announcement (so students can use this later for their homework assignment if they wish)<br><br>


### 💬 🗣️ Communication [55 minutes]  
     
#### 1. *[About 15 of the 55 minutes]* Ice breakers  and introductions, in 8 groups of 3 or thereabouts...
    
1. Each person may bring two emojis to a desert island... reveal your emojis at the same time... for emojis selected more than once the group should select one additional emoji
2. Where are you from, what do you think your major might be, and what's an "interesting" fact that you're willing to share about yourself?
        
#### 2. *[About 10 of the 55 minutes]* These are where all the bullet holes were in the planes that returned home so far after some missions in World War II
    
1. Where would you like to add armour to planes for future missions?
2. Hint: there is a hypothetical dataset of the bullet holes on the planes that didn't return which is what we'd ideally compare against the dataset we observe...
        
![Classic image of survivorship bias of WW2 planes](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Survivorship-bias.svg/640px-Survivorship-bias.svg.png)
           
#### 3. *[About 10 of the 55 minutes]* Monte Hall problem: there is a gameshow with three doors, one of which has a prize, and you select one of the doors and the gameshow host reveals one of the other two unchosen doors which does not have the prize... would you like to change your guess to the other unchosen door?

![](https://mathematicalmysteries.org/wp-content/uploads/2021/12/04615-0sxvwbnzvvnhuklug.png)<br>
       
#### 4. *[About 20 of the 60 minutes]* Discuss the experience of the groups for the WW2 planes and Monte Hall problems

1. For each of these problems, have students vote on whether their groups (a) agreed on answers from the beginning, (b) agreed only after some discussion and convincing, or (c) retained somewhat divided in their opinions of the best way to proceed<br><br>
    
2. Briefely identify the correct answer from the answers the groups arrived at<br><br>
    
3. **[If time permits... otherwise this is something students could consider after TUT]** Prompt a [ChatGPT](https://chat.openai.com/) [or [Copilot](https://copilot.microsoft.com/)] ChatBot to introduce and explain "survivorship bias" using spotify songs as an example and see if students are able to generalize this idea for the WW2 planes problem and if they find it to be a convincing argument to understand the problem<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> This could be done like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00009_copilot_survivorshipbias_spotify.md) or [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00003_gpt3p5_spotify_Survivorship_Bias.md), or you could instead try to approach things more generally like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00004_gpt3p5_general_Survivorship_Bias.md)
> 
> Two ends of the ChatBot prompting spectrum are
> 
> 1. creating an extensive prompt exhuastively specifying the desired response results; or, 
> 2. iteratively clarifying the desired response results through interactive ChatBot dialogue<br><br>
> 
> This is to some degree a matter of preference regarding the nature of ChatBot conversation sessions, but there it may also be a lever to influence the nature of the responses provided by the ChatBot 
</details>
    
4. **[If time permits... otherwise this is something students could consider after TUT]** Prompt a [ChatGPT](https://chat.openai.com/) [or [Copilot](https://copilot.microsoft.com/)] ChatBot to introduce and explain the Monte Hall problem and see if the students find it understandable and convincing<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> ChatBots fail to correctly analyze the Monte Hall problem when they're asked for a formal probabilistic argument...
>
> - [ChatGPT fails by wrongly calculating a probability of 1/2...](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00005_gpt3p5_MonteHallWrong.md)
> - [Copilot fares similarly poorly without substantial guidance...](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00010_copilot_montehallwrong.md)<br><br>
> 
> *demonstrating (a) that there are clear limits to how deeply ChatBots actually "reason", and (b) that they are instead better understood as simply being information regurgitation machines, and (c) that this means  they can suffer from the "garbage in, garbage out" problem if the quality of the information their responses are based on are is poor and inaccurate (as is notoriously the case in the Monte Hall problem, for which many incorrect mathematical analyses have been "published" into the collection of human generated textual data on which ChatBots are based)*
    
</details>


# Course Lecture: Week 01 LEC


## Data Summarization

- **Functions**, **arguments**, and **attributes** via `pd.read_csv()`<br><br>
    - `inplace="ISO-8859-1"` encoding the amazon data set (a subtle introduction of "data" and "object" types)
    - `.shape` and `.columns` (and in the next section `df["new_columns"] = ...`)<br><br>

- **Methods**, **chaining**, and **coercion** as in `df.isnull().sum(axis=1)`<br><br>
    - "data" types (as opposed to "Object" types which will be discussed formally next week)
    - **numeric** `float64` and `int64`, **categorical** `category`, and **object**
    - `.dtypes` and `.astype()`<br><br>
    
- **Summarizing data** with `df.describe()` and **statistics** (as opposed to **Statistics**)<br><br>

    - $\bar x$ the **sample mean** `df['col'].mean()` 

      $\displaystyle \bar x = \frac{1}{n}\sum_{i=1}^n x_i$<br><br> 

    - $s^2$ the **sample variance (var)** `df['col'].var()` and $s$ the **sample standard deviation (std)**  `df['col'].std()`
      
      $\displaystyle s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2 \quad \text{ and } \quad s = \sqrt{s^2}$<br><br>  
      
    - **min** `df['col'].min()`, **max** `df['col'].max()` (and $Q1$, the **median**, and $Q3$ to be discussed later)<br><br>
    
- **Summarizing data** with `df['categorical_column'].value_counts()` and (one more **statistic**) the **mode**<br><br>

- **Sorting**, **indexing**, and **subsetting**<br><br>

    - `.sort_values()`
    - `df[]` 0-based (row) and (column) name indexing (and "index" versus "row")<br>versus fully 0-based indexing with `df.iloc[]` and "slicing" with `:`
    - versus `df[]` or `df.loc[]` **boolean selection** with **logical conditionals**<br> 
      `>=` / `>` / `<=` / `<` /  and `==` (and `!=` in contrast to `=`) and `~` / `&` / `|` (not/and/or)


### **Functions**, **arguments**, and **attributes**


```python
import pandas as pd

url = "https://raw.githubusercontent.com/pointOfive/STA130_F23/main/Data/amazonbooks.csv"
# fail https://github.com/pointOfive/STA130_F23/blob/main/Data/amazonbooks.csv

# 1. demonstrate local file
# 2. demo some ChatGPT

# a *function* with required and default *arguments*
ab = pd.read_csv(url, encoding='UTF-8') # fails
# ab = pd.read_csv(url) # fails, because it defaults to UTF-8
# ab = pd.read_csv(url, encoding="ISO-8859-1")# works!
ab
```


```python
# *attribute* (not a *method*)
ab.shape
```


```python
# *attribute* (not a *method*)
ab.columns
```

### Chaining and coercion and<br>"data" types (as opposed to "Object" types to be discussed later)



```python
# *methods* (with no *arguments)
ab.isnull().sum()  # missing per column
ab.isa().sum()  # missing per column
```


```python
# *methods* (the latter with an optional *argument*)
ab.isna().sum(axis=1)  # missing per row
```


```python
ab['# missing on row'] = ab.isna().sum(axis=1)
ab
```


```python
ab_isna = ab.isna()
print(ab_isna.dtypes)
ab_isna.head()  # now they're all boolean
```


```python
# Why then are these numbers?
print(ab.isna().sum(), end='\n\n')
ab.isna().sum(axis=1)
```


```python
# This is due to something called *coercion* 
# which implicitly changes the data types in an appropriate manner

# But we can explicitly change the types of data ourselves...

print(ab.dtypes)  # originally they were all... "float" and "object" ?
ab.head()  # and `ab['# missing on row'] = ab.isna().sum(axis=1)` become an "int" ?
```


```python
ab_dropna = ab.dropna()
new_data_types = {'Hard_or_Paper': "category", 
                  'NumPages': int,
                  'Pub year': int}
# rather than doing them separately like 
#ab_dropna_v2['Hard_or_Paper'] = ab_dropna_v2['Hard_or_Paper'].astype("object")

# Demo some ChatGPT?

ab = ab.astype(new_data_types)
#ab_dropna = ab_dropna.astype(new_data_types)
#pd.DataFrame({"Orignal": ab.dtypes, "Adjusted": ab_dropna.dtypes})
```


```python
new_column_names = {k:k+" ("+v+")" for k,v in zip(ab.columns,ab_dropna.dtypes.values.astype(str))}
new_column_names
```


```python
# Use inplace=True rather than ab_dropna = ab_dropna.rename(columns=new_column_names)
ab_dropna.rename(columns=new_column_names, inplace=True)  # if you like
ab_dropna.head()  # "objects" are still not really "categories"
```

### Summarizing data with `df.describe()` and *statistics* (as opposed to *Statistics*)

The sample mean, sample variance, and sample standard devation are examples of **statistics** which are important in the discipline of **Statistics**
$$\huge \displaystyle \bar x = \frac{1}{n}\sum_{i=1}^n x_i \quad\quad\quad \displaystyle s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2 \quad\quad\quad s=\sqrt{s^2}$$ 



```python
url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# fail https://github.com/KeithGalli/pandas/blob/master/pokemon_data.csv
pokeaman = pd.read_csv(url)
colnames_wtype = {k:k+" ("+v+")" for k,v in zip(pokeaman.columns,pokeaman.dtypes.values.astype(str))}
pokeaman.rename(columns=colnames_wtype, inplace=True)
pokeaman
```


```python
# Why does this not have all the columns?
pokeaman.describe()  # more coercion... if you see it?
```

Because these are summaries for **numieric** data types...

- $\bar x$ the **sample mean** `df['col'].mean()` 

  $\displaystyle \bar x = \frac{1}{n}\sum_{i=1}^n x_i$ 

- $s$ the **sample standard deviation (std)** `df['col'].std()`

  $\displaystyle s = \sqrt{s^2}$

  > $s^2$ the **sample variance (var)** `df['col'].var()`
  >  
  > $\displaystyle s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2$      
        
- and where **min** `df['col'].min()` and **max** `df['col'].max()` are (hopefully) obvious
- and **25%, 50%, and 75%** are the first, second, and third **quantiles** referred to as $Q1$, the **median**, and $Q3$ (but these will not be discussed later)


### **Summarizing data** with<br><br>`df['categorical_column'].value_counts()`


```python
# Another "explanation" as to why `.describe()` doesn't have all the columns is "because"
# ...obviously this is not an explanation... it's just an example of why 
# `.value_counts()` is what we should use for categorical data
pokeaman['Type 1 (object)'].value_counts()
# where the most frequently occuring value is called the *mode*
```


```python
# And where the `dropna=False` *argument* can be added to include a count of missing values
pokeaman['Type 2 (object)'].value_counts(dropna=False)  # 'Type 1 (object)' doesn't have NaNs
```

### Sorting, indexing, and subsetting<br>OR

#### `.sort_values()`<br><br>`df[]` 0-based (row) and (column) name indexing (and "index" versus "row")<br><br>$\quad$versus fully 0-based indexing with `df.iloc[]` and "slicing" with `:`<br><br>$\quad\quad$versus `df[]` or `df.loc[]` **boolean selection** with **logical conditionals**<br><sub>$\quad\quad\;\;$ `>=` / `>` / `<=` / `<` /  and `==` (and `!=` in contrast to `=`) and `~` / `&` / `|` (not/and/or)</sub>



```python
colnames_wotype = {col: col.split(" (")[0] for col in pokeaman.columns.astype(str)}
pokeaman.rename(columns=colnames_wotype, inplace=True)
```


```python
# sorting
pokeaman.sort_values("Attack", ascending=False) 
```


```python
# indexing V1: 0-based (row) and (column) name indexing 
# [row_sequence_subset] or [column_name_list] or [row_sequence_subset][column_name_list]
# pokeaman[:10]
# or try 
# pokeaman[['Name','Type 1']] # but note that `pokeaman['Name','Type 1']` won't work(!)
pokeaman[:10][['Name','Type 1']]
```


```python
# (and "index" versus "row")
pokeaman.dropna()[:10][['Name','Type 1']]
```


```python
# indexing V2: fully 0-based indexing with df.iloc[] and "slicing" with `:`
# [ rows , cols ] specifically [ rowStart : rowEndPlus1 , colstart : rowEndPlus1 ]

pokeaman.iloc[ :10 , : ]  # pokeaman[:10]
pokeaman.iloc[ 0:10 , : ]  # pokeaman[0:10]
pokeaman.iloc[ 10:20 , 1:3 ]  # pokeaman[10:20][['Name','Type 1']]
```


```python
# (and "index" versus "row")
pokeaman.dropna().iloc[ :10 , 1:3 ] 
```


```python
# (and "index" versus "row")
pokeaman.sort_values(["Attack","Defense"], ascending=[False,True]).iloc[ :10, : ]
```


```python
# indexing V3: *boolean selection* with *logical conditionals*
# indexing V3: df[] or df.loc[ logical_conditional , colname_list ] 
#                   or df.loc[ row_based_indexing , colname_list ] 

pokeaman.Legendary
pokeaman[pokeaman.Legendary]
```


```python
# (and "index" versus "row")
pokeaman.dropna().loc[ :10 , ['Name','Type 1'] ]  # or just rows 
```


```python
~pokeaman.Legendary
pokeaman[~pokeaman.Legendary]
```


```python
(pokeaman["HP"] > 80)  # what would `~(pokeaman["HP"] > 80)` be?
pokeaman[ pokeaman["HP"] > 80 ]
```


```python
# (pokeaman["HP"] > 80) & (pokeaman["Type 2"] == "Fighting")
pokeaman[ (pokeaman["HP"] > 80) & (pokeaman["Type 2"] == "Fighting") ]
```


```python
# something like `pokeaman.Type 2` wouldn't work... why?
pokeaman.loc[~(pokeaman.HP > 120) | (pokeaman.Defense > 180)]
# pokeaman.query("HP > 120 and Legendary == True")
```

> There's probably not time, but if there is... we could review/demo the pokemon data set a little bit more, with more complex *chaining* like `df.dropna().groupby('col1').describe()`?


```python
pokemon.describe()
```


```python
pokemon[["Type 1","Type 2"]].value_counts()
```


```python
pokemon.groupby('Type 1').describe()
```


```python
pokemon.groupby('Type 1').describe().columns
```


```python
pokemon.groupby('Type 1').describe().sort_values(('HP','mean'), ascending=False)
```


```python
pokemon.groupby('Type 1').mean(numeric_only=True).round(3)
```


# Course HW: Week 01 HW


## STA130 Week 01 Homework 

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
    
1. Code and write all your answers (for both the "Prelecture" and "Postlecture" HW) in a python notebook (in code and markdown cells) 
    
> It is *suggested but not mandatory* that you complete the "Prelecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
    
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
- [0.2 points]: Reasonable well-written general definitions for Question "2.2"
- [0.3 points]: Demonstrated understanding regarding Question "4"
<!-- - [0.2 points]: A sensible justification for the choice in Question "7.4" -->
- [0.4 points]: Requested assessment of ChatBot versus google performance in Question "8.3"


### "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]

#### 1. Pick one of the datasets from the ChatBot session(s) of the **TUT demo** (or from your own ChatBot session if you wish) and use the code produced through the ChatBot interactions to import the data and confirm that the dataset has missing values<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> If your TA has not shared a relevant ChatBot session from their **TUT demo** through a piazza post and a Quercus announcement, the **TUT notebook** has links to example ChatBot sessions that you can use; or, ...
> 
> ```python
> # feel free to just use the following if you prefer...
> import pandas as pd
> url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv"
> df = pd.read_csv(url)
> df.isna().sum()
> ```
    
</details>

#### 2. Start a new ChatBot session with an initial prompt introducing the dataset you're using and request help to determine how many columns and rows of data a `pandas` DataFrame has, and then

1. use code provided in your ChatBot session to print out the number of rows and columns of the dataset; and,  
2. write your own general definitions of the meaning of "observations" and "variables" based on asking the ChatBot to explain these terms in the context of your dataset<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> A good initial prompt to start would be be something like
> - "I've downloaded a dataset about characters from animal crossings (from https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/villagers.csv), and I'd like to know what columns of information I have and how much data I have"
> 
> You can further reduce the scope of your inquiry with if needed with something like
> - "I've already downloaded the data and want to understand the size (or dimensions) of the dataset to start with"
> 
> *Some ChatBots can upload your data and do this for you; but, extended usage of this feature [likely requires a paid subscription](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00006_gpt3p5_LoadDataPaywall.md); and, anyway, you need to run the code yourself rather than having a ChatBot do that for you; and, for STA130 we don't want a ChatBot to just do the analysis for us; rather, we instead want ChatBots to help us understand the steps we need to take to analyze the data; so,* **you DO NOT need to purchase an upgraded version of any ChatBots**
> - Free-tier level ChatBots like [GPT4o-mini](https://chat.openai.com/) or [Copilot](https://copilot.microsoft.com/) (which is partially based on [ChatGPT4.0](https://chat.openai.com/), and which you have access to through your UofT account) are sufficiently sophisticated and perfectly appropriate for the STA130 course
    
</details>

#### 3. Ask the ChatBot how you can provide simple summaries of the columns in the dataset and use the suggested code to provide these summaries for your dataset<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> Use your ChatBot session to help you create working examples of using  `df.describe()` and `df['column'].value_counts()` for your dataset (although note that the `.value_counts()` method is not really meant to be used for numeric variables, so if you dataset has only numeric variables, `.value_counts()` might not be particularly informative...)
>
> **ChatBot Response Scope**
>     
> If prompts are not sufficiently focused you will likely get overly broad responses from the ChatBot, but you can always respond with subsequent refinement requests to appropriately limit the scope of the ChatBot responses to focus on addressing your actual content targets; so, 
> - an initially very general inquiry like, "I need help analyzing my data" will likely result in a ChatBot response suggesting a wide variety of approaches and techniques for summarizing your dataset; but, re-prompting the ChatBot with something like, "What's the simplest form of summarization of this dataset that I could do and how do I do it in Python?" or suggesting guidance using the specific summarization methods requested above will helpfully re-orient the ChatBot to your specific interests and needs
> 
> **Jupyter Notebook Hints**
> 
> Jupyter notebook printouts usaully don't show all of the data (when there's too much to show, like if `df.describe()` includes results for many columns), but the printouts just show enough of the data to give an idea of what the results are which is all we're looking for at the moment
> 
> - Consider dividing the code that ChatBot provides you into different jupyter notebook cells so that each cell concludes with a key printed result; the last line of code in a jupyter notebook cell will automatically print out in a formatted manner, so replacing something like `print(df.head())` with `df.head()` as the last line of a cell provides a sensible way to organize your code
> - The printout suggestions above are demonstrated in `STA130F24_CourseProject.ipynb` if looking at an example would be helpful to understand what they're getting at...
    
</details>

#### 4. If the dataset you're using has (a) non-numeric variables and (b) missing values in numeric variables, explain (perhaps using help from a ChatBot if needed) the discrepancies between size of the dataset given by `df.shape` and what is reported by `df.describe()` with respect to (a) the number of columns it analyzes and (b) the values it reports in the "count" column<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> If the dataset you're using does not have (a) non-numeric variables and (b) missing values in numeric variables (e.g., the `"villagers.csv"` example above has only a single numeric variable `row_n` which has no missing values), instead download and use the [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv)" data to answer this question  
>
> In (a) above, the "columns it analyzes" refers to the columns of the output of `df.describe()` which will only include "numeric" columns by default, but you can can see the names of all the columns in a dataset using `df.columns`; and, make sure `df.shape` is refering to the dataset you think it is... if you've loaded a different dataset it might not have been called `df`(!)
>
> **If you get any errors (for example related to column names), copy and paste them as a response to the ChatBot, and see if it can help you resove them by adding the suggested adjustments to your code and then reruning all your code to see if the changes have fixed the problem (and repeat this process as needed until the problems have been resolved).**
    
</details>

#### 5. Use your ChatBot session to help understand the difference between the following and then provide your own paraphrasing summarization of that difference

- an "attribute", such as `df.shape` which does not end with `()`
- and a "method", such as `df.describe()` which does end with `()` 
   

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> The fact that a "method" such as `df.describe()` ends with `()` suggests that "methods" are essentially something that we would call a "function" in programming language terminology; but, without getting too technical or "in the weeds", it might also be worth considering that we could also contrast what the difference is between a "function" in a programming language versus a "function" in mathematics...  
    
</details><br><br>

***Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)!***<br><br>

<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Prelecture VS Postlecture HW
    
Feel free to work on the "Postlecture" HW below if you're making good progress and want to continue: in this case this is particularly reasonable as questions "6" and "7" below directly follow up and extend the "Prelecture" HW questions

*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details>    


### "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]

#### 6. The `df.describe()` method provides the 'count', 'mean', 'std', 'min', '25%', '50%', '75%', and 'max' summary statistics for each variable it analyzes. Give the definitions (perhaps using help from the ChatBot if needed) of each of these summary statistics<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> The answers here actually make it obvious why these can only be calculated for numeric variables in a dataset, which should help explain the answer to "4(a)" and "4(b)" above
>   
> Also notice that when `df.describe()` is used missing values are not explicitly removed, but `df.describe()`  provides answers anyway. Is it clear what `df.describe()` does with the data in each columns it analyzes if there is missing data in the column in question? 
>
> The next questions addresses removing rows or columns from a dataset in order to explicitly remove the presense of any missingness in the dataset (assuming we're not going to fill in any missing data values using any missing data imputation methods, which are beyond the scope of STA130); so, the behavior of `df.describe()` hints that explicitly removing missing may not always be necessary; but, the concern, though, is that not all methods may be able to handle missing data the way `df.describe()` does...
    
</details>

#### 7. Missing data can be considered "across rows" or "down columns".  Consider how `df.dropna()` or `del df['col']` should be applied to most efficiently use the available non-missing data in your dataset and briefly answer the following questions in your own words

1. Provide an example of a "use case" in which using `df.dropna()` might be peferred over using `del df['col']`<br><br>
    
2. Provide an example of "the opposite use case" in which using `del df['col']` might be preferred over using `df.dropna()` <br><br>
    
3. Discuss why applying `del df['col']` before `df.dropna()` when both are used together could be important<br><br>
    
4. Remove all missing data from one of the datasets you're considering using some combination of `del df['col']` and/or `df.dropna()` and give a justification for your approach, including a "before and after" report of the results of your approach for your dataset.<br><br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> Start a new ChatBot session **[but remember to first ask your ChatBot for summaries of your current session and perhaps coding results (so you can supply these in the homework as requested)]**, since your last ChatBot session has likely gotten quite long and has covered a lot of material at this point 
> - It can sometimes be helpful to reset ChatBot sessions to refocus them on the topics of inquiry without too much backlog history that might unintentionally bias things in certain directions and, of course, you can always re-introduce material from earlier conversations as it's relevant, such as for answering "D" based on reintroducing and updating code you made in a previous ChatBot session.  
> 
> #### ChatBot Scope Guidance
> 
> - This question is not interested in the general benefits of imputing missing data, or the general benefits of using `df.dropna()` and/or `del df['col']` to remove missing data, just how to most efficiently remove missing data if a user chooses to do so
> 
> - More sophisticated analyses for "filling in" rather than removing missing data (as considered here) are possible (based on making assumptions about missing data and using specific imputation methods or models) but these are "beyond the scope" of this homework assignment so this topics can be safely ignored for now
> 
> **ChatBot Code Troubleshooting**
> 
> A key issue to be aware of when asking ChatBots for help with something is that they are not running and checking code for correctess, and they often intertwine written instructions with code instructions; so, BEFORE YOU RUN ANY CODE provided by a ChatBot, you should check the following
> 
> 1. If this code changes an object or data, are you sure you want to run this code?
> 2. Can you easily "undo" the results of running code (e.g., from a copy `df_saved=df.copy()` or reloading the data) if running the code doesn't do what you want?
> 3. Is the state of the data what is expected by the code? Or have the objects been updated and changed so they're no longer what the code expects them to be? 
> 
> **If you get any `Python` errors, copy and paste them into the ChatBot prompt and see if it can help you resove them; but, keep in mind the final point above becasue the ChatBot might not be aware of the state of your objects relative to the code it's producing...**

</details><br>





    
#### 8. Give brief explanations in your own words for any requested answers to the questions below

> This problem will guide you through exploring how to use a ChatBot to troubleshoot code using the "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv" data set 
> 
> To initialially constrain the scope of the reponses from your ChatBot, start a new ChatBot session with the following slight variation on the initial prompting approach from "2" above
> - "I am going to do some initial simple summary analyses on the titanic data set I've downloaded (https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv) which has some missing values, and I'd like to get your help understanding the code I'm using and the analysis it's performing"
        
1. Use your ChatBot session to understand what `df.groupby("col1")["col2"].describe()` does and then demonstrate and explain this using a different example from the "titanic" data set other than what the ChatBot automatically provide for you
    
> If needed, you can help guide the ChatBot by showing it the code you've used to download the data **AND provide it with the names of the columns** using either a summary of the data with `df.describe()` or just `df.columns` as demonstrated [here](../CHATLOG/COP/00017_copilot_groupby.md)
    
2. Assuming you've not yet removed missing values in the manner of question "7" above, `df.describe()` would have different values in the `count` value for different data columns depending on the missingness present in the original data.  Why do these capture something fundamentally different from the values in the `count` that result from doing something like `df.groupby("col1")["col2"].describe()`?

> Questions "4" and "6" above address how missing values are handled by `df.describe()` (which is reflected in the `count` output of this method); but, `count` in conjunction with `group_by` has another primary function that's more important than addressing missing values (although missing data could still play a role here).

3. Intentionally introduce the following errors into your code and report your opinion as to whether it's easier to (a) work in a ChatBot session to fix the errors, or (b) use google to search for and fix errors: first share the errors you get in the ChatBot session and see if you can work with ChatBot to troubleshoot and fix the coding errors, and then see if you think a google search for the error provides the necessary toubleshooting help more quickly than ChatGPT<br><br>
    
    1. Forget to include `import pandas as pd` in your code 
       <br> 
       Use Kernel->Restart from the notebook menu to restart the jupyter notebook session unload imported libraries and start over so you can create this error
       <br><br>
       When python has an error, it sometimes provides a lot of "stack trace" output, but that's not usually very important for troubleshooting. For this problem for example, all you need to share with ChatGPT or search on google is `"NameError: name 'pd' is not defined"`<br><br>

    2. Mistype "titanic.csv" as "titanics.csv"
       <br> 
       If ChatBot troubleshooting is based on downloading the file, just replace the whole url with "titanics.csv" and try to troubleshoot the subsequent `FileNotFoundError: [Errno 2] No such file or directory: 'titanics.csv'` (assuming the file is indeed not present)
       <br><br>
       Explore introducing typos into a couple other parts of the url and note the slightly different errors this produces<br><br>
      
    3. Try to use a dataframe before it's been assigned into the variable
       <br> 
       You can simulate this by just misnaming the variable. For example, if you should write `df.groupby("col1")["col2"].describe()` based on how you loaded the data, then instead write `DF.groupby("col1")["col2"].describe()`
       <br><br>
       Make sure you've fixed your file name so that's not the error any more<br><br>
        
    4. Forget one of the parentheses somewhere the code
       <br>
       For example, if the code should be `pd.read_csv(url)` the change it to `pd.read_csv(url`<br><br>
        
    5. Mistype one of the names of the chained functions with the code 
       <br>
       For example, try something like `df.group_by("col1")["col2"].describe()` and `df.groupby("col1")["col2"].describle()`<br><br>
        
    6. Use a column name that's not in your data for the `groupby` and column selection 
       <br>
       For example, try capitalizing the columns for example replacing "sex" with "Sex" in `titanic_df.groupby("sex")["age"].describe()`, and then instead introducing the same error of "age"<br><br>
        
    7. Forget to put the column name as a string in quotes for the `groupby` and column selection, and see if the ChatBot and google are still as helpful as they were for the previous question
       <br>
       For example, something like `titanic_df.groupby(sex)["age"].describe()`, and then `titanic_df.groupby("sex")[age].describe()`
        


#### 9. Have you reviewed the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)
    
</details>
    
***Don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)!***

## Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course<br><br>

<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations

> If the observed data is "no events occured" does this mean the data is "missing" and [should be ignored](https://priceonomics.com/the-space-shuttle-challenger-explosion-and-the-o)?
> 
> - NASA: \<determines temperature doesn't affects "o-ring" by subseting data to just "o-ring" incidents\>
> - Also NASA: \<launches the shuttle on a cold day\>

|No apparent "o-ring" failure and temperature relationship|Apparent between "o-ring" failure and temperature relationship|
|:-|:-|
if you just look at "o-ring" failure event data|if you instead look at ALL the data as you should|
|![](https://etzq49yfnmd.exactdn.com/wp-content/uploads/2022/03/image06-14.png)|![](https://etzq49yfnmd.exactdn.com/wp-content/uploads/2022/03/image02-33.png)|
|![](https://upload.wikimedia.org/wikipedia/commons/8/8b/Shuttle_Challenger_explosion.gif?20190203170223)|![](https://i.makeagif.com/media/10-04-2014/nT57xW.gif)|

<br>
    
</details>    

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>

### Current Course Project Capability Level

> The data we'll use for the STA130 course project is based on the [Canadian Social Connection Survey](https://casch.org/cscs). Please see the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) regarding the appropriate and ethical professional use of this data (available at the bottom of the [CSCS](https://casch.org/cscs) webpage).
> 
> 1. Have a very quick look at the list of available variables using the [link](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view) (again at the bottom of the [CSCS](https://casch.org/cscs) webpage); then, 
> 2. examine the code in the first thirteen code cells of [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb) to get an initital understanding of how we might subset to different studies included in the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) (again accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage); then,     
> 3. review the fourteenth and fifteenth cells (with the comments "Here's a high level summary of the data" and "And here are some explanations about the columns in the data") a little more closely to get a better sense of which columns seem to be the most interesting and whether or not they seem to have a lot of missing data
    
</details>        

### Afterward

Here are few ideas of some other kinds of interactions you might consider exploring with a ChatBot...

> While these are likely to be extremely practically valuable, they are not a part of the homework assignment, so do not include anything related to these in your homework submission

- With respect to improving ones ability in statistics, coding, communication, and other key data science skills
    - what is the ChatBots perception its own capabilities and uses as an AI-driven assistance tool 
    - and does ChatBots assessment of itself influence or agree with your own evalution of the ChatBot? 

- ChatBots can introduce and explain the "World War 2 planes" problem and the "Monte Hall" problem... 
    - how well does do they seem to do and introducing and explaining other "unintuitive surprising statistics paradoxes"?

- If you consider the process of writing about why you chose to take this course, and the skills you were hoping to build through this course with respect to your current ideas about what possible careers 
    - and how do you think the exercise would be different if you framed it as a dialogue with a ChatBot
    - and do you think the difference could be positive and productive, or potentially biasing and distracting?
    
- ChatBots sometimes immediately responds in simple helpful ways, but other times it gives a lot of extraneous information that can be overwheling... are you able to prompt and interact with ChatBots in manner that keeps its reponses helpful and focused on what you're interested in? 

- ChatBots tends to respond in a fairly empathetic and supportive tone...
    - do you find it helpful to discuss concerns you might have about succeeding in the course (or entering university more generally) with a ChatBot?
    
- For what purposes and in what contexts do you think a ChatBot could provide suggestions or feedback about your experiences that might be useful? 

