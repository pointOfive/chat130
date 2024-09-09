# Data Summarization

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


## **Functions**, **arguments**, and **attributes**


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

## Chaining and coercion and<br>"data" types (as opposed to "Object" types to be discussed later)



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

## Summarizing data with `df.describe()` and *statistics* (as opposed to *Statistics*)

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


## **Summarizing data** with<br><br>`df['categorical_column'].value_counts()`


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

## Sorting, indexing, and subsetting<br>OR

### `.sort_values()`<br><br>`df[]` 0-based (row) and (column) name indexing (and "index" versus "row")<br><br>$\quad$versus fully 0-based indexing with `df.iloc[]` and "slicing" with `:`<br><br>$\quad\quad$versus `df[]` or `df.loc[]` **boolean selection** with **logical conditionals**<br><sub>$\quad\quad\;\;$ `>=` / `>` / `<=` / `<` /  and `==` (and `!=` in contrast to `=`) and `~` / `&` / `|` (not/and/or)</sub>



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
