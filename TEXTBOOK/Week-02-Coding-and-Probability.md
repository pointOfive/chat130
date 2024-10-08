
# Course Textbook: Week 02 Probability and Coding

# Chance is simple and Chat MAKES things simple

**Tutorial/Homework: Topics**

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
2. [What are _pandas DataFrame objects_?](week-02-Coding#what-are-pddataframe-objects)
3. [_for word_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding/week-02-Coding#More-for-Loops) _in_ [_sentence.split():_](week-02-Coding#more-list-behavior-with-str-and-split)

**Lecture: New Topics**

1. [_from scipy import stats_, _stats.multinomial_, and probability](week-02-Coding#scipystats) (and `np.random.choice`)
    1. [conditional probability Pr(A|B) and independence Pr(A|B)=Pr(A)](week-02-Coding#conditional-probability-and-independence)


## Tutorial/Homework: Topics

### Types

A `tuple` is an object containing an "sequential collection" of "things" that is created and represented with round brackets.
Tuples are **immutable**, which means that after a tuple is created you cannot change, add, or remove items; so, tuples's are ideal for representing data that shouldn’t really change, like birthday dates or geolocation coordinates of natural landmarks.

```python
example_tuple = (1, 'apple', 3.14, 1) # tuples can contain duplicate elements
example_tuple
```

A `list` is another kind of object containing a "sequential collection" of "things" that is created and represented with square brackets.
Unlike tuples, lists are **mutable**, which means they can be altered after their creation. If you don't want to recreate a tuple from scratch each time you need to change your collection of things, then you should use a list!

```python
example_list = [1, 'banana', 7.77, 1] # lists can also contain duplicate elements like tuples
example_list.append('new item') # here we add a new element onto the list
# to do the same thing with a tuple you'd have to completely create a completely new tuple
# as `example_tuple_update = (1, 'banana', 7.77, 1, 'new item')`
example_list
```

A `dict` ("dictionary") is an object that uses a "key-value" pairs "look up structure" of "things" instead of a "sequential collection" organization that are created and represented using "key-value" pairs inside of curly braces (as demonstrated below).
Dictionaries are **mutable** like lists, but each "key" of a dictionary is **unique** (so it uniquely references its corresponding "value"). Since a `dict` is based on a "look up" system, it is a so-called **unordered** object. This means that unlike tuples and lists which always remember their sequential order, dictionaries do not maintain their order of insertion and can change the order of items when the dictionary is modified.

```python
example_dict = {'id': 1, 'name': 'orange', 'price': 5.99} # There cannot be duplicate "keys" but there could be duplicate "values"
example_dict['quantity'] = 10 # adds a new "key-value" pair
del example_dict['quantity'] # removes a "key-value" pair
```

The use of **dictionaries** to rename the columns of `pandas DataFrame objects` was previously seen in the [Variables and Observations](week-01-Data-Summarization#Variables-and-Observations) section of Week 01 of the course wiki-textbook; and an example of a more elaborate **dictionary** object and its extension (again related to the `pandas DataFrame objects` context) is given in the 
"[What are _pd.DataFrame objects_?](week-02-Coding#what-are-pddataframe-objects)" section below.


### _np.arary_

NumPy is a `Python` library that contains the most efficient versions of standard numerical routines.
For example, a NumPy `np.array` is preferred over a list for its speed and functionality in numerical tasks.
The NumPy library is imported and a `np.array` is created from a list object as follows.

```python
import numpy as np
example_array = np.array([1, 2, 3])
```

An example numerical task that can be done with NumPy is to select a random value from an `np.array` object as follows.

```python
random_element = np.random.choice(example_array)
random_element
```

### _for_ loops

The `range(n)` function in `Python` **generates** numbers from `0` to `n-1`.
> If you try to run the `range(n)` function to produce these values, it won't do anything because it is a so-called **generator** which means it will only produce the actual values within the context of a looping structure which sequentially requests the values.  This is actually clever because it means the actual numbers themselves don't actually have to be stored in memory anywhere, and can instead just be sequentially produced one at a time as needed. 

The `print()` function outputs a displays of its object argument. Since (as discussed above) `range(5)` is a so-called **generator**, if you run the code `print(range(5))`, you will get the following output. 

```python
range(0, 5) # output from running `print(range(5))`
```

The `for i in range(n):` template is the coding construct that is used used to specify the repetition of a block of code `n` times.

> The block of code that the `for` loop repeats will be executed "silently"; so, if you want to display anything inside of a `for` loop you need to explicitly use the `print()` function in the body of your `for` loop as demonstrated below.

```python
for i in range(5): # "iterates" `i` over the values 0, 1, 2, 3, 4
    # the "body" of a `for` loop --
    # the "indented code block" below the `for` statement syntax
    print(i) # is executed sequentially for each value of the `i` "iterator"
```

> `Python` code WILL NOT WORK unless properly indented... this is an interesting "feature" of `Python` coding that helps to make code for readable!

Here's a step by step break down of what the `for` loop code above is doing.

1. Initialization: the `for` loop starts with the keyword `for`
2. Iterator Variable: `i` is the "iterator" variable that will sequentially change with each "iteration" of the `for` loop
3. The `range()` function: `range(5)` will "iteratively" generate the sequence of numbers from `0` to `4` (since `Python` uses "0-indexing"), and these will be sequentially assigned to "iterator" `u` 
4. Loop Body: the code block indented under the `for` loop defines what happens during each "iteration" (in this case, the sequentially assigned values of `i` will be printed)
5. Iteration Process:
    1. In the first iteration, `i` is set to `0`, the first number produced by the `range` generator.
    2. The `print(i)` statement is executed, printing `0` to the screen.
    3. The `for` loop now iterates by setting `i` is set to the next number produced by the `range` generator, which is `1`.
    4. `print(i)` is executed again, this time printing `1`.
    5. This process repeats until `i` has "iterated" through all of the values produced by the `range` generator.
6. Termination: once `i` has reached `4` there are no more `i` "iterations", the loop ends, and the program continues with any code following the loop.

### More _for_ Loops 

It is sometimes useful to iterate through a custom list rather than a `range(n)` generator. 
Below, instead of "iterator" `i`, we denote the "iterator" as `x` to emphasize that it's not a "numerical index iterator".
> This is not strictly necessary, since you can name your "iterator" variable whatever you want to and then access it as such in the body of the for loop.

```python
a_list = ['apple', 'banana', 'cherry'] # or, equivalently: `a_list = "apple banana cherry".split()`
for x in a_list: # note that we don't have to use the `range()` function here!
    print(x)     # we can just "iterate" through the "iterable" list `a_list`!
```

It is additionally sometimes useful to both iterate through a custom list but also still have a "numerical index iterator" as well.
This is done with by wrapping the `enumerate()` function around a list (or tuple) object.

```python
for i,x in enumerate(a_list):
    print(f"Index: {i}") # this useful syntax pastes `i` into the displayed string
    print(f"Value: {x}") # this useful syntax pastes `x` into the displayed string
    print("Iteration Completed")
```

The `enumerate(a_list)` "numerical index iterator" (`i`) to the "iterable" list (`a_list`) and returns it as an enumerate object which the `for` loop understands and unpacks into `i` and `x` at each "iteration" as indicated by the `i,x` syntax. 

One more `for` loop structure that can sometimes be useful is "iterating" through dictionaries based on the `.items()`, `.keys()`, or `.values()` methods of a dictionary.

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
for key in my_dict.keys():
    print(key)

for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")

for value in my_dict.values():
    print(value)
```

### Logical Flow Control

FizzBuzz is a classic programming challenge that’s often used to teach basic programming concepts like loops and conditionals. 
In the FizzBuzz problem, you loop through a range of numbers, and then do the follow for each number.
1. If the number is divisible by 3, you print “Fizz”.
2. If the number is divisible by 5, you print “Buzz”.
3. If the number is divisible by both 3 and 5, you print “FizzBuzz”.
4. If the number is not divisible by 3 or 5, you print the number itself.

Here’s an example FizzBuzz program that illustrates the use of `if` and `else` conditional statements for logical flow control with comments explaining each step.

```python
for i in range(1, 101):  # Loop from 1 to 100

    if i % 3 == 0 and i % 5 == 0:  # Check if divisible by both 3 and 5
        print("FizzBuzz")
    elif i % 3 == 0:  # Check if divisible by 3
        print("Fizz")
    elif i % 5 == 0:  # Check if divisible by 5
        print("Buzz")
    else:  # If not divisible by 3 or 5
        print(i)
```

1. The `for` loop sets up the iteration (from `1` to `100` in this example).
2. The first `if` statement checks for the first condition (divisible by both 3 and 5).
    1. the **modulus** operation `%` returns the remainder of "i divided by 3"; so, it's just an operation like `i+3`, `i-3`, or `i/3`; but, if the remainder is `0` then it means that "i divides by 3 perfectly"
    2. The `and` construction in `i % 3 == 0 and i % 5 == 0` means that both `i % 3 == 0` AND `i % 5 == 0` must be `True` for the `if` statement to succeed
    3. If the `if` statement succeeds (because its "argument is `True`) then the "body" (the indented code block beneath the `if` statement) of the `if` statement is executed
3. The next two `elif` ("else if") statements each subsequently sequentially check if their respective conditions (divisible by 3, and then 5) are true statements, and execute their code block "bodies" if so
    1. Using `elif` instead of `if` "connects" the logical statements into a single logical control flow sequence whose conditions can be understood in an "else if" manner that can help improve the clarity of the checks. 
4. The `else` statement covers the case where none of the above conditions are `True` and concludes the logical control flow sequence.
5. The `print()` function outputs the result based on the condition that’s met.

This structure allows the program to make decisions at each iteration, using logical flow control structures within a `for` loop to print out a mix of numbers and the words “Fizz”, “Buzz”, and “FizzBuzz” based on the divisibility of each number. 

**`try-except` blocks**

A similar logical flow control structure to `if`/`else` statements is the `try-except` block. Rather than checking for `True` conditions however, `try-except` blocks check for the presence of "run time errors" which are stored as a kind of `Python` object known as an `Exception`. In the code below, `Python` tries to run "one"/"two" but doesn't know how to do this; but, the `except Exception as e` construct allows the nature of the error to be captured as an `Exception` object and named `e` which can then be printed out and examined without cause the code to fail (as it would if you tried to run `"one"/"two"` without wrapping a `try-except` block around it). 

```python
try:
    "one"/"two"
except Exception as e:
    print(f"An error occurred: {e}")
```


## Tutorial/Homework: Lecture Extensions

### More types

While we will consider **types of data** (like **numerical** or **categorical**) to inform decisions about what kind of analysis are going to be the most appropriate for a given dataset (as we saw when discussing [_.describe()_](week-01-Data-Summarization#types-i) and [_.value_counts()_](week-01-Data-Summarization#types-i)), this is somewhat different than considering the **type** of an **object** in Python. To determine the **type** of an **object** in Python we us the `type()` function, which returns the specific data structure (called the **class**) of the **object** in the Python programming language. 

For example:
```python
x = 10
print(type(x))  # <class 'int'>

y = "Hello"
print(type(y))  # <class 'str'>

z = [1, 2, 3]
print(type(z))  # <class 'list'>

import numpy as np
my_array = np.array([1, 2, 3, 4])
print(type(my_array))  # Output: <class 'numpy.ndarray'>

a = True
print(type(a))  # <class 'bool'>

b = (5 > 3)  # This expression evaluates to True
print(type(b))  # <class 'bool'>

x2 = 3.14
print(type(x2))  # <class 'float'>

y = float(10)  # Converting an integer to a float
print(type(y))  # <class 'float'>
```

Here, `type()` returns the **class** of the **object**, which tells you what kind of data structure it is: `int`, `str`, etc. (and we've previously introduced [_tuple_](week-02-Coding#Types-II), [_list_](week-02-Coding#Types-II), [_dict_](week-02-Coding#Types-II) **object types**). In the `bool` examples, `True` and the result of a comparison `5 > 3` are both of **type** `bool`. In the `float` examples, `3.14` is a **numeric** (floating-point) decimal number, and converting `10` to a `float` gives a `float` type as well. And the final line of code explicitly converts (as opposed to implicitly **coercing**) an `int` type into a `float`. 

Something to consider would be what might happen the other way, if we tried to convert a `float` (such as `x2` above) to an `int` (as in `int(x2)`).  More generally, what different **object types** types might naturally convert to other **object types**? If you recall how [**coercion**](week-01-Data-Summarization#Boolean-Values-and-Coercion) automatically converted `bool` types to `int` types (because there was a well-known rule that made it obvious how this would be done), this might be a good example to start thinking about what possibilities make sense (or don't make sense).  

### More indexing

Different **object types** (of course) serve different purposes. It may be intuitive to imagine potential uses a `list` **object**; but, it's interesting to note that we **index** (and "negative" **index**) into a `list` (just as we **iterate** through `for` loops) using an `int` and can rely on `bool` **objects** when constructing **logical conditionals** for **boolean selection* as demonstrated below.

```python
print(z[0])  # Output: 1
print(z[-1])  # Output: 3
print(z[1])  # Output: 2
print(z[-2])  # Output: 2

my_array[my_array<3]  # Output: array([1, 2])
```

The indexing shown above is actually just analogous to the [row-based indexing](week-01-Data-Summarization#sorting-and-iloc-indexing) and [boolean selection indexing](week-01-Data-Summarization#logical-conditionals-boolean-selectionsubsetting-and-loc-indexing-v2) using **logical conditionals** that we previously. Indeed, you will be able to **slice** index into `list` and `np.array` **objects** just as with `pandas` **DataFrames**. Ask a ChatBot for examples of how to select elements from `list` and `np.array` **objects** if you are curious to see some of the options available for doing this. 

### More _np.array_

While Python's built-in **object types** like `list` and `str` (as discussed below) are powerful, sometimes you need more specialized tools for handling **numeric** data. The `np.array()` function allows you to create "arrays" specifically designed for efficient computation of mathematical operations with large amounts of **numeric** data compared to a Python `list` (of **numeric** values). In addition to its extensive functionality, the computational speed benefits of `numpy` are a big reason why `numpy` is (indeed) a popular library for numerical computing. The reason `np.array` **objects** can be offer faster computational performance is because the **object type** of the items in an `np.array` **objects** must all be identical. 

In the code above, `my_array` is an `np.array` **object**, and every element of  The reason `my_array` is of **type** `int` (or, technically, an `int64` which explicitly indicates that `numpy` is using a 64-bit integer **object type**). This can be seen using the `dtype` **attribute** of the `np.array` **object**. Notice the similarity, but distinction between `type()` and `.dtype` (with the latter being the way to see the homogenous **object type** of an `np.array` **object**). 

```python
print(type(my_array))  # Output: <class 'numpy.ndarray'>
print(my_array.dtype)  # Output: int64 (or another integer type depending on your system)
```

Also notice that `.dtype` **attribute** of an `np.array` **object** serves exactly the same purpose as the analogous `.dtypes` **attribute** of a `pd.DataFrame` **object** [introduced previously](week-01-Data-Summarization#pandas-column-data-types); and similarly; that the `type()` (and changing the **object type** demonstrated [above](week-02-Coding#more-types)) of an **object** should remind you of the `.astype()` **method** introduced alongside `.dtypes` for setting the **type** of data with a column of a `pd.DataFame` **object**.

> The `numpy` model and `np.arrays` are a key part of working with **numeric** data in Python, offering more advanced capabilities for mathematical operations and data manipulation than basic Python lists. So keep an eye on the the `numpy` library. It will likely be something that you'll likely come across in different context relatively frequently in the future. 


### More _list_ behavior with _str_ and _.split()_

In Python, a **string** (`str` **object type**) is a sequence of **characters** enclosed within (either single or double) quotes. **Strings** are one of the most commonly used **object types** for representing text (obviously), from single words to entire sentences or paragraphs. For example, `sentence` below is a **string object**. 

```python
sentence = "Learning Python is fun!"
```

You can treat strings as sequences of individual characters, and each character has an index (starting from 0), just as if it was a `list` or an `np.array`! For instance, to access the first, sixth, eleventh, and ninth, third to last, and last characters of the string we would use the following.

```python
first_char = sentence[0]
print(first_char)  # Output: 'L'

first_char = sentence[5]
print(first_char)  # Output: 'i'

first_char = sentence[10]
print(first_char)  # Output: 't'

first_char = sentence[8]
print(first_char)  # Output: ' '

first_char = sentence[-3]
print(first_char)  # Output: 'u'

first_char = sentence[-1]
print(first_char)  # Output: '!'

```

**Strings**, however, are **immutable**, which means that once created, **strings** cannot change individual characters within the string. You may recall that this **immutable** behavior is what distinguishes a `tuple` (which like a **string** is **immutable**) compared to a `list` (which unlike a **string** is **mutable**). So, trying to modify a specific letter, like `sentence[8] = '-'` will result in an error. Nonetheless, **strings** can still be manipulated in various ways to simply create new strings as needed. So, to "change" a **string** means to just create a new different **string**, rather than editing the original **string**. 

One of the most useful operations on **strings** is to break them apart into a **list**. As you know, a **list** in Python is an ordered collection of items, each of which can be of any **object type**. As you might expect, if you break a **string** apart into a **list**, the resulting elements of the **list** will be the (sub) **strings** (or **characters**) of the original **string**. But since a **list** is **mutable**, the "items" of the "string" in the converted **list** form could then be changed. Then (as should be reasonably expected) the modified **list** could be converted back into a **string**, producing an "edited" version of the original **string**.

To convert a **string** into a **list** of words, you can use the `split()` method of the **string object**. By default, `split()` divides the **string** wherever there is a space, creating a list of words. **Strings** also have a `join()` method, the most useful construction of which is for the `" "` blank space string, which as demonstrated below can below can be used to reconstruct a **string** from a **list**.

```python
words = sentence.split()
print(words)  # Output: ['Learning', 'Python', 'is', 'fun!']

words[3] = 'not'

words.append('too')  # the append method of a list modifies a list
words.append('bad')  # by adding an item to the end of the list
words += ['once', 'you', 'get', 'the', 'hang', 'of', 'the', 'process']  # '+' operator overloading

" ".join(words)
```

As you can see above, converting a **string** to a **list** allows us to work with the words of a sentence individually, and leverage **list** all the functionality of **list** operations. For example, we could now iterate over the words with a `for` loop, count the number of words with `len(words)`, or access and modify individual words or expense the **list** "sentence" (as we've done above), etc. 


### Operator Overloading

In Python, the same **operator** can behave differently depending on the **object types** being operated on. This is called **operator overloading** and is a specific case of a broader concept known as **polymorphism** (meaning "many behaviors"). **Polymorphism** in the form of **operator overloading** allows different **object types** to respond to the same **operation** in ways that are most appropriate to their **object type**.

Let's take the `+` **operator** as an example. In Python, `+` behaves differently depending on whether it’s applied to numbers (like `float` or `int`), **strings**, or `list` **object types**. For instance, `+` **concatenates** (or joins) **strings** together. This is of course an example of **operator overloading** because `+` is being used in a way that is specific to **strings**. 

```python
greeting = "Hello, " + "world!"
print(greeting)  # Output: Hello, world!
```

Similarly, when `+` is applied to two `list` **object types**, it **concatenates** (or joins or combines) the two into one, extending the original `list` with the elements of the second one. You may recognize that you've already seen this behavior above, but here it is again in a slightly different manner. We used `+=` in the version of this example above, which is just shorthand for combining and assigning in one step. 

```python
words = ['Learning', 'Python', 'is', 'not', 'too', bad']
new_words = ['once', 'you', 'get', 'the', 'hang', 'of', 'the', 'process']
words = words + new_words  # words += ['once', 'you', 'get', 'the', 'hang', 'of', 'the', 'process']
print(words)
# Output: ['Learning', 'Python', 'is', 'not', 'too', 'bad', 'once', 'you', 'get', 'the', 'hang', 'of', 'the', 'process']
```

It actually makes a lot of sense that both **string** and `list` **object types** behave in a similar way regarding **concatenation**. Especially when you remember that you can **index** into a **string** using `int` just as you can with a `list` **object types**. 

To drive the **concatenation** behavior of **operator overloading** when it comes to `+` and **strings**, recall how the construction `df.isna().sum(axis=1)` counts the number of missing values across rows by **coercing** `True` and `False` to `1` and `0` and summing them. What, then, would `df[["string_column_1","string_column_1"]].sum(axis=1)` do?
 

### What are _pd.DataFrame objects_?
  
Week 02 formally introduced `list`, `dict`, `np.array` and `str` "object" `types` (as opposed to "data" `types`); but, you actually encountered `str`, `list`, and `dict` (**dictionary**) `python object types` in Week 01 (perhaps without particularly noticing) in the [Missingness I](week-01-Data-Summarization#Missingness-I), [boolean values and coercion](week-01-Data-Summarization#Boolean-Values-and-Coercion), and [`Pandas` column data `types`](week-01-Data-Summarization#pandas-column-data-types) sections of the course wiki-textbook where they were used to defined `pandas DataFrame objects`.

```python
# Python `dict` types can be defined with curly brackets "{" and "}"
data = {
    'age': [25, 32, 47, 51], # 'age' is an `str` "string" type; `[25, 32, 47, 51]` is a `list`
    'name': ['Alice', 'Bob', 'Charlie', 'David'], 
    'income': [50000, 60000, 70000, 80000],
    'has_pet': ['yes', 'no', 'no', 'yes']
}
df = pd.DataFrame(data)
```

So a `pandas DataFrame object` is fundamentally a **dictionary**, with column names corresponding to the "keys" of the **dictionary** and the values in the rows of the column corresponding to the "values" in the **dictionary** which are **lists** of data (all having the same length).

> Technically, `pandas` first transforms each `list` (or `tuple`) into an `np.array` and then further transforms this into a `pd.Series`, finally making the `pandas DataFrame object` a collection of columns of  `pd.Series` objects which are accessed in the manner of a dictionary.

The fundamental **dictionary** nature of a `pandas DataFrame object` is reflected in the way columns are referenced when working in `pandas`, as seen in the [Types I](week-01-Data-Summarization#Types-I) and [Missingness II](week-01-Data-Summarization#Missingness-II) sections in Week 01:

```python
df['age']  # returns the 'age' column
del df['name']  # removes the 'name' column from the df object
# both of which function analogously to how `dict` objects are managed

# And, unsurprisingly, data is added to a `pd.DataFrame` object 
# in just the same analogous manner as for a `dict` object
df['city'] = ['New York', 'Los Angeles', 'Chicago', 'Houston']
# just like how the data would be added to the original dictionary object
data['city'] = ['New York', 'Los Angeles', 'Chicago', 'Houston']
```


## Lecture: New Topics


### _scipy.stats_

Probability is the mathematical framework that allows us to model chance (or uncertainty). In many real-world situations, we deal with events or outcomes that are not certain, and probability helps us quantify the likelihood of these events. Python, with libraries like `numpy` and `scipy`, provides powerful tools for handling probability distributions, statistical methods, and random events. The `stats`  module within the `scipy` library (i.e., `scipy.stats`) provides a wide range of statistical functions and probability distributions (such as the normal distribution, binomial distribution, and many others, some of which we will introduce later). These tools allow us to model different types of random events and calculate relevant probabilities of interest. To get started, we’ll import the `stats` submodule from `scipy` as follows, but you may sometimes see this functionality imported with alternative aliasing, such as `import scipy.stats as ss`, etc.

```python
from scipy import stats
```

Now that we have `stats`, let's consider our first **probability distributions**. We'll start with the **multinomial distribution**. This models the probabilities of selecting `n` things from `k` options (potentially choosing each option more than once if `n` is greater than `k`). The simplest version of this would be if `n=1`, then we'd just be choosing one of the `k` options. An example of a **multinomial distribution** would be rolling a six-sided die. If we just roll once, `n=1` and `k=6` and we'll see the face up side of the die (which will be one of the outcomes 1 through 6 if we're talking about a normal die).  If you roll the die multiple times, or roll multiple identical dice (like in Yahtzee where you start by rolling 5 dice), then `n` changes but `k` does not. So in Yahtzee where you roll five dice, `n=5` and `k=6`. 

So far you've probably been imagining a "fair die" or "fair dice", meaning that the chance of each of the `k` outcomes (or sides of a die in our ongoing example) is equally likely. But the **multinomial distribution** allows for some flexibility here. It has another aspect that we've not yet considered which is the "chance" of each of the `k` outcomes, and we usually refer to this as `p`.  The `p` needs to be a "list" of `k` probabilities which sum to one (which makes intuitive sense if you think about it a bit). So, in our die example, `p` will be six fractions (or decimal numbers) between 0 and 1 which together sum to 1. Here's how you use `scipy.stats` to model the two examples we've considered so far, followed by one more examples where we're rolling a die that is not "fair".

```python
from scipy import stats

# Suppose we're rolling a single die and the probabilities of each face are equal (1/6).
one_fair_die = stats.multinomial(n=1, p = [1/6] * 6)  # ready to roll
# `[1/6] * 6` above is another interesting example of *operator overloading*
# `[1/6] * 6` turns out to produce `[1/6,1/6,1/6,1/6,1/6,1/6]`... if you can't guess why, ask a ChatBot!

# notice that `k` is implied by the length of `p`

one_fair_die.rvs(size=1)  # rvs stands for "random variable sample"
# `size` is the number of times to do a "random variable sample" of whatever the `one_fair_die` thing is 
# so `size=1` here then means "role one die" just one time (since `n=1`)

one_fair_die.rvs(size=5)  # but here it means "role five dice"

# Consider the difference in the output between the following
# stats.multinomial(n=1, p = [1/6] * 6).rvs(size=5)
# stats.multinomial(n=5, p = [1/6] * 6).rvs(size=1)
# and see if you can articulate what exactly the similarity and difference is between these two things
# and what then the following is
# stats.multinomial(n=5, p = [1/6] * 6).rvs(size=5)

one_UNfair_die = stats.multinomial(n=1, p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])  # ready to roll
# We'll have to make sure were know which die face outcome corresponds to each probability...

one_UNfair_die.rvs(size=5)  # roll the unfair die five times
# Or is it role five "identically unfair dice"? Well... it's the same thing!
```

If you've understood above the sort of strange interchangeable similarity between `n` and `size` above, well done!
While they may seem redundant, they let us specify things like `stats.multinomial(n=5, p = [1/6] * 6).rvs(size=10)` which can be interpreted as "role 5 dice ten times" (as in a standard Yahtzee game).  This shows us that we can conceptualize the event of "picking `n` things from `k` choices" as something that can be hypothetically repeated over and over. That said, there's actually another, perhaps simpler and clearer way to create random samples from a **multinomial distribution** in Python. Consider the output of the code below and see if makes sense to you. Then compare the nature of the output below to the nature of the output of the code above.  Are you able to figure out how the output below is related to output from `stats.multinomial(n,p).rvs(size)` for different choices of `n` and `size`? if you can't quite tell, ask a ChatBot!

```python
import numpy as np
# Roll a six-sided die 10 times
rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=10, p=[1/6] * 6)
print(rolls)
```

### Conditional Probability and Independence

The last things we want to consider here are the notions of **conditional probability** and **independence**. Let's start with **conditional probability**, which takes the notational form $\Pr(A|B)$. Here's a question: is there such a thing as a "hot streak" when rolling dice? Say you're trying to roll "sixes" on a die, and you've rolled three in a row already(!), do you think you're more likely than usual, or less likely than usual to get another "six" on your next roll? Let's ask this question in notation of **conditional probability**. Are these two equal? 

$\Pr(\textrm{rolling a six}) = Pr(\textrm{rolling a six} | \textrm{the last three roll were a six})?$

That is, does the next die roll depend on the previous die rolls, or is it **independent** of them? What we're asking here is if there's a relevant **conditional probability** or if the events being considered **independent** (so there's really just a single **probability** and the idea of a **conditional probability** is not really necessary). So there is either an **independence** between two events, or (sort of contrarily "opposite") there will be a meaningful **conditional probability** that changes the probability of the events based on the outcomes of the other. 

If you think there's no such thing as a "hot streak" and your next roll does not depend on your last roll then you're saying the equality above is true, which means you're saying the next die roll is **independent** of the previous dice rolls and there's really no notion of a **conditional probability** (because it's just a **probability**). This is true, so long as you're really rolling a "fair die" randomly. So when a **conditional probability** statement can simplify, like if $\Pr(A|B) = \Pr(A)$ meaning that knowing $B$ does not change the probability of $A$ occurring, then this is when we say that $A$ and $B$ are **independent**.  In the **multinomial distribution**, the `n` selections of the `k` different possible options is assumed to be **independent**. So, the chances that we'll choose each of the `k` different possible options could be different (depending on their relative probabilities given by `p`), but each time we choose an outcome (one of the `n` selections we make), this does depend on which options we've previously chosen (if we're imagining choosing our `n` selections sequentially). 

This doesn't mean that we couldn't sequentially change our value of `p` in some sort of sequentially dynamic process that uses different **multinomial distributions** over time. But, it does mean that for  `n` selections from `k` options drawn from a **multinomial distributions** with a fixed unchanging `p`, the `n` selections are **independent** and do not change in response to each other or affect each other in any way. And it's actually also interesting to consider again here the `stats.multinomial(n,p).rvs(size)` specification.  The **independence** of the **multinomial distribution** means that the `n` choices for the `k` options related to `stats.multinomial(n,p)` do not depend on each other.  But, owing to the definition of "random variable sample", the `.rvs(size)` notion of repeating a the "`n` choices for the `k` options" game `size` times is itself also based on **independence**.  This means that the outcomes of different repetitions of the "`n` choices for the `k` options" games also do not affect each other.  

But there might be other examples where this is not true? Can you think of any? How about an example of drawing cards from a deck? Does the probability of drawing an Ace change if you've previously drawn and removed cards from the deck? 
