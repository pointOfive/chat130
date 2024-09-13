
# Course Textbook: Week 02 Probability and Coding
 
**TUT/HW Topics**

1. python object types... [_tuple_](week-02-Coding#Types), [_list_](week-02-Coding#types), [_dict_](week-02-Coding#Types)
2. another key data type... [_np.array_](week-02-Coding#np-array) (and `np.random.choice`)
3. for loops... [_for i in range(n):_](week-02-Coding#for-loops)
    1. [_print()_](week-02-Coding#for-loops)
    2. [_for x in some_list:_](week-02-Coding#More-for-Loops)
    3. [_for i,x in enumerate(some_list):_](week-02-Coding#More-for-Loops)
    4. ~`for key,val in dictionary.items()` and `dictionary.keys()` and `dictionary.values()`~
4. logical flow control... [_if_](week-02-Coding#Logical-Flow-Control), [_elif_](week-02-Coding#Logical-Flow-Control), [_else_](week-02-Coding#Logical-Flow-Control)
    1. ~[_try-except_ blocks](week-02-Coding#Logical-Flow-Control)~

**LEC Extensions**

1. more object types... [_type()_](week-02-Coding#more-types) 
    1. [more indexing for "lists"](week-02-Coding#more-indexing)
    2. [more _np.array_ with _.dtype_](week-02-Coding#more-nparray) 
    3. [more "list" behavior with _str_ and _.split()_](week-02-Coding#more-list-behavior-with-str-and-split)
        1. ~text manipulation with `.apply(lambda x: ...)`, `.replace()`, and `re`~
    4. [operator overloading](week-02-Coding#operator-overloading)
2. [What are _pandas DataFrame objects_?](week-02-Coding#what-are-pddataframe-objects)
3. [_for word_](https://github.com/pointOfive/stat130chat130/wiki/Week-02-Coding/week-02-Coding#More-for-Loops) _in_ [_sentence.split():_](week-02-Coding#more-list-behavior-with-str-and-split)

**LEC New Topics**

1. [_from scipy import stats_, _stats.multinomial_, and probability](week-02-Coding#scipystats) (and `np.random.choice`)
    1. [conditional probability Pr(A|B) and independence Pr(A|B)=Pr(A)](week-02-Coding#conditional-probability-and-independence)


## TUT/HW Topics

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


## LEC Extensions / New Topics

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

### _scipy.stats_

Probability is the mathematical framework that allows us to model chance (or uncertainty). In many real-world situations, we deal with events or outcomes that are not certain, and probability helps us quantify the likelihood of these events. Python, with libraries like `numpy` and `scipy`, provides powerful tools for handling probability distributions, statistical methods, and random events. The `stats`  module within the `scipy` library (i.e., `scipy.stats`) provides a wide range of statistical functions and probability distributions (such as the normal distribution, binomial distribution, and many others, some of which we will introduce later). These tools allow us to model different types of random events and calculate related probabilities of interest. To get started, we’ll import the `stats` submodule from `scipy` as follows, but you may sometimes see this functionality imported with alternative aliasing, such as `import scipy.stats as ss`, etc.

```python
from scipy import stats
```

Now that we have `stats`, let's consider our first **probability distributions**. We'll start with the **multinomial distribution**. This models the probabilities of selecting `N` things from `n` options (potentially choosing each option more than once if `N` is greater than `n`). The simplest version of this would be if `N=1`, then we'd just be choosing one of the `n` options. An example of a **multinomial distribution** would be rolling a six-sided die. If we just roll once, `N=1` and `n=6` and we'll see the face up side of the die (which will be one of the outcomes 1 through 6 if we're talking about a normal die).  If you roll the die multiple times, or roll multiple identical dice (like in Yahtzee where you start by rolling 5 dice), then `N` changes but `n` does not. So in Yahtzee where you roll five dice, `N=5` and `n=6`. 

> Since we're using very similar notation for the two key ideas here, `N` and `n`, you'll need to think carefully and clearly about what the different `N` and `n` are actually referring to. But that's why we're naming them so similarly. We want to make sure that you pause to think carefully about the distinction between the two ideas. Hopefully right now the difference is pretty clear. But later in more general circumstances stances, things might not be so obvious. 

So far you've probably been imagining a "fair die" or "fair dice", meaning that the chance of each of the `k` outcomes (or sides of a die in our ongoing example) is equally likely. But the **multinomial distribution** allows for some flexibility here. It has another aspect that we've not yet considered which is the "chance" of each of the `k` outcomes, and we usually refer to this as `p`.  The `p` needs to be a "list" of `k` probabilities which sum to one (which makes intuitive sense if you think about it a bit). So, in our die example, `p` will be six fractions (or decimal numbers) between 0 and 1 which together sum to 1. Here's how you use `scipy.stats` to model the two examples we've considered so far, followed by one more examples where we're rolling a die that not "fair".

```python
from scipy import stats

# Suppose we're rolling a single die and the probabilities of each face are equal (1/6).
one_fair_die = stats.multinomial(n=1, p = [1/6] * 6)  # ready to roll
# `[1/6] * 6` above is another interesting example of *operator overloading*
# `[1/6] * 6` turns out to produce `[1/6,1/6,1/6,1/6,1/6,1/6]`... if you can't guess why, ask a ChatBot!

N = 1
one_fair_die.rvs(N)  # rvs stand for "random variable sample"
# which here just means, "role one die" since `N=1`

N = 5
one_fair_die.rvs(N)  # but here it means "role five dice"

one_UNfair_die = stats.multinomial(n=1, p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25] )  # ready to roll
# We'll have to make sure were know which die face outcome corresponds to each probability...

one_UNfair_die.rvs(N)  # roll the unfair die five times
# Or is it role five "identically unfair dice"? Well... it's the same thing!
```

There's actually another, more clear way to do this with code that we should consider using for now. Consider the output of the code below and see if makes sense to you. Then compare the nature of the output below to the nature of the output of the code above.  Are you able to figure out how the random is output from the `stats.multinomial(...).rvs(N)` code is formatted? if you can't guess why, ask a ChatBot!

```python
import numpy as np
# Roll a six-sided die 10 times
rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=10, p=[1/6] * 6)
print(rolls)
```

### Conditional Probability and Independence

The last things we want to consider here are the notions of **conditional probability** and **independence**. Let's start with **conditional probability**, which takes the notational form $\Pr(A|B)$. Here's a question: is there such a thing as a "hot streak" when rolling dice? Say you're trying to roll "sixes" on a die, and you've rolled three in a row already(!), do you think you're more likely than usual, or less likely than usual to get another "six" on your next roll? Let's ask this question in notation of **conditional probability**. Are these two equal? 

$\Pr(\textrm{rolling a six}) = Pr(\textrm{rolling a six} | \textrm{the last three roll were a six})?$

That is, does the next die roll depend on the previous die rolls, or is it **independent** of them? What we're asking here is if there a relevant **conditional probability** or if the events being considered **independent** so there really just a single **probability** and the idea of a **conditional probability** is not really necessary. This then gives the meaning of **independence**, and of course this is defined relatively (and sort of contrarily "opposite") to  **conditional probability**.

If you think there's no such thing as a "hot streak" and your next roll does not depend on your last roll then you're saying this equality is true, which means you're saying the next die roll is **independent** of the previous dice rolls and there's really no notion of a **conditional probability** because it's just a **probability**. This is true, so long as you're really rolling a "fair die" randomly. So when a **conditional probability** statement can simplify, like if $\Pr(A|B) = \Pr(A)$ meaning that knowing $B$ does not change the probability of $A$ occurring, then this is when we say that $A$ and $B$ are **independent**.  In the **multinomial distribution**, the `N` selections of the `n` different possible options is assumed to be **independent**. So, the chances that we'll choose the `n` different possible options could be different (depending on their relative probabilities given by `p`), but each time we choose an outcome (one of the `N` selections we make), this does depend on which options we've previously chosen (if we're imagining choosing our `N` selections sequentially). This doesn't mean that we couldn't sequentially change our value of `p` in some sort of sequentially dynamic process that uses different **multinomial distributions** over time. But, it does mean that for  `N` selections from `n` options drawn from a **multinomial distributions** with a fixed unchanging `p`, the `N` selections are **independent** and do not change in response to each other or affect each other in any way. 

But there might be other examples where this is not true? Can you think of any? How about an example of drawing cards from a deck? Does the probability of drawing an Ace change if you've previously drawn and removed cards from the deck?

# Course Tutorial: Week 02 TUT


## STA130 TUT 02 (Sep13)<br><br> 👨‍💻 👩🏻‍💻 <u>Coding with data types, for loops, and logical control<u>

### ♻️ 📚 Review / Questions [10 minutes]
1. Follow up questions and clarifications regarding regarding **notebooks, markdown, ChatBots, or `Python` code** previously introduced in the Sep06 TUT and Sep09 LEC 

> 1. We're continuing to dive into using **notebooks, markdown**, and `Python` as a tool: building comfort and capability working with **ChatBots** to leverage `Python` code is the objective of the current phase of the course...<br><br>
>
> 2. *Questions about [HW01](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW01_DueSep12.ipynb) which was due yesterday (Thursday Sep12) should be asked in OH from 4-6PM ET Tuesday, Wednesday, and Thursday (see the [Quercus course homepage](https://q.utoronto.ca/courses/354091) for further details)*<br><br>
> 
> 3. *And same story, and even moreso for questions about [HW02](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW02_DueSep19.ipynb) which was due next Thursday (Sep19) (e.g., regarding the "pre-lecture" questions...)*

### 🚧 🏗️ Demo (using Jupyter Notebook and ChatBots) [50 minutes]


#### 1. **[35 of the 50 minutes]** Demonstrate (using a ChatBot to show and explain?) some traditional `python` coding structures

1. `tuple()`, `list()`, vs `dict()  # immutable and mutable "lists" vs key-value pairs`<br><br>
    
2. some `NumPy` functions:<br><br>
    
    1. `import numpy as np`
    2. `np.array([1,2,3]) # a faster "list"`
    3. `np.random.choice([1,2,3])`<br><br>
        
3. `for i in range(n):`, `for x in a_list:`, `for i,x in enumerate(a_list):`, and `print()`<br><br>
    
    1. `variable` as the last line of a "code cell" in a notebook "prints" the value
    2. but `print()` is needed inside `for` loop if you want to output something<br><br>
        
4. `if`/`else` conditional statements<br><br> 

    1. perhaps with `x in b_list`<br>or `i % 2 == 0` to treat evens/odds differently  (sort of like the infamous "FizzBuzz" problem that some people [can't complete](https://www.quora.com/On-average-what-is-the-proportion-of-applicants-that-cannot-pass-a-simple-FizzBuzz-test-based-on-your-personal-experience-or-on-facts) as part of a coding interview challenge)
    2. note the "similarity" to the `try-except` block structure when that's encountered in the code below since it's used there (despite being a "more advanced" 
        
#### 2. **[5 of the 50 minutes]** Reintroduce the [Monty Hall problem](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/GPT/SLS/00001_gpt3p5_MonteHall_ProblemExplanation_v1.md) and see which of the coding structures above you recognize (or do not see) in the Monty Hall simulation code below

#### 3. **[10 of the 50 minutes]** Use any remaining time to start a demonstration of using a ChatBot to (a) understand what the code below is doing and (b) suggest an improved streamlined version of the `for` loop simulation code that might be easier to explain and understand
    
> ChatGPT version 3.5 [was very effective](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/GPT/SLS/00003_gpt3p5_MonteHall_CodeDiscussion_v1.md) for (b), while Copilot was shockingly bad on a [first try](../CHATLOG/wk2/COP/SLS/00001_creative_MonteHall_CodeDiscussion_v1.md) but was able to do better [with some helpful guidance](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/COP/SLS/00002_concise_MonteHall_CodeDiscussion_v2.md).


```python
# Cell to demo what the above code does

```


```python
# Add more code cells to keep a record of the demos

```


```python
# Monte Hall Simulation Code -- not the only way to code this, but it's what Prof. Schwartz came up with...

import numpy as np
all_door_options = (1,2,3)  # tuple
my_door_choice = 1  # 1,2,3
i_won = 0
reps = 100000
for i in range(reps):
    secret_winning_door = np.random.choice(all_door_options)
    all_door_options_list = list(all_door_options)
    # take the secret_winning_door, so we don't show it as a "goat" losing door
    all_door_options_list.remove(secret_winning_door)
    try:
        # if my_door_choice was secret_winning_door then it's already removed
        all_door_options_list.remove(my_door_choice)
    except:
        pass
    # show a "goat" losing door and remove it
    goat_door_reveal = np.random.choice(all_door_options_list)
    all_door_options_list.remove(goat_door_reveal)

    # put the secret_winning_door back in if it wasn't our choice
    # we previously removed it, so it would be shown as a  "goat" losing door
    if secret_winning_door != my_door_choice:
        all_door_options_list.append(secret_winning_door)
    # if secret_winning_door was our choice then all that's left in the list is a "goat" losing door
    # if secret_winning_door wasn't our choice then it's all that will be left in the list

    # swap strategy
    my_door_choice = all_door_options_list[0]

    if my_door_choice == secret_winning_door:
        i_won += 1

i_won/reps
```




    0.66777



### 💬 🗣️ Communication [40 minutes]
    
#### 1. **[5 of the 40 minutes]** Quickly execute the "rule of 5" and take 5 minutes to break into 5 new groups of about 5 students and assign the following 5 questions to the 5 groups (and note that this instruction uses fives 5's 😉). Consider allowing students to preferentially select which group they join by calling for volunteers for each prompt, and feel free to use 5 minutes from the next (dicussion) sections doing so if you choose to (since this could be viewed as being a part of the "discussion").

> *For each of the prompts, groups should consider the pros and cons of two options, the potential impact of a decision to persue one of the options, and how they take into account how uncertainty influences their thinking about the options.<br><br>*
>
> <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
>     
> I asked a ChatBot to create a group activity for you that was related to decision-making under uncertainty using probability, and it produced the following questions. 
> 
> *This is a little bit like one of the ideas in the "Afterward" of HW01 asking a ChatBot to suggest and explain some other, perhaps less well-known "unintuitive surprising statistics paradoxes" (besides the "World War 2 Plane" and "Monte Hall" problems)*
>
> </details>

1. **Stock Investment Strategy:** Students are investors trying to maximize their returns in the stock market. They must decide between two investment strategies: "diversified portfolio" or "focused portfolio." Each strategy has different probabilities of success based on market conditions.<br><br>
    
    1. Diversified Portfolio: Spread investments across multiple industries.
    2. Focused Portfolio: Concentrate investments in a few high-potential stocks.<br><br>
        
2. **Healthcare Treatment Decision:** Students are healthcare professionals deciding between two treatment options for a patient's condition. Each treatment has different success rates and potential side effects.<br><br>
    
    1. Treatment A: High success rate but moderate side effects.
    2. Treatment B: Lower success rate but minimal side effects.<br><br>
        
3. **Sports Team Strategy:** Students are coaches of a sports team planning their game strategy. They must decide between two tactics: "offensive strategy" or "defensive strategy." Each strategy has different probabilities of winning based on the opponent's strengths and weaknesses.<br><br>
    
    1. Offensive Strategy: Focus on scoring goals/points aggressively.
    2. Defensive Strategy: Prioritize defense to prevent the opponent from scoring.<br><br>
        
4. **Career Path Decision:** Students are recent graduates deciding between two career paths: "corporate job" or "entrepreneurship." Each path has different probabilities of success and factors to consider, such as job security, income potential, and work-life balance.<br><br>
    
    1. Corporate Job: Stable income but limited growth opportunities.
    2. Entrepreneurship: Higher potential for success but greater risk and uncertainty.<br><br>
        
5. **Environmental Conservation Strategy:** Students are environmental activists advocating for conservation efforts in a wildlife reserve. They must decide between two conservation strategies: "habitat preservation" or "species reintroduction." Each strategy has different probabilities of achieving long-term sustainability for the ecosystem.<br><br>
    
    1. Habitat Preservation: Protect existing habitats from human encroachment.
    2. Species Reintroduction: Reintroduce endangered species to restore ecological balance.


#### 2. **[15 to 20 of the 40 minutes]** Each group plans and prepares a brief (approximately 3 minute) summary (a) introducing their problem context and (b) outlining their decision and the rationale behind it. The group presentations should address

1. the expected outcomes of their decision
2. the risks involved
3. and why they believe their choice is the best in light of their characterization of the degree uncertainty present in their context 

#### 3. **[15 to 20 of the 40 minutes]** Each group gives their (approximately 3 minute) planned presentation. If time permits, engaging in some (students or TA) Q&A seeking clarification or challenging group decisions would be ideal.

Groups who manage to plan a presentation where multiple group members actively part of the presentation should be awarded "gold stars", figuratively, of course, since actualy gold, or stars, of gold star stickers are unfortunately momentarily in short supply


# Course Lecture: Week 02 LEC

## Probability and Coding

#### 1. Conditional Probability and Indepdenence

1. **Probability** 

    $\displaystyle \Pr(A)\quad \textrm{or} \quad\Pr(X=x)$<br><br>
    
2. **Conditional Probability** 

    $\displaystyle \Pr(\;A\,|\,B\;)\quad$ or $\quad\Pr(\; Y=y\,|\,X=x\;)$<br>
    
    ChatBots are something like the following specifications...

    1. **Markov**: $\Pr(\; W_{i+1}=w_{i+1}\,|\,W_i=w_i\;)$  
    2. **Bigram**: $\Pr(\; W_{i+2}=w_{i+2}\,|\, W_{i+1}=w_{i+1}, W_i=w_i\;)$  
    3. **Trigram**: $\Pr(\; W_{i+3}=w_{i+3} \,|\, W_{i+2}=w_{i+2}, W_{i+1}=w_{i+1}, W_i=w_i\;)$ 
    4. **Context**: $\Pr(\; W_{i+3}=w_{i+3} \,|\, W_{i+1}=w_{i+1}, W_i=w_i, C=c\;)$<br><br>

3. **Independence** 

    $\displaystyle \Pr(A)=\Pr(\;A\,|\,B\;)\quad$ or $\quad\Pr(Y=y) = \Pr(\; Y=y\,|\,X=x\;)$

#### 2. Multinomial distributions

1. `from scipy import stats`
2. `stats.multinomial(p=probability, n=categories).rvs(size=attempts)`
3. `import numpy as np` and `np.array()`
4. `np.random.seed(initialization)` and `np.random.choice(options, size=draws, replace=True, p=None)`

#### 3. python string manipulation for a Markovian ChatBot

- `avatar.dtypes` and `df.col.str.upper()`
    - `.replace` and `import re` "regular expressions" ("regexp") are demonstrated but will not be tested 
- **Operator overloading** `+` and `.sum().split(' ')`
- `for i in range(n)` and `for x in lst` and `for i,x in enumerate(lst)`
- `list()` and `dict()`
- `if`/`else`



```python
import numpy as np
np.random.choice?
```


```python
import pandas as pd
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-08-11/avatar.csv"
avatar = pd.read_csv(url) #avatar.isnull().sum() #avatar[avatar.isnull().sum(axis=1)>0]
avatar[:10]
```


```python
avatar.dtypes
```


```python
avatar.character.value_counts()#[:10]
```


```python
avatar.chapter.value_counts()#[:10]
```


```python
#words = ("\n"+avatar.dropna().character.str.upper()+": "+avatar.dropna().character_words+" ").sum().split(' ')
#words = ("\n"+avatar.dropna().character.str.upper()+": "+avatar.dropna().character_words+" ").sum().split(' ')
words = ("\n"+avatar.character.str.upper().str.replace(' ','.')+": "+avatar.full_text+" ").sum().split(' ')
```


```python
#from collections import defaultdict
word_used = dict()#defaultdict(int)
next_word = dict()#defaultdict(lambda: defaultdict(int))
for i,word in enumerate(words[:-1]):
    
    if word in word_used:
        word_used[word] += 1
    else: 
        word_used[word] = 1
        next_word[word] = {}
        
    if words[i+1] in next_word[word]:
        next_word[word][words[i+1]] += 1 
    else:

        next_word[word][words[i+1]] = 1
```


```python
next_word
```


```python
import numpy as np
from scipy import stats
```


```python
current_word = "\nKatara:".upper()
print(current_word, end=' ')
for i in range(100):
    probability_of_next_word = np.array(list(next_word[current_word].values()))/word_used[current_word]
    randomly_chosen_next_word = stats.multinomial(p=probability_of_next_word, n=1).rvs(size=1)[0,:]
    current_word = np.array(list(next_word[current_word].keys()))[1==randomly_chosen_next_word][0]
    print(current_word, end=' ')
```


```python
import re
avatar.full_text = avatar.full_text.apply(lambda string: re.sub(r'\[.*?\]', lambda match: match.group(0).replace(' ', '_ '), string))
avatar.loc[avatar.character=='Scene Description','full_text'] = avatar.full_text[avatar.character=='Scene Description'].str.replace(' ', '- ')
words = ("\n"+avatar.character.str.upper().str.replace(' ','.')+": "+avatar.full_text+" ").sum().split(' ')
```


```python
avatar[:10]
```


```python
word_used2 = defaultdict(int)
next_word2 = defaultdict(lambda: defaultdict(int))
for i,word in enumerate(words[:-2]):
    word_used2[word+' '+words[i+1]] += 1
    next_word2[word+' '+words[i+1]][words[i+2]] += 1 
```


```python
next_word2
```


```python
current_word_1 = "\nKatara:".upper()
current_word_2 = "Water."
print(current_word_1, end=' ')
print(current_word_2, end=' ')
for i in range(100):
    probability_of_next_word = np.array(list(next_word2[current_word_1+' '+current_word_2].values()))/word_used2[current_word_1+' '+current_word_2]
    randomly_chosen_next_word = stats.multinomial(p=probability_of_next_word, n=1).rvs(size=1)[0,:]
    current_word_1,current_word_2 = current_word_2,np.array(list(next_word2[current_word_1+' '+current_word_2].keys()))[1==randomly_chosen_next_word][0]
    print(current_word_2.replace('_', '').replace('-', ''), end=' ')
```


```python
word_used3 = defaultdict(int)
next_word3 = defaultdict(lambda: defaultdict(int))
for i,word in enumerate(words[:-3]):
    word_used3[word+' '+words[i+1]+' '+words[i+2]] += 1
    next_word3[word+' '+words[i+1]+' '+words[i+2]][words[i+3]] += 1 
```


```python
current_word_1 = "\nKatara:".upper()
current_word_2 = "Water."
current_word_3 = "Earth."
print(current_word_1, end=' ')
print(current_word_2, end=' ')
print(current_word_3, end=' ')
for i in range(100):
    probability_of_next_word = np.array(list(next_word3[current_word_1+' '+current_word_2+' '+current_word_3].values()))/word_used3[current_word_1+' '+current_word_2+' '+current_word_3]
    randomly_chosen_next_word = stats.multinomial(p=probability_of_next_word, n=1).rvs(size=1)[0,:]
    current_word_1,current_word_2,current_word_3 = current_word_2,current_word_3,np.array(list(next_word3[current_word_1+' '+current_word_2+' '+current_word_3].keys()))[1==randomly_chosen_next_word][0]
    print(current_word_3.replace('_', '').replace('-', ''), end=' ')
```


```python
from collections import Counter, defaultdict
characters = Counter("\n"+avatar.character.str.upper().str.replace(' ','.')+":")

nested_dict = lambda: defaultdict(nested_dict)
word_used2C = nested_dict()
next_word2C = nested_dict()

for i,word in enumerate(words[:-2]):
    
    if word in characters:
        character = word
        
    if character not in word_used2C:
        word_used2C[character] = dict()
    if word+' '+words[i+1] not in word_used2C[character]:
        word_used2C[character][word+' '+words[i+1]] = 0
    word_used2C[character][word+' '+words[i+1]] += 1

    if character not in next_word2C:
        next_word2C[character] = dict()
    if word+' '+words[i+1] not in next_word2C[character]:
        next_word2C[character][word+' '+words[i+1]] = dict()
    if words[i+2] not in next_word2C[character][word+' '+words[i+1]]:
        next_word2C[character][word+' '+words[i+1]][words[i+2]] = 0
    next_word2C[character][word+' '+words[i+1]][words[i+2]] += 1
        
        
```


```python
current_word_1 = "\nKatara:".upper()
current_word_2 = "Water."
print(current_word_1, end=' ')
print(current_word_2, end=' ')
for i in range(100):
    if current_word_1 in characters:
        character = current_word_1

    probability_of_next_word = np.array(list(next_word2C[character][current_word_1+' '+current_word_2].values()))/word_used2C[character][current_word_1+' '+current_word_2]
    randomly_chosen_next_word = stats.multinomial(p=probability_of_next_word, n=1).rvs(size=1)[0,:]
    current_word_1,current_word_2 = current_word_2,np.array(list(next_word2C[character][current_word_1+' '+current_word_2].keys()))[1==randomly_chosen_next_word][0]
    print(current_word_2.replace('_', '').replace('-', ''), end=' ')
```


```python

```


```python

```

# Course Homework: Week 02 HW


## STA130 Homework 02

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
- [0.3 points]: Assignment completion confirmed by working "final" code and ChatBot summaries for "3"
- [0.3 points]: Written submission evaluation and enagement confirmation with ChatBot summaries for "6"
- [0.3 points]: Evaluation of engagement and evaluation of written communication in "7"
        

### "Pre-lecture" HW [*completion prior to next LEC is suggested but not mandatory*]

#### 1. Begin (or restart) part "3(a)" of the **TUT Demo** and interact with a ChatBot to make sure you understand how each part the Monte Hall problem code above works<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _ChatBots typically explain code fairly effectively, so a ChatBot will probably be very helpful if you share the full Monte Hall problem code; but, you can always introduce more specific and targetted follow-up prompts that help with focus, re-redirection, and response format regarding the ChatBot responses as needed._ 
>
> _ChatBots won't always re-introduce and re-explain the Monte Hall problem itself, so if you need it to do so you may need to specifically request this as part of your prompt or follow up interactions._

</details>


#### 2. Extend your ChatBot sessions to now address part "3(b)" of the **TUT Demo** and interact with your ChatBot to see if it can suggest a simpler, more streamlined way to code up this *for* loop simulation so the process is more clear and easier to understand; then, describe any preferences you have in terms of readibility or explainability  between the original code and the code improvements suggested by the ChatBot<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _The links in the TUT Demo show that there can be variation in the quality of the code improvements suggested by ChatBots; however, it's most likely that a ChatBot is going to be able to greatly reduce the number of steps/lines of code and hence complexity of understanding the problem. ChatBots can provide a good amount of explanation and inline clarifying code comments and provide more simpler more intuitive code that can transform something that looks a bit scary at first to something that's easy to follow and make sense of. Of course, in doing so, a ChatBot may introduce functions that you've technically not learned or seen before; but, the amount of simplification and clarifying comments is probably going to more than compensate for this; and, you'll have seen a learned a little bit more about what's possible through this process, which is the ideal experience we're hoping you'll see here._ 
    
</details>
        

#### 3. Submit your preferred version of the Monty Hall problem that is verified to be running and working with a final printed output of the code; then, add code comments explaining the purpose of each line of the code<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _Your ChatBot will likely do much of this for you, but verify for yourself that you understand each comment and reword comments wherever you think it would be better to explain it differently._
>
> _Remember to ask for summaries of your current session and paste these into your homework notebook  (including link(s) to chat log histories if you're using ChatGPT)_

</details>
 

#### 4. Watch the embedded video tutorial on Markov chains in the next Jupyter cell below to understand their application and relevance for ChatBots; then, after watching the video, start a new ChatBot session by prompting that you have code that creates a "Markovian ChatBot"; show it the first version of the "Markovian ChatBot code" below; and interact with the ChatBot session to make sure you understand how the original first version of the "Markovian ChatBot code" works<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _If the ChatBot prompts you as to how you will "train" your own "Markovian ChatBot" you can reply that you'll just use a series of stories with a lot of different characters_
> 
> _Ask for summaries of this second ChatBot session and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)_
    
</details>
    


```python
# Markov Chains and Text Generation
from IPython.display import YouTubeVideo
YouTubeVideo('56mGTszb_iM', width = 550)
```

<details class="details-example"><summary style="color:blue"><u>Continue now...?</u></summary>

### Pre-lecture VS Post-lecture HW

Feel free to work on the "Post-lecture" HW below if you're making good progress and want to continue: for **HW 02** continuing could be reasonable because questions "5-7" below directly follow up and extend "Pre-lecture" HW question "4"

*The benefits of continue would are that (a) it might be fun to try to tackle the challenge of working through some problems without additional preparation or guidance; and (b) this is a very valable skill to be comfortable with; and (c) it will let you build experience interacting with ChatBots (and beginning to understand their strengths and limitations in this regard)... it's good to have sense of when using a ChatBot is the best way to figure something out, or if another approach (such as course provided resources or a plain old websearch for the right resourse) would be more effective*
    
</details>    

### "Post-lecture" HW [*submission along with "Pre-lecture" HW is due prior to next TUT*]

#### 5. Recreate (or resume) the previous ChatBot session from question "4" above, and now  prompt the ChatBot session that you have a couple extensions of the code to show it, and then show it each of the extentions of the "Markovian ChatBot code" below in turn



1. Without just supplying your ChatBot session with the answers, see if the ChatBot can figure out what the extensions in the code do; namely, making character specific Markov chains, and using bigrams (rather than just the previous word alone) dependency... prompt your ChatBot session with some hints if it's not seeming to "get it"<br><br>
    
2. Interact with your ChatBot session to have it explain details of the code wherever you need help understanding what the code is doing and how it works<br><br>
    
3. Start yet another new ChatBot session and first show the ChatBot the original "Markovian ChatBot code" below, and then tell ChatBot that you have an extension but this time just directly provide it the more complicated final extension without ever providing the intermediate extension code to the ChatBot session and see if it's still able to understand everything extension does; namely, making character specific Markov chains, and using bigrams (rather than just the previous word alone) dependency... prompt the ChatBot with some hints if it's not seeming to understand what you're getting at...<br><br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> **ALERT: Time Warning**. Regarding the comments below (which will likely be relevant and useful for you), you might find the potential learning experience that this provides to be a quite the rabbit total rabbit hole and time sink. You might end up finding out that you spent way more time than I should on learning the code!! So be mindful of your time management as there is much to do for many classes!
>    
> _As you may or may not have already experienced in the previous problem, a ChatBot applied to this problem is likely to start explaining a bit more knowledge about Python than you need to know (as a student just trying to learn stats+DS); however, you'll probably feel like this "out of scope" context information is helpful to know (or at least be aware of) and easy to understand and learn if you use some addtional prompts to dig deeper into them. A ChatBot will be quite good at explaining and helping understand smaller chunks of code; however, if given too much information at once it can gloss over some information._
>   
> _That said, some topics here are potentially quite and advanced and too tricky! You might be able to ask the ChatBot to simplify its explanations and that might help a bit. But on the other hand, some topics, such as, "how does `nested_dict = lambda: defaultdict(nested_dict)` work?" might just simply be too advanced to really admit a simpler explanation via a ChatBot. You'll have to let these sorts of things go, if you come across explanations that just aren't improving or helping at at. In the case of `defaultdict(nested_dict)` specifically, the details here are well beyond the scope of STA130 and can be very safely ignored for now. The code will have reviewed and "walked thorugh" in LEC, but the perspectives espoused there will be the extent of the formal commentary and information regarding the coding topics we encounter in the Markov ChatBots code here._
>     
> _Unlike with the Monte Hall problem, we will not inquire with the ChatBot to see if it can suggest any streamlining, readability, or usability improvements to the alternative versions of the "Markovian ChatBot code" we're examining_
>     
> - _because doing so seems to result in the attempted creation of dubiously functional modular code with a focus on reusability (which is likely a result of ChatBot design being primarily a "computer science" topic), so ChatBot reponses here tend to orient around programming and system design principles (despite "Markovian" very much being a "statistics" topic)_
>     
> _Programming and system design principles are beyond the scope of STA130; but, they are critical for modern data science careers... if you are interested in pursuing a data science career, it is imperitive that you complete courses like CSC263, CSC373, and perhaps an additional "systems design" course_
> 
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot)_
    
</details>
     

#### 6. Report on your experience interacting with ChatBots to understand the Monte Hall problem and "Markovian ChatBot" code

1. Discuss how quickly the ChatBot was able to be helpful for each of the above questions, and if so, how?<br><br>
    
2. Discuss whether or not interacting with ChatBot to try to figure things out was frustrating or unhelpful, and if so, how?<br><br>
    
3. Based on your experiences to date (e.g., including using ChatBots to troubleshoot coding errors in the previous homework), provide an overall assessment evaluating the usefulness of ChatBots as tools to help you understand code<br>

#### 7. Reflect on your experience interacting with ChatBot and describe how your perception of AI-driven assistance tools in the context of learning coding, statistics, and data science has been evolving (or not) since joining the course<br><br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _Question "7" and the next question "8" are somewhat related to the first bullet point in the suggested interactions of the "Afterword" to the Homework from last week... consider reviewing that if you'd like a little extra orienting around what these questions are trying to have you explore_
   
</details>

#### 8. ChatBots consume text data available on the web or platforms, and thus represents a new way to "search consensensus" that condenses and summarizes mainstream human thought<br><br>

1. Start a new ChatBot session and discuss the relevance of learning and adaptability, communication, coding, and statistics and data analysis as skills in the modern world, especially with respect to career opportunities (particularly in the context of the data science industry)<br><br>
    
2. See if ChatBot thinks you could be a statistician or data scientist without coding or doing data analysis, and then transition your ChatBot conversation into a career exploration discussion, using the ChatBot to identify the skills that might be the most valuable for a career that you're interested<br><br>
    
3. Ask for a summary of this ChatBot session and paste it into your homework notebook (including link(s) to chat log histories if you're using ChatBot)<br><br>
    
4. Paraphrase the assessments and conclusions of your conversation in the form of a reflection on your current thoughts regarding your potential future career(s) and how you can go about building the skills you need to pursue it<br><br>

5. Give your thoughts regarding the helpfulness or limitations of your conversation with a ChatBot, and describe the next steps you would take to pursue this conversation further if you felt the information the ChatBot provides was somewhat high level and general, and perhaps lacked the depth and detailed knowledge of a dedicated subject matter expert who had really take the time to understand the ins and outs of the industry and career path in question.
<br><br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _While question 8 is not a part of the rubric, it is nonetheless a very good exercise that will likely be valuable for you if you engage it them sincerely_
    
</details>


#### 9. Have you reviewed the course [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br><br>
  
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
    
</details>


```python
# Markovian Chatbot

# from collections import defaultdict
word_used = dict() # defaultdict(int)
next_word = dict() # defaultdict(lambda: defaultdict(int))
for i,word in enumerate(words[:-1]):

    if word in word_used:
        word_used[word] += 1
    else:
        word_used[word] = 1
        next_word[word] = {}

    if words[i+1] in next_word[word]:
        next_word[word][words[i+1]] += 1
    else:
        next_word[word][words[i+1]] = 1
```


```python
# Markovian Chatbot Extension #1

word_used2 = defaultdict(int)
next_word2 = defaultdict(lambda: defaultdict(int))
for i,word in enumerate(words[:-2]):
    word_used2[word+' '+words[i+1]] += 1
    next_word2[word+' '+words[i+1]][words[i+2]] += 1 
```


```python
# Markovian Chatbot Extension #2

from collections import Counter, defaultdict
# `avatar` is a dataset, and `character` is one of it's columns
characters = Counter("\n"+ avatar.character.str.upper().str.replace(' ','.')+":")
# this code changes the type of the `character` column to `str`; then,
# makes the text uppercase, and replaces spaces with '.'

nested_dict = lambda: defaultdict(nested_dict)
word_used2C = nested_dict()
next_word2C = nested_dict()

for i,word in enumerate(words[:-2]):
    if word in characters:
        character = word
        
    if character not in word_used2C:
        word_used2C[character] = dict()
    if word+' '+words[i+1] not in word_used2C[character]:
        word_used2C[character][word+' '+words[i+1]] = 0
    word_used2C[character][word+' '+words[i+1]] += 1
    
    if character not in next_word2C:
        next_word2C[character] = dict()
    if word+' '+words[i+1] not in next_word2C[character]:
        next_word2C[character][word+' '+words[i+1]] = dict()
    if words[i+2] not in next_word2C[character][word+' '+words[i+1]]:
        next_word2C[character][word+' '+words[i+1]][words[i+2]] = 0
    next_word2C[character][word+' '+words[i+1]][words[i+2]] += 1
```

## Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course <br><br>


<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations

    
> 1. If you've not heard of the "reproducibility crisis" in science, have a ChatBot explain it to you
> 2. If you've not heard of the "open source software" (versus proprietary software), have a ChatBot explain it to you
> 3. "Reproducibility" can also be considered at the level of a given data analysis project: can others replicate the results of code or analysis that you've done?
>    1. Discuss with a ChatBot how jupyter notebooks and github can be used facilitate transparency and reproducibility in data analysis
> 4. Discuss with a ChatBot what the distinction is between replicability of scientific experiments, versus the replicability of a specific data analysis project, and what your responsibility as an analyst should be with respect to both
> 5. Do you think proprietary (non "open source software") software, such as Microsoft Word, Outlook, and Copilot tends to result in high quality products?  
>     1. Do you think software product monopolies (such as the UofT dependence on Microsoft products) makes the world a better place?
</details>    

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>

### Current Course Project Capability Level
   
**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.

> At this point in the course you should be able to create a `for` loop to iterate through and provide **simple summaries** of some of the interesting columns in the course project data
>
> 1. Create two versions of the code, one for numeric and the other for categorical data,  which provide a printout format that displays relavent summaries and the missing data counts for a given set of (either numerical or categorical) columns being examined
>
> 2. Combine the two separate `for` loops into a single `for` loop using an `if`/`else` **conditional logic structure** that determines the correct printout format based on the data type of the column under consideration  
>     1. *Being able to transform existing code so it's "resuable" for different purposes is one version of the programming design principle of "polymorphism" (which means "many forms" or "many uses") [as in the first task above]*
>     2. *A better version of the programming design principle of "polymorphism" is when the same code can handle different use cases [as in the second tast above]*
>     3. *Being able run your code with different subsets of columns as interest in different variables changes is a final form of the programming design principle of "polymorphism" that's demonstrated through this exercise*   
    
</details>        
