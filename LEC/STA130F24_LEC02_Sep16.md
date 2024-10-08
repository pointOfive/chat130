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
