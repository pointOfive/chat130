# STA130 TUT 11 (Nov15)<br><br> 🌳🌲 <u>Decision Tree Classification <br>with _sklearn_ and confusion matrices<u>


## ♻️ 📚 Review  / Questions [30 minutes]

### 1. **[15 of the 30 minutes]** Follow up clarification questions regarding **multiple linear regression**?  Which **classification** introduced today builds directly upon...

> 1. How are **predictors variables** used to **predict** an **outcome variable**? E.g.,
> 
>    $$Y_i = \beta_0 + \beta_1x_i + \beta_2 1_{B}(k_i) + \beta_3 1_{C}(k_i) + \beta_4x_i1_{B}(k_i) + \beta_5 1_{B}(k_i)1_{C}(k_i) + \beta_6x_i^2$$
> 
>
> 2. What's the difference between **predictive performance** and **coefficient hypothesis testing** using p-values?
> 3. Can **R-squared** ever get worse with each additional predictor variable added to the model?

### 2. **[15 of the 30 minutes]** **Train-Test** "in sample" versus "out of sample" validation



```python
from sklearn import datasets
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
```


```python
cancer_data = datasets.load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer_data.data, 
                         columns=cancer_data.feature_names)
```


```python
# Notice the binary 'target' 
# with 'target_names': array(['malignant', 'benign'], dtype='<U9')
cancer_data
```


```python
print(cancer_data['DESCR'])
```


```python
import matplotlib.pyplot as plt
for i in cancer_df:
    if all(cancer_df[i]>0):
        cancer_df[i]=np.log(cancer_df[i])

_ = cancer_df.plot()
```


```python
# Randomly split into 50% training data and 50% testing data
# Why?
np.random.seed(130)
training_indices = cancer_df.sample(frac=0.5, replace=False).index.sort_values()
testing_indices = cancer_df.index[~cancer_df.index.isin(training_indices)]
```


```python
# Recall the binary 'target' 
# But notice here the outcome variable is continuous (not binary) 
# (and same for the predictor variables, binary 'target' not used)
linear_spec_4 = '''
scale(Q('mean area')) ~ scale(Q('texture error')) + scale(Q('smoothness error'))
                      + scale(Q('mean fractal dimension')) + scale(Q('mean smoothness'))
                      + scale(Q('mean symmetry')) * scale(Q('area error'))
                      * scale(Q('worst texture'))
                      * scale(Q('worst smoothness'))
                      * scale(Q('worst symmetry'))
                      * scale(Q('worst concave points'))
                      * scale(Q('worst compactness'))
                      * scale(Q('worst concavity'))
'''

linear_spec_3 = '''
scale(Q('mean area')) ~ scale(Q('texture error')) * scale(Q('smoothness error'))
                      * scale(Q('mean fractal dimension')) * scale(Q('mean smoothness'))
                      * scale(Q('mean symmetry')) 
'''

linear_spec_2 = '''
scale(Q('mean area')) ~ scale(Q('texture error')) + scale(Q('smoothness error'))
                      + scale(Q('mean fractal dimension')) + scale(Q('mean smoothness'))
                      + scale(Q('mean symmetry')) 
'''
linear_spec_1 = '''
scale(Q('mean area')) ~ scale(Q('texture error')) + scale(Q('smoothness error'))
'''
MLR = smf.ols(linear_spec_3, data=cancer_df.loc[training_indices,:])
MLR_fit = MLR.fit() # Fit the mulitple lienar regression model (MLR)

```


```python
# "In sample" performance based on the "training data"
np.corrcoef(MLR_fit.predict(cancer_df.loc[training_indices,:]),
            cancer_df.loc[training_indices,"mean area"])[0,1]**2
```


```python
# "Out of sample" performance based on the "testing data"
# Why?
np.corrcoef(MLR_fit.predict(cancer_df.loc[testing_indices,:]),
            cancer_df.loc[testing_indices,"mean area"])[0,1]**2
```

- `linear_spec_1`: The model fit as evaluated by R-squared the "proportion of variation explained" is better for data the model was trained on and worse for new data 
    - Why might this be expected?  What explains this?


- `linear_spec_2`: The "in sample" performance remains better compared to the "out of sample" performance, but overall performance is improved and relatively speaking the "out of sample" performance 
    - Why might both of these be expected?  What explains these?


- `linear_spec_3`: something has gone wrong
    - `linear_spec_4`: something has gone very wrong


- When "out of sample" predictive "proportion of variation explained" performance is not much worse than the "in sample" performance, then this suggests that the model seems to represent the data fairly well...
    - Why?




## 🚧 🏗️ Demo (Introducing Classification Decision Trees and Confusion Matrices) [30 minutes] 

### 1. **[12 of the 30 minutes]** The concept of a *classification decision tree*

Some questions to keep in mind (and hopefully answer) as you go

- What's similar about this code and the **multiple linear regression code** above? 
- A **classification decision tree** is a model like **multiple linear regression**... how so?
- **Predictions** of **classification decision tree** are made differently than those of **multiple linear regression**... how so?



```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Initialize and train the Decision Tree classifier
clf = DecisionTreeClassifier(max_depth=2, random_state=42)

# Specify outcome and "Design Matrix"
# y ~ col1 + col2 + ... + colk
clf.fit(X=cancer_df.iloc[training_indices, :], 
        y=cancer_data.target[training_indices])
# The outcome has changed compared to the earlier multiple linear regression model...
```


```python
# Details regarding code are not the point and will be discused and explore later in LEC

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plot_tree(clf, feature_names=cancer_data.feature_names.tolist(), 
          class_names=cancer_data.target_names.tolist(), 
          filled=True, rounded=True)
plt.title("Decision Tree (Depth = 2)")
plt.show()
```

###  2. **[18 of the 30 minutes]** The **concept** of confusion matrix... 

> Details regarding code are not the point and will be discused and explore later in LEC

Some questions to keep in mind (and hopefully answer) as you go

- **Classification** and **regression** are distinguised by the nature of the **outcome variables** they predict... How so? 
- What should be similar about **model performance evaluation** for **classification decision tree** to what we've seen previously?
- Are multiple kinds of **model performance evaluation metrics** possible for **classification decision tree** based on the **confusion matrix**?



```python
# Make Benign 0 and Malignant 1
# JUST RUN THIS ONCE!!
cancer_data.target = 1-cancer_data.target 
cancer_data.target_names = np.array(['Benign\n(negative)','Malignant\n(positive)'])
```


```python
# Details regarding code are not the point and will be discused and explored later in LEC

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on the test set
y_pred_depth_2 = clf.predict(cancer_df.iloc[testing_indices, :])

# Generate the confusion matrix
conf_matrix = confusion_matrix(cancer_data.target[testing_indices], y_pred_depth_2)

# Get the target names for 'benign' and 'malignant'
target_names = cancer_data.target_names.tolist()

# Set up a confusion matrix with proper labels using target names
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Predicted {label}' for label in target_names], 
            yticklabels=[f'Actually {label}' for label in target_names])

plt.title('Confusion Matrix (Depth = 2)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add custom labels for FP, FN, TP, and TN
plt.text(0.5, 0.1, "True Negative (TN)", fontsize=12, color='red', ha='center', va='center')
plt.text(1.5, 0.1, "False Positive (FP)", fontsize=12, color='red', ha='center', va='center')
plt.text(0.5, 1.1, "False Negative (FN)", fontsize=12, color='red', ha='center', va='center')
plt.text(1.5, 1.1, "True Positive (TP)", fontsize=12, color='red', ha='center', va='center')

plt.show()
```

## 💬 🗣️ Communication Activity **[40 minutes]**

1. **[7 minutes]** Break into your **course project groups** and confer regarding the purpose of the **train-test** "in sample versus out-of-sample" **validation** framework.


2. **[6 minutes]** Identify an answer to the previous question which all groups in agreement are as close to unanimously satisfied with as possible (in the allotted time). 


3. **[7 minutes]** Have two group provide an example scenario in which the implications of **Type I errors** (wrongly rejecting the **null hypothesis**) and **Type II errors** (failing to reject a **false null hypothesis**) can be demonstrated. 


4. **[8 minutes]** Have to other groups provide an example context involving **predictions** using a **classification decision tree** in which the implications of **false positive (FP)** and **false negative (FP) predictions** can be discussed and considered.


5. **[6 minutes]** Lead a discussion with all groups exploring the analogy (and distinction) between **Type I/Type II errors** and **false positive/negative (FP/FN) predictions**. 


6. **[6 minutes]** Lead a discussion with all groups exploring what they believe the  meaningfulness, purpose, or relevance of the **train-test** "in sample versus out-of-sample" **validation** framework is relative to the **errors** considered here.


7. If time permits, explore with all the groups whether or not they have identified any notion of a difference between the natures of the consequences of **false positive** versus **false negative (FP/FN) predictions**.

