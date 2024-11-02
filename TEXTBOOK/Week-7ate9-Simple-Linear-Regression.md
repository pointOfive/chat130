# Normal Distributions gettin' jiggy wit it

**LEC 1 New Topics**

1. [Correlation Association (IS NOT Causation)](week-7ate9-Simple-Linear-Regression#Correlation-Association-IS-NOT-Causation)
    1. [DO NOT USE Correlation to Measure ANYTHING EXCEPT "Straight Line" Linear Association](week-7ate9-Simple-Linear-Regression#DO-NOT-USE-Correlation-to-Measure-ANYTHING-EXCEPT-Straight-Line-Linear-Association)
    2. [Correlation is just for Y = mx + b](week-7ate9-Simple-Linear-Regression#Correlation-is-just-for-y--mx--b)
2. [Simple Linear Regression is Just a Normal Distribution](week-7ate9-Simple-Linear-Regression#Simple-Linear-Regression-is-Just-a-Normal-Distribution)
    1. [Terminology: predictor, outcome, intercept and slope coefficients, and error terms](week-7ate9-Simple-Linear-Regression#Terminology-predictor-outcome-intercept-and-slope-coefficients-and-error-terms)

**TUT/HW Topics**

1. [_import statsmodels.formula.api as smf_](week-7ate9-Simple-Linear-Regression#statsmodel)
2. [_smf.ols_](week-7ate9-Simple-Linear-Regression#smf-ols])
    1. ["R-style" formulas I](week-7ate9-Simple-Linear-Regression#r-style-formulas-i])
    2. ["quoting" non-standard columns](week-7ate9-Simple-Linear-Regression#quoting])
3. [_smf.ols("y~x", data=df).fit()_ and _.params_](week-7ate9-Simple-Linear-Regression#fitting-models) $\hat \beta_k$ versus $\beta_k$
    1. [_.fittedvalues_](week-7ate9-Simple-Linear-Regression#fittedvalues)
    2. [_.rsquared_ "variation proportion explained"](week-7ate9-Simple-Linear-Regression#rsquared-variation-proportion-explained)
    3. [_.resid_ residuals and assumption diagnostics](week-7ate9-Simple-Linear-Regression#resid-residuals-and-assumption-diagnostics)
4. [_smf.ols("y~x", data=df).fit().summary()_ and _.tables[1]_ for Testing "On Average" Linear Association](week-7ate9-Simple-Linear-Regression#testing-on-average-linear-association)

**LEC 2 New Topics / Extensions**

1. [Two(2) unpaired samples group comparisons](week-7ate9-Simple-Linear-Regression#two2-unpaired-samples-group-comparisons)
2. [Two(2) unpaired sample permutation tests](week-7ate9-Simple-Linear-Regression#two2-unpaired-sample-permutation-tests)
3. [Two(2) unpaired sample bootstrapping](week-7ate9-Simple-Linear-Regression#two2-unpaired-sample-bootstrapping)
4. [Indicator variables and contrasts linear regression](week-7ate9-Simple-Linear-Regression#indicator-variables-and-contrasts-linear-regression)

**Out of scope:**

1. Material covered in future weeks
2. Anything not substantively addressed above...
3. ...such as all the stuff around multi/bivariate normal distribution and their covariance matrices, ellipses and their math and visual weirdness outside of a 1:1 aspect ratio, and eigenvectors and eigenvalues and major axis lines, etc...
4. ...such as the mathematical formulas correlation, but just noting that they sort of just look like formulas for variance...


## LEC 1 New Topics

### Correlation Association (is Not Causation)

For two random variables, $X$ and $Y$, **correlation** is notated as $r_{X,Y} = Cor(X,Y)$. Correlation may between $1$ and $-1$, with either indicating "perfect straight line correlation" and $0$ indicating "no linear association". 

If two random variables
like $X$ ice cream sales in a month and $Y$ shark attacks in a month "positively correlate", then $r_{X,Y}>0$.  But this does not mean there is a "causation" relationship between them. Or what about pirates versus climate change? These are "negatively correlated" since there's a lot less pirates and now while at the same time there's also observably more changes in the atmosphere. So has the loss of pirating on the high seas has caused climate change? No. 

A better explanation for these are that "summer" is the reason for increased ice cream sales $X$ and shark attacks $Y$, and "modernization" which has reduced pirate activity while increasing human pollution. 

There are two other considerations to keep in mind as well regarding the "causation is not correlation" adage.

1. Are the variables being measured the *real causes*? Or are they *just proxies* for what's really happening? For example, do parents heights *cause* their children's heights? Well, no, it's actually their *genes* and those subsequent effects that are what actually contribute to a child's height. Parent heights is just a proxy for their genes. 
2. Real causation is *complicated* so there are *ALWAYS* MANY COMPLEX THIGNS contributing to causal pathways. If we say $X$ causes $Y$ we have already started to lose the more holistic perspective of the situation which is actually the more correct view. We should avoid oversimplifications that lead us towards an incomplete view of what's actually happening. Just say $X$ and $Y$ correlate. 

#### DO NOT USE Correlation to Measure ANYTHING EXCEPT "Straight Line" Linear Association

**Correlation** measures the strength of "straight line" **linear association** between two **random variables** $X$ and $Y$ in a **scatter plot** of **observational pairs** of data points $(x_i, y_i)$. When there is a lot of data relative to the spread of the data, the scatter plot actually tends to have an ellipse "American football shape". And the numeric correlation value can tell you exactly the "wideness" that the "football shape" takes on. You would be able to easily find examples of how to interpret the numeric correlation value online. In class I thought a fun way to familiarize yourself with this was to [play a game](https://www.guessthecorrelation.com/).

The following is **Anscombe's Quartet** which is the classic demonstration showing that correlation as a measure the strength of "straight line" **linear association** only "works" if there is *indeed* **linear association** in the data. Below this figure the "American football shape" that can be seen in Galton's famous "regression to the mean" data. 

```python
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Load Anscombe's quartet from Seaborn
df = sns.load_dataset("anscombe")

# Function to calculate stats (correlation, mean, and std)
def calculate_stats(subset):
    correlation = np.corrcoef(subset['x'], subset['y'])[0, 1]
    mean_x = subset['x'].mean()
    mean_y = subset['y'].mean()
    std_x = subset['x'].std()
    std_y = subset['y'].std()
    return correlation, mean_x, mean_y, std_x, std_y

# Create a 2x2 subplot layout with Plotly
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.1, 
                    subplot_titles=[f"Dataset {d}" for d in df['dataset'].unique()])

# Plot scatter and regression line for each dataset
for i, dataset in enumerate(df['dataset'].unique()):
    subset = df[df['dataset'] == dataset]
    correlation, mean_x, mean_y, std_x, std_y = calculate_stats(subset)
    row,col = (i // 2) + 1, (i % 2) + 1
    
    # Add scatter plot
    fig.add_trace(go.Scatter(x=subset['x'], y=subset['y'],
                  mode='markers', marker=dict(size=10), 
                  name=f"Dataset {dataset}"), row=row, col=col)
    
    # Add regression line
    slope, intercept = np.polyfit(subset['x'], subset['y'], 1)
    fig.add_trace(go.Scatter(x=subset['x'], y=slope * subset['x'] + intercept,
                  mode='lines', name=f"Fit {dataset}",
                  line=dict(color='red')), row=row, col=col)

    # Add a separate trace for the annotation in the bottom right corner
    stats_annotation = (f"Corr: {correlation:.2f}<br>"f"Mean(x): {mean_x:.2f}<br>"
                        f"Mean(y): {mean_y:.2f}<br>"f"Std(x): {std_x:.2f}<br>"f"Std(y): {std_y:.2f}")
    
    fig.add_trace(go.Scatter(x=[max(subset['x']) - 1.2],  
                             y=[min(subset['y']) + 1],  
                             mode='text', text=[stats_annotation],
                             showlegend=False), row=row, col=col)

fig.update_layout(height=600, width=800, title_text="Anscombe's Quartet with Correlation, Mean, and Std Dev", 
                  margin=dict(l=50, r=50, t=50, b=50))
fig.show()
```

```python
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load Galton's parent-child height dataset
galton_data = sm.datasets.get_rdataset("GaltonFamilies", "HistData").data

# Select midparentHeight and childHeight for classic Galton analysis
Y = galton_data['midparentHeight']
x = galton_data['childHeight']
# Usually x "predicts" Y but it's here backwards for aspect ratio purposes:
# an ellipse shape is actually way more fkn weird to look at (for a human) than you would first think
#x = galton_data['midparentHeight']
#Y = galton_data['childHeight']

# Create a scatter plot with alpha transparency
fig = px.scatter(galton_data, x='childHeight', y='midparentHeight', 
                 title='Midparent vs Child Height')
fig.update_traces(marker=dict(size=8, opacity=0.5))

# THE DETAILS OF THE FOLLOWING ARE OUT OF SCOPE
# - multi/bivariate normal distribution and their covariance matrices
# - ellipses and their math and visual weirdness outside of a 1:1 aspect ratio
# - eigenvectors and eigenvalues and major axis lines, etc. etc.
# ALL OF THESE ARE way BEYOND THE SCOPE OF STA130:
# They're just here so that I can have some pictures for illustration

# Function to calculate the ellipse points (for the bivariate normal ellipse)
def get_ellipse(mean, cov, n_std=1.0, num_points=100):
    """Generate coordinates for a 2D ellipse based on covariance matrix."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([np.cos(theta), np.sin(theta)])  # unit circle

    # Ellipse transformation: scale by sqrt(eigenvalues) and rotate by eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    ellipse_coords = np.dot(eigvecs, np.sqrt(eigvals)[:, np.newaxis] * circle * n_std)

    # Shift ellipse to the mean
    ellipse_coords[0] += mean[0]
    ellipse_coords[1] += mean[1]

    return ellipse_coords, eigvecs

# Calculate covariance matrix and mean
cov_matrix = np.cov(x, Y)
mean_vals = [np.mean(x), np.mean(Y)]

# Get ellipse coordinates and eigenvectors
ellipse_coords, eigvecs = get_ellipse(mean_vals, cov_matrix, n_std=2)
ellipse_x, ellipse_Y = ellipse_coords

# Get the first eigenvector (for the primary direction)
primary_direction = eigvecs[:, 1]  # First eigenvector (primary direction)

# Plot the ellipse
ellipse_trace = go.Scatter(x=ellipse_x, y=ellipse_Y, mode='lines',
                           line=dict(color='green', width=2, dash='dash'),
                           name='American Football Shape')
fig.add_trace(ellipse_trace)

# THE DETAILS OF THE FOLLOWING ARE OUT OF SCOPE
# - multi/bivariate normal distribution and their covariance matrices
# - ellipses and their math and visual weirdness outside of a 1:1 aspect ratio
# - eigenvectors and eigenvalues and major axis lines, etc. etc.
# ALL OF THESE ARE way BEYOND THE SCOPE OF STA130:
# They're just here so that I can have some pictures for illustration

# Add correlation annotation
correlation = np.corrcoef(x, Y)[0, 1]
annotations = [dict(x=min(x), y=max(Y), xanchor='left', yanchor='top',
                    showarrow=False, text=f'Correlation: {correlation:.2f}', 
                    font=dict(color='black'), bgcolor='white')]

# Update layout with corrected annotations
fig.update_layout(title="Galton's Midparent vs Child Height:<br>Regression to the mean IS NOT the Primary Direction",
                  xaxis_title="Child Height", yaxis_title="Midparent Height", annotations=annotations)

# Set square aspect ratio for the axes
fig.update_xaxes(scaleanchor="y")  # X-axis is anchored to Y-axis
fig.update_yaxes(constrain="domain")  # Constrain the Y-axis to the domain of the plot
#fig.update_xaxes(range=[55, 80])  # Fixed x limits
fig.update_layout(height=400, width=800)
fig.show()
```

### Correlation is just for Y = mx + b

Hopefully the equation of a line $Y = mx + b$ doesn't cause too much discomfort for anyone here. Students should know $m$ in the classic "rise over run" manner, whereby the "run" change in $x$ results in a "rise" in $Y$ of $mx$  $(m \times x)$. And the offset $b$ just controls the global shift of the line. There's really just not too much to say here. This is about as easy as it gets.

What can be pointed out here is that when `trendline='ols'` is added to Galton's parent-child heights data cloud, there is certainly going to be a "straight line" **linear association** in the data that's given by $Y = mx + b$. And remember, such a $Y = mx + b$ situation *is required* in order for the numeric correlation value to have a reliably meaningful interpretation. So, the takeaway here is that $Y = mx + b$ and correlation must go hand in hand.  

```python
# You can see this if you replace `fig = px.scatter(...` above with this
fig = px.scatter(galton_data, x='childHeight', y='midparentHeight', 
                 trendline='ols',  # Add a linear trendline
                 title='Midparent vs Child Height with Trendline')

# And the line in this case happens to be the following
# which you can confirm by including or not include the code below
m = 0.1616177526315975  # Slope
b = 58.4194455765732    # Intercept
trendline_x = np.array([55,79])
trendline_Y = b + m*np.array([55,79])
fig.add_trace(go.Scatter(x=trendline_x, y=trendline_Y, mode='lines',
                         line=dict(color='purple', width=2),
                         name='yhat = 0.16 + 58.41x'))
```

### Simple Linear Regression is Just a Normal Distribution

**Simple Linear Regression** is just a normal distribution. It so happens to have a mean location parameter controlled by a $Y = mx + b$ "straight line" **linear equation** which we'll now pretentiously write as $\beta_0 + \beta_1 x_i$. But we can therefore just write out what entire **simple linear regression** model actually is, $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$, which is frankly just really much more complicated than the $Y = mx + b$ equation for a line. The only extension is that $\epsilon_i$ is now a **random variable** drawn from the **normal distribution** $N\left(0, \sigma\right)$ which we write as $\epsilon_i \sim \mathcal N\left(0, \sigma\right)$. Otherwise, again, these's just basically nothing to this. It's super simple. Maybe that's why it's called **Simple Linear Regression**?

$$ 
\Large
\begin{align}
Y_i \sim{}& \mathcal{N}\left( \beta_0 + \beta_1 x_i, \sigma\right) && \longleftarrow\text{here we think of $Y_i$ as the random variable}\\
Y_i ={}& \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right) && \longleftarrow\text{here we think of $\epsilon_i$ as the random variable which implies $Y_i$}
\end{align}
$$

To **simulate** data from the **Simple Linear Regression**, and see what **correlation** it induces, you can use the following code. 

```python
df = pd.DataFrame({'x': x, 'Y': Y})

# Calculate the correlation
correlation = np.corrcoef(df['x'], df['Y'])[0,1]

fig = px.scatter(df, x='x', y='Y', 
                 trendline='ols',  # Add a linear trendline
                 trendline_color_override='red',
                 title='Y vs x')
fig.update_traces(marker=dict(size=8, opacity=0.5))

fig.add_annotation(text=f'Correlation: {correlation:.2f}', 
                   xref='paper', yref='paper', 
                   x=0.05, y=0.95, showarrow=False,
                   font=dict(size=12, color='black'), bgcolor='white')
fig.show()
```

Here are a couple example specifications that could be used for the **simulation**. Feel free to play around with the different choices of the parameters a bit to get a sense of how this model can be used to **simulate** different **relationships** and **correlations** between $x$ and $Y$.

```python
# Version 1
n = 25
# Arbitrarily define x and then genrate Y
x = stats.uniform(10, 10).rvs(n_points)
Y = 0 + x + stats.norm(loc=0, scale=10).rvs(size=n)

# Version 2
n = 934
# Arbitrarily define x and then genrate Y
x = galton_data['Parent Mid Height']
# x = stats.norm(loc=x.mean(), scale=x.std()).rvs(size=n)
beta0 = -100 # galton_data['Child Height'].mean()
beta1 = 2
Y = beta0 + beta1*x + stats.norm(loc=0, scale=3).rvs(size=n)
```

Did you figure out how you can control **correlation** with the **simple linear model**? Does it depend on the line $Y=mx+b$? Or what about $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$ and $\sigma$? Is that what matters for **correlation**? 

#### Terminology: predictor, outcome, intercept and slope coefficients, and error terms

Here's some terminology that you need to know to talk about **Simple Linear Regression**.

- $Y_i$ is the so-called **outcome**
- $x_i$ is then the **predictor**
- $\beta_0$ is the **intercept coefficient**
- $\beta_1$ is the **slope coefficient**
- $\epsilon_i$ is called the **error term**

That's it. 

> But here's a little more commentary on these. 
>
> - **Outcome** $Y_i$ is a **continuous numeric variable**
> 
>   **Outcome** $Y_i$ can also be called a **response**, **dependent**, or **endogenous variable** in some domains and contexts
> 
> - **Predictor variable** $x_i$ is a **numeric variable**
> 
>   Fow now we'll consider $x_i$ to be a **continuous** numeric variable, but this is not necessary, and we will consider versions of $x_i$ later
> 
> - **Predictor variable** $x_i$ can also be called an **explanatory**, **independent**, or **exogenous variable**, or a **covariate** or **feature** (which are the preferred terms in the statistics and machine learning domains, respectively)
> 
> - **Intercept** $\beta_0$ and **slope** $\beta_1$ **coefficients** are the two primary **parameters** of a **Simple Linear Regression** model
> 
>   **Intercept** and **slope** describe a **linear** ("straigh line") relationship between **outcome** $Y_i$ and **predictor variable** $x_i$
> 
> - **Error** $\epsilon_i$ (also sometimes called the **noise**) makes **Simple Linear Regression** a **statistical model** by introducing a **random variable** with a **distribution**
> 
> - The $\sigma$ **parameter** is a part of the **noise distribution** and controls how much vertical variability/spread there is in the $Y_i$ data off of the line: $\sigma$ is an "auxiliary" **parameter** in the sense that there is usually more interest in $\beta_0$ and $\beta_1$ than $\sigma$
> 
> - **Errors** $\epsilon_i$ (in conjuction with the **linear form**) define the **assumptions** of the **Simple Linear regression** Model specification
>
>   <u>but these **assumptions** are not the focus of further detailed reviewed here</u>
> 
> > Further details regarding the assumptions are as follows, but they are not the focus are not the focus of further detailed reviewed now.
> >
> > The first four assumptions associated with the **Simple Linear regression** model are that
> > 
> > 1. the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
> > 2. the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma$ does not change as a function of $x_i$)
> > 3. the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)
> > 4. and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
> >
> > and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that
> >
> > 5. the $x_i$ **predictor variable** is **measured without error**
> > 6. and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)


## TUT/HW Topics


### _statsmodels_

The `Python` modules providing analogous functionality to "base" `R` statistical programming language are the `scipy.stats` and `statsmodels` modules. The `scipy.stats` module provides functionality related to statistical **distributions** (which we've previously started exploring), but it also provides some **hypothesis testing** capabilities (which we've not examined since we've instead been exploring this topic through **simulation**). The `statsmodels` module, on the other hand, combines the **distributional** and **hypothesis testing** concepts of the `scipy.stats` module to allow us to "fit" **linear regression** models and examine various aspects of the **fitted models**, including analyzing them using **hypothesis testing**.  

The `statsmodels` module is well-regarded for its comprehensive range of tools for statistical modeling and **hypothesis testing** which help us evaluate and understanding evidence about relationships between variables in many fields, such as economics, biology, and social sciences. While there are a number of ways to use the `statsmodels` module, for STA130 we'll always its "formula" version.

```python
import statsmodels.formula.api as smf
```


### _smf.ols_ 

The "ols" in `smf.ols()` stands for "ordinary least squares". The **ordinary least squares** methodology will be explored further in HW7ate9, but suffice it to say for now that it is "how linear regression models are fit to data" in STA130. To specify a **simple linear regression** model in the context of a `pandas DataFrame object` dataset `df` where the ("**independent**" **predictor**) column `x` variable is used to predict the ("**dependent**" **outcome**) column `y` variable, use the "formula" `"y ~ x"` which corresponds to 

$$Y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i , \sigma) \quad \text{ or } \quad  Y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad \text{ where } \quad \epsilon_i \sim \mathcal{N}(0, \sigma)$$

```python
linear_specification = "y ~ x"  # automatically assumes the "Intercept" beta0 will be included
model_data_specification = smf.ols(linear_specification, data=df)
```

In the example above `x` and `y` would be names of columns in `df`. If `x` and `y` in `df` respectively corresponded to "parents average height" and "child's height" (as in Francis Galton's [classical study](https://en.wikipedia.org/wiki/Regression_toward_the_mean#Discovery) which popularized the concept of "regression to the mean" which all **regression** methodologies are now named after), then `model_data_specification` indicates using "parents average height" (`x`) to predict "child's height" (`y`). If rather than `x` and `y` the column names were just `parent_height` and `child_heigh"` directly, then we would instead just use 

```python
import pandas as pd
import statsmodels.formula.api as smf

# Simplified version of Galton's data: this dataset is small and simplified for illustrative purposes
# The actual dataset used by Galton was much larger and included more precise measurements and other variables; however,
# this should give you a sense of the type of data Galton worked with
francis_galton_like_data = {
    'parent_height': [70.5, 68.0, 65.5, 64.0, 63.5, 66.5, 67.0, 67.5, 68.0, 70.0,
                      71.5, 69.5, 64.5, 67.0, 68.5, 69.0, 66.0, 65.0, 67.5, 64.0],
    'child_height': [68.0, 66.5, 65.0, 64.0, 63.0, 65.5, 67.0, 67.5, 68.0, 70.0,
                     70.5, 69.5, 63.5, 66.0, 68.5, 69.0, 66.0, 64.5, 67.5, 63.5]
}
francis_galton_df = pd.DataFrame(francis_galton_like_data)

linear_specification = "child_height ~ parent_height"  # not `linear_specification = "y~x"`
model_data_specification = smf.ols(linear_specification, data=francis_galton_df)
```

#### R-Style formulas I

The `linear_specification` is a so-called "R-style formula" which provides a simple way to specify the **linear form** of a **regression** model. A "formula" of the form `"y ~ x"` automatically assumes the "Intercept" $\beta_0$ will be included in the model specification, with **outcome** "y" on the left and **predictors** "x" on the right, as described in the previous section. The "outcome on the left and predictors on the right" formulation becomes increasingly intuitive in the context of **multiple** (as opposed to **simple**) **linear regression**. The `"child_height ~ parent_height"` is a **simple linear regression** specification because there is a single **predictor** variable; whereas, `"child_height ~ parent_height + nationality"` is a **multiple linear regression** specification because there is more than one **predictor** variable. We will return to **multiple linear regression** in Week 07, so consideration of this topic can be safely postponed for now.

#### Quoting

If the columns referenced by the **linear form** defined in `linear_specification` have "spaces" or "special characters" (like an apostrophe) then a special "quote" `Q("...")` or `` syntax is required to reference them.  It might be easier and better for readability to just use `df.rename(...)` instead to change the name in these situations...

```python
linear_specification = 'Q("child\'s height") ~ Q("parents average height")' # or
linear_specification = "`child\'s height` ~ `parents average height`"
```


### Fitting Models 

After specifying the **linear form** and the dataset of a **regression model** with `smf.ols(...)` the model is then **estimated** or "fit" using the `smf.ols(...).fit()` **method** which provides the **fitted model** (**estimate** of the **theoretical model**) from the dataset 

$$\hat y_i = \hat \beta_0 + \hat \beta_1 x_i \quad \textrm{ (in the case of simple linear regression)}$$

```python
data_fitted_model = model_data_specification.fit()  # estimate model coefficients and perform related calculations
data_fitted_model.params  # the estimated model coefficients beta0 and beta1 (in the case of simple linear regression)
```

For `linear_specification = "child_height ~ parent_height"` 

- The rows of `data_fitted_model.params` will be `Intercept` ($\hat \beta_0$) and `parent_height` ($\hat \beta_1$)
- The `Intercept` $\hat \beta_0$ is automatically assumed to be a part of the **linear form** specification, and the name of the **fitted slope coefficient** $\hat \beta_1$ will match that of the "**independent**" **predictor** variable used in the model
    - The **fitted slope coefficient** $\hat \beta_1$ is an **estimate** of the average change in the "**dependent**" **outcome** variable for a "one unit" increase of the "**independent**" **predictor** variables
    - The **fitted intercept coefficient** $\hat \beta_0$ must then be the **estimate** of the average of the "**dependent**" **outcome** variable when the "**independent**" **predictor** variable has the value $0$
        - $\hat \beta_0$ may sometimes have a sensible interpretation, but often (as in the case of `parent_height` for which the "average parents height of $0$" could never happen) it doesn't really correspond to anything meaningful in the real world
        - $\hat \beta_1$ is really what captures and represents the best estimate of the **linear relationship** between `x` and `y` based on the data you have since this **estimates** the "average change in the **outcome** for a one-unit increase in the **predictor**"
- The **fitted model** is the "straight line" that (with respect to **ordinary least squares** methodology) "best" reflects the **linear association** observed in the dataset between the "**independent**" **predictor** and "**dependent**" **outcome** variables corresponding to the hypothesized **linear association** of the **theoretical model**

> The **fitted** [(simple) linear regression] model $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ **estimates** the **theoretical** [(simple) linear regression] model
>
> $$Y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i , \sigma) \quad \text{ or } \quad Y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad \text{ where } \quad \epsilon_i \sim \mathcal{N}(0, \sigma)$$
>
> If the **theoretical model**  was "true" with $\beta_1 = 2$, then an estimated coefficient of $\hat \beta = 1.86$ would be the result of sample to sample variability. A different sample might produce an estimated coefficient of $\hat \beta = 2.18$. In practice, $\hat \beta_1$ is used to make inferences about $\beta_1$.

The **estimated** $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ line specified by `data_fitted_model.params` can be visualized using `px.scatter(..., trendline="ols"`) since "ols" indicates the same **ordinary least squares** fit of the data used by `smf.ols(...).fit()`. 

```python
import plotly.express as px
# Create the scatter plot with the OLS fit line
px.scatter(francis_galton_df, x='parent_height', y='child_height', 
           title="Simple Linear Regression", trendline="ols")
```


#### _.fittedvalues_ 

The **fitted model coefficients** from the `.params` **attribute** specifying the $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ formula define the **fitted values** (and other **predictions**) of the **fitted model**, but the **fitted values** are also immediately available from the `.fittedvalues` **attribute**.  

```python
data_fitted_model.fittedvalues  # model "predicted" *fitted values*, which for the current example is equivalent to 
# y_hat = data_fitted_model.params[0] + data_fitted_model.params[1] * francis_galton_df['parent_height']
```

The $\hat y_i$ are the **fitted model predictions** of the "expected average value" of $y_i$ for a given value of $x_i$. In the case of the **fitted values**, the "expected average value" $\hat y_i$ "predictions" are both "based on the observed $y_i$" and "made for the observed $y_i$". This is why "predictions" has been put in quotes with respect to **fitted values**. The **fitted values** are not really proper **predictions** since they are "predictions" of the same $(y_i, x_i)$ pairs that are used to **estimate** the **fitted model** $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ which defines the **predictions** in the first place. 

> The notion that **fitted values** uses $(y_i, x_i)$ to make their own "predictions" $(\hat y_i, x_i)$ raises the topic of **in sample** versus **out of sample prediction**.  We will return to this topic and the use of the `.predict()` **method** (as opposed to the `.fittedvalues` **attribute** in the context of **multiple linear regression** in Week 07.


#### _.rsquared_ "variation proportion explained"

The **squared correlation** between the "predicted" **fitted values** $\hat y_i$ and observed "**dependent**" **outcomes** $y_i$ is referred to as **R-squared**, and can be accessed through the `.rsquared` **attribute** of a **fitted model**. While the mathematical details are beyond the scope of STA130, the **R-squared** measures "the proportion of variation explained by the model". So, e.g., an R-squared value of 0.7 means that 70% of the "variation" in the "**dependent**" **outcome**  variables $y_i$ is "explained" by the model. 

```python
import numpy as np

# This measures "the proportion of variation explained by the model"
data_fitted_model.rsquared  # model **R-squared**, which for the current example is equivalent to 
# np.corr_coef(data_fitted_model.fittedvalues, francis_galton_df['child_height'])[0,1]**2
```


#### _.resid_ residuals and assumption diagnostics

Analogously to **fitted values**, the **residuals** of the **fitted model** $\textrm{e}_i = \hat \epsilon_i = y_i - \hat y_i$ can be accessed through the `.resid` **attribute**.

```python
data_fitted_model.resid
```

The **residuals** of the **fitted model** $\textrm{e}_i = \hat \epsilon_i = y_i - \hat y_i$ **estimate** the **error terms** of the **theoretical model** $\epsilon_i$. As such, **residuals** can be used as **diagnostically** to assess **linear regression model assumptions**, such as **normality**, **heteroskedasticity**, and **linearity**. 


```python
# If *residuals* do not appear approximately normally distributed, the "normality assumption" is implausible
residual_normality = px.histogram(data_fitted_model.resid, title="Histogram of Residuals")
residual_normality.update_layout(xaxis_title="Does this appear normally distributed?", yaxis_title="Count")
residual_normality.show()
```

```python
# If *residuals* heights appear to systematically change over the range of y-hat, the "heteroskedasticity assumption" is implausible
residual_heteroskedasticity = px.scatter(x=data_fitted_model.fittedvalues, y=data_fitted_model.resid, 
                                         title="Does this systematically change over y-hat?")
residual_heteroskedasticity.update_layout(xaxis_title="Fitted Values", yaxis_title="Residuals")
residual_heteroskedasticity.add_hline(y=0, line_dash="dash", line_color="red")
residual_heteroskedasticity.show()
```

```python
# If y versus y-hat does not follow the "linearity" of the "y=x" line, the "linearity assumption" is implausible
fitted_values = data_fitted_model.fittedvalues
residual_linearity = px.scatter(x=fitted_values, y=francis_galton_df['child_height'], 
                                title="Is this relationship 'linear' in a 'y=x' way?")
residual_linearity.update_layout(xaxis_title="Fitted Values", yaxis_title="Outcome")
residual_linearity.add_scatter(x=[min(fitted_values), max(fitted_values)], y=[min(fitted_values), max(fitted_values)],
                               mode='lines', line=dict(color='red', dash='dash'), name="The 'y=x' line")
residual_linearity.show()
```


### Testing "On Average" Linear Association

The information needed for statistical **hypothesis testing** analysis of the **fitted model** is accessed using the `.summary()` **method**.

```python
data_fitted_model.summary()  # generates a detailed report of statistical estimates and related information; but, ... 
data_fitted_model.summary().tables[1]  # what we need for *hypothesis testing* is contained in its `.tables[1]` attribute
```

The `coef` column of `data_fitted_model.summary().tables[1]` corresponds to `data_fitted_model.params` discussed above; but, the additional `P>|t|` column includes the **p-values** associated with a **hypothesis test** for each of the **coefficients**.
- Specifically, the values in `P>|t|` are **p-values** for a **null hypothesis** that the **true coefficient value** is zero.
- In the case of the **true slope coefficient** $\beta_1$ representing the "average change in the **outcome** for a one-unit increase in the **predictor**" 
    - this **null hypothesis** corresponds to the assumption of "**no linear association** between the **outcome** and **predictor variables** 'on average'".
- Formally, this "no linear relationship" **null hypothesis** would then be stated as $H_0: \beta_1 = 0$ where $\beta_1$ is the hypothesized **parameter** value which the associated **statistic** $\hat \beta_1$ **estimates**
- And rarely would there be interested in $H_0: \beta_0 = 0$ since an **intercept** value of $0$ does not generally have a meaningful interpretation, and the context of **linear regression** is typically just concerned with evaluating the relationship between the **outcome** and **predictor variables**.

> Recalling the definition of a **p-value**, in the current context a **p-value** would be "the probability of that a dataset would result in an **estimated coefficient** as or more extreme than the **observed coefficient** if the **null hypothesis** (that the **true coefficient value** is zero) were true". The familiar "as or more extreme" definition of the **p-value** is reflected in the `P>|t|` column name, which can be further seen to indicated a "two-sided" hypothesis test specification. 

A small **p-value** for the **slope coefficient** $\hat \beta_1$ therefore suggests the implausibility of the **null hypothesis** and hence suggests that there actually likely is "a **linear association** 'on average' between the outcome and predictor variables" with $\hat \beta_1$ capturing the "on average" change of this **linear association**. 

> The `std err` and `t` columns relate to the **sampling distribution** "under the null", and the `[0.025 0.975]` columns provide **inference** for **estimated coefficients** in the form of **confidence intervals**, but all of this is beyond the scope of STA130 so we'll just concern ourselves with appropriately using the **p-values** from the `P>|t|` column. 

## LEC Two(2) New Topics

### Two(2) unpaired samples group comparisons

4. one, paired, and two sample tests

So far in the course we've seen **paired samples** turned into **paired sample differences**, and we've seen the **one sample** "coin flipping" null hypothesis. Both of these are really just treated as **single sample analyses**. The following table breaks this down in order to demonstrate the natural generalization of this situation where we ACTUALLY *really have* **two independent samples**.

|                 | Paired Samples  | Unpaired Samples | 
|-----------------|-----------------|------------------|
| One Sample      | Paired Difference (continuous)<br>Paired Comparisons (binary)|Coin Flipping or $\underset{\textrm{another chosen distribution}}{H_0: f{\theta=\theta_0}(X=x)}$ |  
| Two Sample      | Treated as One Sample | NEW CATEGORY TO CONSIDER!!! | 

By **two independent samples** we mean we have **two unpaired samples** representing **two different groups**.
In this context it is natural for interest to focus on potential differences between the two **populations** from which the two **samples** are drawn from. Interestingly, this is not so different from the internet in **paired sample** context. The difference is that there the two groups may be the same individuals before and after some intervention, or some other natural pairing such as twins, husbands and wives, a parent and a child, etc. But again interest in these contexts lies in examining differences or changes between two "populations" in some abstract sense. The real difference then between paired and unpaired samples is that continuous paired difference or binary paired comparisons allow a paired sample context to be treated as in the manner of a single sample analysis.

But this then shows that the real difference in consideration is the difference between **one sample** and **two sample** analysis contexts. But interestingly, there is again a similarity across these two different modes of analysis that is worth emphasizing. Namely, characterizing evidence using **hypothesis testing** and performing statistical inference using **bootstrapped confidence intervals** are (perhaps as should be expected) available for both **one sample** *and* **two sample** data analysis contexts. The next two sections will discuss these new methods, which are respectively (obviously) referred to as **Permutation Testing** (based on label shuffling) and what we'll call just **"Double" Bootstrapping** for the purposes of our course (even though that's not an "official" name). 

|                                       | One Sample                                   | Two Sample                          |
|---------------------------------------|----------------------------------------------|-------------------------------------|
| Hypothesis Testing<br>using p-values  | $H_0$ Coin Flipping<br>Sampling Distribution | Permutation Test<br>label shuffling |
| Confidence Intervals<br>for Inference | Bootstrapping | "Double" Bootstrapping |


### Two(2) unpaired sample permutation tests

The idea of a **permutation test** starts with the **null hypothesis**.

$$H_0: \textrm{There is not difference between these two populations}$$

If this **null hypothesis** *is true*, then "label shuffling" would not actually have any meaning. 
So with respect to $H_0$ each of the follow **permutations** of the actual group label are equally reasonable (since group label *does not matter* if the there's no difference between the two populations). 

| Observation value | Actual group label | Shuffled group label 1 | ... 2 | ... 3 |
|-------------------|--------------------|------------------------|-------|-------|
| 7 | A | A | B | B |
| 4 | A | A | A | B |
| 7 | B | A | A | A |
| 9 | B | B | A | B |
| 1 | A | B | B | A |
| 1 | B | B | B | A |

However, this **label shuffling** then provides a mechanism to **simulate** samples under the assumption that the null hypothesis is true. And of course these can be used to **simulate** the **sampling distributions** of our statistic of interest *under the assumption that the null hypothesis is true*. And then from there estimate the **p-value** of the **actually observed statistic** *relative to this sampling distribution*.

To illustrate this, suppose we're VERY interested in understanding the Pokémon universe, and in particular we REALLY care to learn if "Water" and "Fire" Pokémon represent an apartheid-esque patriarchal tyranny. As we all no doubt are.

```python
import pandas as pd
import plotly.express as px

url = "https://raw.githubusercontent.com/KeithGalli/pandas/master/pokemon_data.csv"
# fail https://github.com/KeithGalli/pandas/blob/master/pokemon_data.csv
pokeaman = pd.read_csv(url)

fire = pokeaman['Type 1'] == 'Fire'
water = pokeaman['Type 1'] == 'Water'
pokeaman[ fire | water ]
```

In this case then, what we're interested in asking is whether or not the difference observed in the box plot below could be just be due to the random chance of the sample of Pokémon that from the Pokémon universe that we happen to know about here on earth thanks to "[t]he Pokémon Company (株式会社ポケモン, [Kabushiki Gaisha](https://en.wikipedia.org/wiki/Kabushiki_Gaisha) Pokemon, TPC)"

> which according to [https://en.wikipedia.org/wiki/The_Pokémon_Company](https://en.wikipedia.org/wiki/The_Pokémon_Company) is 
"a Japanese company responsible for" letting humans here on earth learn about the Pokémon universe thanks to their efforts regarding "[brand management](https://en.wikipedia.org/wiki/Brand_management), production, [publishing](https://en.wikipedia.org/wiki/Publishing), [marketing](https://en.wikipedia.org/wiki/Marketing), and [licensing](https://en.wikipedia.org/wiki/License) of the [Pokémon](https://en.wikipedia.org/wiki/Pok%C3%A9mon) [franchise](https://en.wikipedia.org/wiki/Media_franchise), which consists of [video games](https://en.wikipedia.org/wiki/Pok%C3%A9mon_(video_game_series)), a [trading card game](https://en.wikipedia.org/wiki/Pok%C3%A9mon_Trading_Card_Game), [anime television series](https://en.wikipedia.org/wiki/Pok%C3%A9mon_(TV_series)), [films](https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon_films), [manga](https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon_manga), [home entertainment](https://en.wikipedia.org/wiki/Home_video) products, merchandise, and other ventures."

```python
display(pd.DataFrame(pokeaman[ fire | water ].groupby('Type 1')['Sp. Atk'].mean()))
display(pd.DataFrame(pokeaman[ fire | water ].groupby('Type 1')['Sp. Atk'].mean().diff()))
print(pokeaman[ fire | water ].groupby('Type 1')['Sp. Atk'].mean().diff().values[1])

fig = px.box(pokeaman[ fire | water ], x="Type 1", y="Sp. Atk", 
    title="Distribution of 'Sp. Atk' between Water and Fire Type Pokémon: Are These Different??")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

# #https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
# import plotly.offline as pyo
# # Set notebook mode to work in offline
# pyo.init_notebook_mode()
```

So, indeed, how *DO* we determine if the averages difference observed in the above box plots is likely to reflect ACTUAL 
Pokémon racism? Or if it might just reflect the random of chance of the adventures of Ash Ketchum, Pikachu, Serena, Misty, Togepi, and Brock? 

> That group DOES *drink a lot* (it is known) and don't really seem to actually ever walk very straight towards any sensible objective of tryin' to get somewhere (or anywhere for that matter) in the show, just sayin'. And I mean, getting "random battles" in the game is, indeed, as one would say, *pretty random*.

Well, we do a bunch of **label shuffling** to **simulate** samples under the assumption that the null hypothesis is true. Ah, like-a-so,
where from here we're **simulate** the sampling distribution of our sample statistic of interest under the assumption that the null hypothesis (that there is not difference between groups so labels don't actually matter) is true. 

```python
import numpy as np 

groups4racism = pokeaman[ fire | water ].copy()

# Set parameters for bootstrap
n_bootstraps = 1000  # Number of bootstrap samples
sample_size = len(groups4racism)  # Sample size matches the original dataset
label_permutation_mean_differences = np.zeros(n_bootstraps)

for i in range(n_bootstraps):
    groups4racism['Shuffled Pokeaman Race'] = groups4racism['Type 1'].sample(n=sample_size, replace=True).values
    label_permutation_mean_differences[i] = \
        groups4racism.groupby('Shuffled Pokeaman Race')['Sp. Atk'].mean().diff().values[1] 
```

So the above code assumes there's *no racism* the Pokémon universe, and based on this and the label shuffling we can therefore do under this assumption, we can then compute the **p-value** of our **observed sample statistic** (relative to the assumption of this null hypothesis under consideration) in order to complete the **permutation test**.

```python
fig = px.histogram(pd.DataFrame({"label_permutation_mean_differences": label_permutation_mean_differences}), nbins=30,
                                title="Mean Difference Sampling under SHUFFLED Pokeaman Race")

mean_differene_statistic = groups4racism.groupby('Type 1')['Sp. Atk'].mean().diff().values[1]

fig.add_vline(x=mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f".<br><br>Shuffled Statistic >= Observed Statistic: {mean_differene_statistic:.2f}",
              annotation_position="top right")
fig.add_vline(x=-mean_differene_statistic, line_dash="dash", line_color="red",
              annotation_text=f"Shuffled Statistic <= Observed Statistic: {-mean_differene_statistic:.2f}<br><br>.",
              annotation_position="top left")
fig.show()  # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

print("p-value",
      (abs(label_permutation_mean_differences) >= abs(mean_differene_statistic)).sum()/n_bootstraps)
```

Hmm... **p-value** is **STRONG** evidence against the null hypothesis according to the table below. 
I'd therefore REJECT the null hypothesis assumption that the (Fire or Water) type of Pokémon doesn't matter when it comes to the "Sp. Atk" power that you're gonna have.  Pokémon def probably racist, y'all. Sorry to burst your bubble.

| p-value                | Evidence                                         |
| ---------------------- | ------------------------------------------------ |
| $$p > 0.1$$            | No evidence against the null hypothesis          |
| $$0.1 \ge p > 0.05$$   | Weak evidence against the null hypothesis        |
| $$0.05 \ge p > 0.01$$  | Moderate evidence against the null hypothesis    |
| $$0.01 \ge p > 0.001$$ | Strong evidence against the null hypothesis      |
| $$0.001 \ge p$$        | Very strong evidence against the null hypothesis |

|![](https://media1.tenor.com/m/IYTiSjV9028AAAAd/squirtle-pokemon-squirtle.gif)|![](https://i.kym-cdn.com/entries/icons/original/000/029/740/detpikachu.jpg)|
|:-:|:-:|
|![](https://static0.gamerantimages.com/wordpress/wp-content/uploads/2020/10/Charmander-meme-pokemon.jpg?q=150&fit=crop&w=300&dpr=1.0) ![](https://imgix.ranker.com/user_node_img/50039/1000767304/original/you-can-t-not-sing-it-photo-u1?auto=format&amp;q=60&amp;fit=crop&amp;fm=pjpg&amp;dpr=1&amp;w=350) |![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuVq8sz6FeiHpsgtbuE7XTuLmREbEPhkt2B9QCT1wXKuVqGCl32ACsW0sETLuhs2VmZKE&usqp=CAU)![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTF4Nfz0ZE-1P6c7BQg4aoDOmSpPoSfsfkdUzFoIVaeQAv5d6BPdKlSX0VCH7KvsZWtQxE)![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwVlYHUe6hgL2fml88yWzNk39N84G2nGFr6A)![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRv0hd0FGzE3rpCD--YBg9nRJv8W932asQHIZJcaRJHRUSxRvwOixg4jnsMvtqxQOg6W7U&usqp=CAU) |


### Two(2) unpaired sample bootstrapping

As always, making a decision regarding a null hypothesis based on the strength of evidence executes the scientific process that hypothesis testing is. But it doesn't tell us the full story. We get that by considering **statistical inference**.  And as you will hopefully recall, the way we can provide **statistical inference** is through **confidence intervals**.  And of course the way we create **confidence intervals** in our class is through **bootstrapping**. The trick to **confidence intervals** in the **two sample** context is to simply perform so-called **"double" bootstrapping**. Now don't get confused, **"double" bootstrapping** IS NOT "double `for` loops. Actually, **"double" bootstrapping** is just a name that got *made up* (literally yesterday) because it describes how **bootstrapping** works in the **two independent sample** context. Namely, **EACH SAMPLE** is **bootstrapped** separately side-by-side. So, **bootstrapping** for the **two independent sample** context is thus **"double" bootstrapping**.  The code below illustrates exactly what is meant by this. 

```python
within_group_bootstrapped_mean_differences = np.zeros(n_bootstraps)
for i in range(n_bootstraps):
    double_bootstrap = \
        groups4racism.groupby("Type 1")[["Type 1","Sp. Atk"]].sample(frac=1, replace=True)
    within_group_bootstrapped_mean_differences[i] = \
        double_bootstrap.groupby('Type 1')["Sp. Atk"].mean().diff().values[1]
    
np.quantile(within_group_bootstrapped_mean_differences, [0.05,0.95])    
```

Are you able to tell what's happening here? First, the original data is separated into its groups by the `groupby`. Then the labels and the data is selected (so all other columns in the data frame are removed). Then, **within each group** each **subgroup sample** is **bootstrapped** in the usual manner. The two sample means and their differences are then computed. This of course is the **mean difference statistic** of interest. And finally, a 90% **"double" bootstrapped confidence interval** is created. If you don't like this **confidence level** screw you. Sue me. Ah but you can't. 'Cuz it's Canada not the US. Too bad so sad sorry I'm not sorry (but VERY Canadian since I'm saying sorry when I don't really mean it). 


### Indicator variables and contrasts linear regression

So, with **permutation testing** and the **"double" bootstrap** methods above we can provide **evidence** *against a null hypothesis* and **inference** *for a population parameter*. And these two methods are quite great. But is there another way, you ask? Why yes, there is. How did you know? Maybe because there's probably actually about a million other ways. But they're all based exactly on the idea of a **sampling distribution of a statistic**. That is, ALL STATISTICAL METHODS ARE BASED ON either (A) the **sampling distribution of a statistic under a null hypothesis** (in order to compute **p-values** and perform **hypothesis** testing) or (B) a **sampling distribution of the sample statistic** (in order to provide **inference** about the corresponding **population parameter**). For the latter task (of **inference**), we'll often rely on the **bootstrapped sampling distribution of the sample statistic** which we've (hopefully) become increasingly familiar and comfortable with through the methods of our course. But, in later statistics courses you'll additionally learn how to create other kinds of **confidence intervals** based on approximations derived from theoretical derivations (based on some assumptions about the data). In fact, the "statistic plus and minus two standard errors" is exactly such a theoretically derived confidence interval approximation (based on some assumptions about the data). This style of "statistic plus and minus two standard errors" confidence interval *is in fact provided in the output of fitted simple linear regression models*. So we'll note that later. But first, to start with, let's begin by considering how to use simple linear regression models for hypothesis testing in the **two independent samples** context. 

The idea that we need to do this is called an **indicator variable**. The **simple linear regression model** we've considered so far is

$$Y_i = \beta_0 + \beta_1 \times x_i + \epsilon_i \quad \textrm{where there is a distributional assumption on the error term } \epsilon_i \textrm{ and } x_i \textrm{ is a continuous valued numeric variable.}$$  

But here's what amounts to being essentially the same model, except a **qualitative categorical** variable $k_i$ is now substituted for $x_i$

$$Y_i = \beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i) + \epsilon_i  \quad \textrm{where $1_{[\text{group}]}(k_i)$ will become the value of $1$ if ``$k_i=\textrm{group}$'' is }\texttt{True}\textrm{, and will otherwise become the value of $0$.}$$

So under this specification, $1_{[\text{group}]}(k_i)$ is a **binary variable**, and it's called an **indicator variable**. It's called an **indicator variable** because it *indicates* whether or not $k_i=\textrm{group}$ is `True`. If $k_i=\textrm{group}$ is `True` then the **binary variable** is $1$, and otherwise it's $0$. Interestingly, this means that the **qualitative categorical** variable $k_i$ might not itself be a **binary variable**!  No matter. The **indicator variable** $1_{[\text{group}]}(k_i)$ *is* a **binary variable**. 

Since $1_{[\text{group}]}(k_i)$ is a binary variable, $\beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i)$ behaves slightly differently than $\beta_0 + \beta_1 \times x_i$. The **continuous variable** $x_i$ can take on values that are not limited to just $1$ or $0$; so, therefore $\beta_1$ in the canonical simple linear regression model specification capture the "rise over run" association observed between $Y_i$ and $x_i$. But this is not the way to think about interpreting things for the **binary variable** version of this specification $\beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i)$. In this **binary variable** version, $\beta_1$ captures the so-called **contrast** relative to $\beta_0$ which occurs whenever the $1_{[\text{group}]}(k_i)$ binary variable is *"turned on"*. But what then does this **contrast** mean? Well, *it defines the* **_difference_** between the two groups defined by $1_{[\text{group}]}(k_i)$. Namely, the group where $k_i=\textrm{group}$ is `True`, and "everybody else" is the "other group" where $k_i=\textrm{group}$ is `False` (or $k_i\neq\textrm{group}$ is `True`). So if $1_{[\text{group}]}(k_i)$ is *"off"* because $k_i\neq\textrm{group}$, then the value of the model is $\beta_0$ (because in this case $1_{[\text{group}]}(k_i)$ is $0$). And if $1_{[\text{group}]}(k_i)$ is *"on"* because $k_i=\textrm{group}$, then the value of the model is $\beta_0+\beta_1$ (because in this case $1_{[\text{group}]}(k_i)$ is $1$). So there are two cases.

0\. If $k_i\neq\textrm{group}$ then $\beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i) = \beta_0 + \beta_1 \times 0 = \beta_0$ 

1\. If $k_i=\textrm{group}$ then $\beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i) = \beta_0 + \beta_1 \times 1 = \beta_0 + \beta_1$ 

So, the **contrast** $\beta_1$ captures *the difference* between the two groups. But now recall what **simple linear regression** provides in terms of **hypothesis testing**.  Namely, it provides **hypothesis testing** for $H_0: \beta_1 = 0$. But now in the context of the **indicator variable** version of simple linear regression, this **null hypothesis** has the interpret ion of meaning "no difference between groups". So let's now examine the output of a **fitted indicator variable version of the simple linear regression model**.

```python
import statsmodels.formula.api as smf

# Model (a): Predict Attack based on Defense
model_fit = smf.ols(formula="Q('Sp. Atk') ~ Q('Type 1')", data=groups4racism).fit()
print(model_fit.rsquared)
model_fit.summary().tables[1]
```

|beta0-hat / beta1-hat  |  coef	        |std err|t	|P>\|t\||[0.025| 0.975]|
|-----------------------|---------------|-------|-------|-------|------|-------|
|Intercept	        |88.9808	|4.070	|21.860	|0.000	|80.943	|97.019|
|Q('Type 1')[T.Water]	|-14.1683	|4.926	|-2.876	|0.005	|-23.895|-4.442|

From this output we see that the "group" when $1_{[\text{group}]}(k_i)$ is `Water`.
So the other group is `Fire` (just because that's the only other kind of pokeamans in this data. 
So what is $\hat \beta_0$ then? Well that's the `Intercept` so $\hat \beta_0 \approx 89$.
And so that's then the average of the out come (`Sp. Atk`) in the `Fire` group.
So then $\hat \beta_1 \approx -14.16$ which is the **contrast** (difference) from the average of the `Fire` group to the average of the `Water` group (which is $\hat \beta_0 + \hat \beta_1 \approx 89 -14.16 = 74.84$.

And what else can we say here? We have a **p-value** relative to $H_0: \beta_1 = 0$ of `0.005`.
Again this indicates **strong** evidence against the null hypothesis.  

This should by now feel very reminiscent of what **permutation testing** we previously considered. Indeed, what we have here is a **theoretical p-value** *based on an approximation derived on the basis of* **the assumptions of the regression model** (which are encoded in the the distributional assumption that **error terms** $\epsilon_i \sim \mathcal{N}(0, \sigma)$ and the statement of the **linear form**, here 
$Y_i = \beta_0 + \beta_1 \times 1_{[\text{group}]}(k_i) + \epsilon_i$ but perhaps more generally simply stated as $Y_i = \beta_0 + \beta_1 \times x_i + \epsilon_i$. The previously considered **permutation test** in contrast is a **nonparametric hypothesis test** which does not rely on the so-called **parametric** *normality assumption*. 

*But now remember* that ALL STATISTICAL METHODS ARE BASED ON either (A) the **sampling distribution of a statistic under a null hypothesis** (in order to compute **p-values** and perform **hypothesis** testing) or (B) a **sampling distribution of the sample statistic** (in order to provide **inference** about the corresponding **population parameter**). 
Regarding (A) the definition of the **p-value** used to give evidence against $H_0: \beta_1 = 0$ through the **indicator variable formulation** of the simple linear regression specification is the same as ever. And this is clearly reflected in the labeling of the **p-value** `P>|t|` as given in the output table of the fitted model. 

- A **p-value** is `P>|t|` *the probability of seeing a statistic as or more extreme that what was observed if the null hypothesis was true.*

The difference is that the **permutation test** uses "label shuffling" to **simulate** the **sampling distribution** of *the sample statistic under the assumption of the null hypothesis (of not difference between the groups)*, and computes a **p-value** from there; whereas, the **indicator variable formulation** of the simple linear regression specification instead *assumes the assumption regarding* the **_normality of the error terms_**, etc. is correct, which allows for *a theoretical approximation* of the **sampling distribution** of $\hat \beta_1$ *under the null hypothesis assumption* $H_0: \beta_1 = 0$. 

- The assumption regarding the **_normality of the error terms_**, etc. can diagnostically assessed by examining the **distribution of the residuals** $\hat \epsilon_i = Y_i - \hat y_i$; specifically, the appropriateness  of the claim that **this distribution appears to reasonably be characterized as being _approximately_ normally distribute.**
- Would you be able to assess this assumption in the context of the current analysis? If so, what is your opinion of this assumption in the context of the current analysis? Does it appear to be a reasonable assumption? Or does it appear to be egregiously unlikely and misguided? 

To conclude on a high note here, we've seen that we can do hypothesis testing examining group difference using **simple linear regression** on the basis of **indicator variables**.  But **inference** using **confidence intervals** WILL ALWAY BE PREFERABLE TO HYPOTHESIS TESTING FRAMEWORKS since it can itself be used to make decisions with associated **confidence levels** just as **classical hypothesis testing** does, AND IT ALSO additionally provides a plausible range of values (where plausible is defined in terms the **confidence level** corresponding to the **confidence interval**) for what the actual true value of the parameter of interest might be according to the evidence available in the data. This allows us to meaningful understand and interpret what it exactly is that we understand about the population on the basis of the sample data at hand. This kind of **inference** is so, so much more usefully actionable than simply characterizing the evidence against the null (but which of course **confidence intervals** CAN STILL EVEN DO). So now you're asking, "okay, so what's this have to do with indicator version of simple linear regression that we're considering here?".  Well let me tell you. What do you think the columns `[0.025` and `0.975]` mean in the output table of a fitted linear regression model? 

I rest my case. 

So, bottom line: pokeaman Pokémon is probably maybe racist no doubt. **Strong** evidence to believe that. Trust. 