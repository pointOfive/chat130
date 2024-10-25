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

1. indicator variables
2. two sample group comparisons
3. normality assumption diagnostic
4. one, paired, and two sample tests
4. two sample permutation tests
5. two sample bootstrapping


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

Did you figure out how you can control **correlation** with the **simple linear model**? Does it depend on the line $Y=mx+b$? Or what about $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$ and $\sigma$ is that matters for **correlation**? 

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