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

> The `std err` and `t` columns relate to the **sampling distribution** "under the null", and the `[0.025 0.975]` columns provide **inference** for **estimated coefficients** in the form of **confidence intervals**, but all of this is beyond the scope of STA130 so we'll just concern ourselves with appropriately using the **p-values** from the `P>|t|` column. # STA130 LEC 07 (Oct 21)

## .THE STA130 COURSE PROJECT.<br>...HAS ENTERED THE BUILDING...

### 9:10/1:10 A word from the CO (Commanding Officer)<br><sub>My Dad's USAF Pentagon so I love my weird acronyms like this one</sub>
    
CO: **Christine Ovcaric**  
Program Manager, The Sandbox  
Uoft Experiential Learning Hub

The **Community Engaged Learning (CEL)** Project
- **Dr. Heather Hermant** (Community Partnerships)
- **Dr. Michelle Arnot** (Pharma/Toxicology)
- **Dr. Ashley Waggoner Denton** (Psych)
- **Dr. Naomi Levy-Strumpf** (Human Biology)

### 9:20/1:20 A fireside chat with Dr. Kiffer Card

- the Scientific Director of the Canadian Alliance for Social Connection and Health
- President and Chair for the Mental Health and Climate Change Alliance
- President of the Island Sexual Health Community Health Centre
- Director of Research for GenWell
- Assistant Professor with the Faculty of Health Sciences at Simon Fraser University
- yada yada yada blah blah blah etc. etc. etc. Dr. KC **JUST DO IT** if you see what I mean

### 9:45/1:45

- Sarah Shafi
- Jazmyn Crasto
- Samantha Rahamatali

From HMB301 **Biotechnology** with **Dr. Naomi Levy-Strumpf**

> Students gain an appreciation for how science, government and society drive the development of biotechnology products. Topics include emerging immunotherapies, “living therapeutics”, emerging challenges, CRISPR-based therapeutics, emerging diagnostics, and stem cells and regenerative medicine.




```python
from IPython.display import YouTubeVideo
YouTubeVideo('rmuRoAf9-bo', width=800, height=500)
```

## Wait, Linear Regression is just a--?<br>YES -- A Normal Distribution Model

1. ~~Elvis~~ THE STA130 COURSE PROJECT HAS LEFT THE BUILDING

    1. But the game isn't over: https://www.guessthecorrelation.com/
        1. The game is afoot.
        2. No, the game is not literally "a foot" -- that's just a Sherlock the Holmie's quote
        3. **to celebrate how we gonna sleuth out this Canadian Social Connection Survey data**
        4. and because Benedict Cumberbatch be havin' the best quotes: **"Data, data, data. I cannot make BRICKS without CLAY"**
    
|![](https://images6.fanpop.com/image/photos/36500000/Sherlock-Holmes-Sherlock-BBC1-image-sherlock-holmes-sherlock-bbc1-36580721-538-339.jpg)|![](https://i.imgflip.com/97bm6o.jpg)|
|-|-|
| | |
    
2. **Correlation IS NOT Causation?**

    1. Ice Cream does not cause Shark Attacks?
    2. Parents Height does NOT CAUSE Childs Height? 
        1. Does Height CAUSE weight? 
    3. What IS **correlation** anyway? 
    4. There are MANY kinds of associations (besides correlation) that are not actually LINEAR
        1. And if you try to "measure" them with **THE LINEAR ASSOCIATION measure CORRELATION** you CAN FOR SURE get the wrong idea of "what the picture looks like"
    
3. Simple Linear Regression

    1. Outcome $Y_i$
    2. Predictor $x_i$
    3. Intercept $\beta_0$ coefficient
    4. Slope $\beta_1$ coefficient
    5. Error term $\epsilon_i$

### Let's Play

https://www.guessthecorrelation.com/

### Correlation IS NOT Causation

in the since that the following can't be true

### ~~Ice Cream CAUSES Shark Attacks!!!~~

|![](https://i.ibb.co/3fhg46G/dd-6-image1.jpg)|
|-|
|![](https://pbs.twimg.com/media/GDuW8vmXkAArDmt.jpg)|
|![](https://biostatsquid.com/wp-content/uploads/2022/11/Slide21-1.png)|


### Correlation IS NOT Causation

in the sense that there must be other interesting factors at play

### Parents HEIGHTS (alone) DOES NOT CAUSES Child Height!!!

Okay, so first, here's some more of that old timey shit
- Remember, we're themeing hard on the homie Benedict Sherlock



| This is like a "2D histogram" of parents "mid height" versus their adult children's height (with the so-called "marginal" histogram counts there on the right) |
|:-|
|![](https://d3i71xaburhd42.cloudfront.net/562e90c04e43254ab35b7987e0dabd228d040e97/8-Table1-1.png)|

| And these are interesting (EXTREMELY "CLASSIC") figures showing so-called|"Regression to the mean" phenomenon where relationships always "less strong"  |
|-:|:-|
|<img src="https://i.namu.wiki/i/Ujk5wRoPLVwPyTunKcJhuOWJtlRMjNmnvUo97obITxpudQNC_jxI-Gda9foIuVmSXwK77A6GET3ybvN2Cs21FA.webp" width="500"/>|<img src="https://www.cs.bu.edu/fac/snyder/cs132-book/_images/L23LinearModels_6_0.png" width="500"/>|



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
                 trendline='ols',  # Add a linear trendline
                 title='Midparent vs Child Height with Trendline')
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

# Define the line points based on the primary direction
line_length = 4  # Length of the primary direction line
line_x = np.array([mean_vals[0] - line_length * 3 * primary_direction[0], 
                   mean_vals[0] + line_length * 3 * primary_direction[0]])
line_Y = np.array([mean_vals[1] - line_length * 3 * primary_direction[1], 
                   mean_vals[1] + line_length * 3 * primary_direction[1]])

# Plot the ellipse
ellipse_trace = go.Scatter(x=ellipse_x, y=ellipse_Y, mode='lines',
                           line=dict(color='green', width=2, dash='dash'),
                           name='Covariance Ellipse')
fig.add_trace(ellipse_trace)

# Add the primary direction line through the ellipse
primary_direction_line = go.Scatter(x=line_x, y=line_Y, mode='lines',
                                    line=dict(color='red', width=3, dash='dot'),
                                    name='Primary Direction')
fig.add_trace(primary_direction_line)

# Right-align annotations for the regression line
annotations = [dict(x=min(x), y=min(Y)+4, xanchor='left', yanchor='bottom', 
                    showarrow=False, text='Regression to Mean',
                    font=dict(color='blue'), bgcolor='white'),
               dict(x=max(x), y=min(Y), xanchor='right', yanchor='bottom', 
                    showarrow=False, text='"Line of Best Fit"<br>`trendline="ols"`',
                    font=dict(color='blue')),
               dict(x=max(x), y=max(Y)-4, xanchor='right', yanchor='bottom', 
                    showarrow=False, text='Major Axis', font=dict(color='red'))]

# THE DETAILS OF THE FOLLOWING ARE OUT OF SCOPE
# - multi/bivariate normal distribution and their covariance matrices
# - ellipses and their math and visual weirdness outside of a 1:1 aspect ratio
# - eigenvectors and eigenvalues and major axis lines, etc. etc.
# ALL OF THESE ARE way BEYOND THE SCOPE OF STA130:
# They're just here so that I can have some pictures for illustration

# Add correlation annotation
correlation = np.corrcoef(x, Y)[0, 1]
annotations.append(dict(x=min(x), y=max(Y), xanchor='left', yanchor='top',
                   showarrow=False, text=f'Correlation: {correlation:.2f}', 
                   font=dict(color='black'), bgcolor='white'))

# Define your mx + b trendline formula
# For example: y = 0.5x + 2
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(x, Y)
m = 0.1616177526315975  # Slope
b = 58.4194455765732    # Intercept
trendline_x = np.array([55,79])
trendline_Y = b + m*np.array([55,79])
fig.add_trace(go.Scatter(x=trendline_x, y=trendline_Y, mode='lines',
                         line=dict(color='blue', width=2),
                         name='yhat = 0.16 + 58.41x'))

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


```python
# https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```

### So this "Regression to the mean" is from dudess on left (while dudess on the right is Dar for the Win)

More [Queens](https://www.youtube.com/watch?v=-tJYN-eG1zk) Supporting [Queens](https://www.youtube.com/watch?v=hFDcoX7s6rE) Serving Looks 
- Absolutely Fierce and Fabulous Boss Babe SLAY [Queens](https://www.youtube.com/watch?v=fJ9rUzIMcZQ)


|Sir Francis Galton, alleged [child prodigy](https://www.youtube.com/watch?v=rmHDhAohJlQ), confirmed cousin to | Charles Darwin, all up [on the origin](https://en.wikipedia.org/wiki/On_the_Origin_of_Species) of [UrFavNewTerm](https://www.youtube.com/watch?v=JKWCVuWeK-8) "Regression" |
|:-|-:|
|![](https://upload.wikimedia.org/wikipedia/commons/e/ec/Francis_Galton_1850s.jpg)|![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Charles_Darwin_seated_crop.jpg/440px-Charles_Darwin_seated_crop.jpg)|
|This man ^^^^^ INVENTED "REGRESSION to the mean" y'all|This man ^^^^^ appears -- as you can see from the figure above -- to be|
| | EXTREMELEY ENVIOUS of the other man|
| | HAHA I guess he just never did anything meaningful in his life HAHA what a loser| 

### Correlation IS NOT Causation

in the sense that there must OBVIOUSLY be other EASILY identifiable FACTORS at play

### Does Height CAUSE weight?

YOU tell ME

### What IS Correlation anyway?

**Well it's the following which you can try to make sense of later if you want to BUT WHO CARES FOR NOW**

$\LARGE r = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{(n-1) S_x S_Y}$

where

$\LARGE s_x = \sqrt{\frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n - 1}} \quad \text{and} \quad s_Y = \sqrt{\frac{\sum_{i=1}^n (Y_i - \bar{Y})^2}{n - 1}}$

so 

$\LARGE r = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{\sqrt{\left( \sum_{i=1}^n (x_i - \bar{x})^2 \right) \left( \sum_{i=1}^n (Y_i - \bar{Y})^2 \right)}}$ 

- $r$ = sample correlation coefficient
- $n$ = number of paired observations
- $x_i$ = value of the $i$-th observation of variable $x$
- $Y_i$ = value of the $i$-th observation of variable $Y$
- $\bar{x}$ = mean of variable $x$
- $\bar{Y}$ = mean of variable $Y$


### Correlation IS NOT Causation
### Correlation JUST measures the Empirical Strength of a Linear Relationship
- It doesn't tell you anything about WHY two things have an association with each other

### But that's it: IT JUST TELLS US WHAT THE PICTURE LOOKS LIKE IN A SINGLE NUMBER (without $n$)



```python
from scipy import stats

np.random.seed(10)
n_points = 25
ice_cream_sales = stats.uniform(10, 10).rvs(n_points)
noise = np.random.normal(0, 2, n_points)
shark_sightings = ice_cream_sales + noise
data = pd.DataFrame({'Ice Cream Sales': ice_cream_sales,
                     'Shark Sightings': shark_sightings})

correlation = data['Ice Cream Sales'].corr(data['Shark Sightings'])

fig = px.scatter(data, x='Ice Cream Sales', y='Shark Sightings', 
                 trendline='ols',  # Add a linear trendline
                 trendline_color_override='red',
                 title='Ice Cream Sales vs. Shark Sightings')

fig.add_annotation(text=f'Correlation: {correlation:.2f}', 
                   xref='paper', yref='paper', 
                   x=0.05, y=0.95, showarrow=False,
                   font=dict(size=12, color='black'), bgcolor='white')
fig.show()
```


```python
from scipy.stats import multivariate_normal

np.random.seed(1)
n = 934  # Number of points

# Define means and standard deviations for midparent and child heights
mean = [66, 64]  # Mean heights for midparent and child
std_dev = [3, 3]  # Standard deviations for midparent and child
correlation = 0.32

# Create the covariance matrix
covariance = np.array([[std_dev[0]**2, correlation * std_dev[0] * std_dev[1]],
                       [correlation * std_dev[0] * std_dev[1], std_dev[1]**2]])

# Generate bivariate normal data
data = multivariate_normal(mean, covariance).rvs(n)
galton_data = pd.DataFrame(data, columns=['Parent Mid Height', 'Child Height'])

# Calculate the correlation
correlation = galton_data['Parent Mid Height'].corr(galton_data['Child Height'])

fig = px.scatter(galton_data, x='Child Height', y='Parent Mid Height', 
                 trendline='ols',  # Add a linear trendline
                 trendline_color_override='red',
                 title='Midparent vs Child Height with Trendline')
fig.update_traces(marker=dict(size=8, opacity=0.5))

# Add correlation annotation
fig.add_annotation(text=f'Correlation: {correlation:.2f}', 
                   xref='paper', yref='paper', 
                   x=0.05, y=0.95, showarrow=False,
                   font=dict(size=12, color='black'), bgcolor='white')
fig.show()
```

### Correlation IS NOT Causation
### Correlation JUST measures the Empirical Strength of a Linear Relationship
- It doesn't tell you anything about WHY two things have an association with each other

### But that's it: it's the linear association pictured (without $n$) in a single number <u>AND/BUT you can BREAK IT</u>



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

### There are MANY kinds of associations (besides correlation) that are not actually LINEAR
- And if you try to "measure" them with **THE LINEAR ASSOCIATION measure CORRELATION** you CAN FOR SURE get the wrong idea of "what the picture looks like"

## Simple Linear Regression

$$ 
\Large
\begin{align}
Y_i = {}& \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)\\
Y_i \sim {}& \mathcal N\left( \beta_0 + \beta_1 x_i, \sigma\right)
\end{align}
$$


- Outcome $Y_i$
- Predictor $x_i$
- Intercept $\beta_0$ coefficient
- Slope $\beta_1$ coefficient
- Error term $\epsilon_i$


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


```python
# Arbitrarily define x and then genrate Y

n = 25
x = stats.uniform(10, 10).rvs(n_points)
Y = 0 + x + stats.norm(loc=0, scale=10).rvs(size=n)

n = 934
x = galton_data['Parent Mid Height']
# x = stats.norm(loc=x.mean(), scale=x.std()).rvs(size=n)
beta0 = -100 # galton_data['Child Height'].mean()
beta1 = 2
Y = beta0 + beta1*x + stats.norm(loc=0, scale=3).rvs(size=n)
```

# The Homework this time around is VERY DIFFERENT
### It's VERY LONG. It's VERY, VERY DEMANDING. You will do/understand COMPLICATED SIMULATIONS
### You don't turn it in until AFTER you get back from READING WEEK (Thursday before TUT as usual)
### Your Project Proposals ARE DUE ON MONDAY IMMEDIATELY UPON RETURN FROM READING WEEK

- The HW is longer since there's substantially more time to do it
- However, I still need to finalize the HW and make the rubric, which 
    - I expect to do tomorrow, Tuesday Oct 22.
    - My apologies for not being quite ready this time around
    - And similarly, the textbook for linear regression has not yet been finalized 
        - but I will do so ASAP, ideally by tomorrow-tomorrow, Wednesday Oct 22.
- A draft of the "Course Project Proposals" assignment is available in the CP folder on the course github
    - This is due on Monday, Nov 04 the day you return from your reading week
    - I will alert the class with an announcement when the final I need to 

# STA130 TUT 7ate9 (Oct25)<br><br>📈❓ <u>Simple Linear Regression</u><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Model Fitting / Hypothesis Testing)

## ♻️ 📚 Review / Questions [10 minutes]

### 1. Follow up questions and clarifications regarding the ideas of **correlation** and the "straight line association" model of **Simple Linear Regression** from the Oct21 LEC<br>

<details class="details-example"><summary><u><span style="color:blue">Simple Linear Regression Terminology</span></u> (reference for <b>Communication Activity #1 question 2</b> below)</summary>

$$ \Large Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma^2\right)$$

- **Outcome** $Y_i$ is a **continuous numeric variable**

> **Outcome** $Y_i$ can also be called a **response**, **dependent**, or **endogenous variable** in some domains and contexts

- **Predictor variable** $x_i$ is a **numeric variable**

> - Fow now we'll consider $x_i$ to be a **continuous** numeric variable, but this is not necessary, and we will consider versions of $x_i$ later
> - **Predictor variable** $x_i$ can also be called an **explanatory**, **independent**, or **exogenous variable**, or a **covariate** or **feature** (which are the preferred terms in the statistics and machine learning domains, respectively)

- **Intercept** $\beta_0$ and **slope** $\beta_1$ are the two primary **parameters** of a **Simple Linear Regression** model

> **Intercept** and **slope** describe a **linear** ("straigh line") relationship between **outcome** $Y_i$ and **predictor variable** $x_i$

- **Error** $\epsilon_i$ (also sometimes called the **noise**) makes **Simple Linear Regression** a **statistical model** by introducing a **random variable** with a **distribution**

- The $\sigma^2$ **parameter** is a part of the **noise distribution** and controls how much vertical variability/spread there is in the $Y_i$ data off of the line: $\sigma^2$ is an "auxiliary" **parameter** in the sense that interest is usually in $\beta_0$ and $\beta_1$ rather than $\sigma^2$

> - **Errors** $\epsilon_i$ (in conjuction with the **linear form**) define the **assumptions** of the **Simple Linear regression** Model specification
> - <u>but these **assumptions** are not the focus of further detailed reviewed here</u>

</details>    
    
<details class="details-example"><summary><u><span style="color:blue">Further details regarding the assumptions</span></u> (which <b>should not the focus of further detailed reviewed here</b>)</summary>

> The first three assumptions associated with the **Simple Linear regression** model are that<br><br>
> 
> 1. the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
> 2. the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
> 3. the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
>
>    and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that<br><br>
> 4. the $x_i$ **predictor variable** is **measured without error**
> 5. and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
> 6. and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)

</details><br>  
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> This TUT will introduce **Hypothesis Testing** in the **Simple Linear Regression** context for the purposes of evaluating a **null hypothesis** assumption of "no association" between two numeric variables $Y$ and $x$ relative to an **alternative hypothesis** of "straight line association" (meaning that changes in the $x$ variable have corresponding changes in the $Y$ variable "on average")
    
</details>

### 2. Follow up questions and clarifications regarding concepts associated with the **sampling distribution** topic <u>*[REALLY needs to be addressed in OH at this point]*<br><br> HW this time is Going To Be DIFFERENT: you MUST understand simulation to do it</u>

> AKA **Hypothesis Testing**, **Sampling Distribution under the Null Hypothesis**, and related topics regarding interpretation from Oct04 TUT and Oct11 TUT; AND, **Sampling Distribution**, **Bootstrapped Confidence Intervals**, and related topics regarding interpretation from Sep27 TUT and Sep30 LEC and Oct07 LEC 

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> Understanding the fundamental underlying mechanism of a **sampling distribution** (most easily demonstrated through **simulation**) is necessary for creating a deep understanding of its (and *the*) two primary applications in **statistics**: **Hypothesis Testing** and **[Bootstrapped] Confidence Intervals**
> 
> Further clear understanding regarding the abstract components and process of **Hypothesis Testing** (decision making regarding parameters using null hypotheses and p-values) is additionally needed from here, as this serves as an unavoidably necessary pre-requesite foundation upon which the enevitable extension of **Hypothesis Testing** to more advanced analyses (such **Multiple Linear Regression** and **permutation testing**, and **Simple Linear Regression** which will be the focus of this TUT) are based
    
</details><br>



## 💬 🗣️ Communication Activity #1 [20 minutes]

To the best of your abilty, recreate the <u>**FIVE**</u> groups from the **Communication Acivity** of the previous (Oct04 and Oct11) TUTs <br><br>

<details class="details-example"><summary style="color:blue"><u>Stella McStat's Wheel of Destiny</u></summary>

We should all by now hopefully be VERY familiar with this by this point in time given that this was a focus of the Oct04 and Oct11 TUTs **and was heavily featured on the midterm exam**...
    
### The Wheel of Destiny

Stella McStat had been running a small-time gambling operation on campus for several months during her first year at UofT... 

- For each spin of the wheel, two gamblers take part. For a toonie each (\\$2 Canadian), Stella sells one a red ticket and one a black ticket  (i.e., total \\$4). Then Stella spins the Wheel of Destiny. The person who holds the colour on which the spinner stops gets \\$3.50 (Stella keeps \\$0.50 per spin for running the game and providing snacks).

Stella just bought a new spinner, the critical piece of equipment for this game. She's heard some mixed reviews about the manufacturer she has purchased from. Before she beings using this spinner, she wants to make sure that it is, in fact, fair (meaning, she wants both colours to come up equally often). Because of the set-up of the game, Stella has no incentive to cheat and wants the game to be as fair as possible.

Everything phystical and mechanical that Stella can examine about the wheel seems fine; there is the same number of sectors of each colour and they each have the same area. BUT! Stella has a great idea and decides to come to YOU, her statistical guru, and ask you to verify that the new spinner is fit to use. Is Stella's game is "fair" (even if somewhat illegal)?

| <img src="https://i.postimg.cc/BvqJwBwc/stella2.png" style="height: 450px;"/> |  <img src="https://i.postimg.cc/vm3GRxJR/fair.png" style="height: 450px;"/> |
|-|-|
|An Exercise for Illustrating the Logic of Hypothesis Testing|Adapted from Lawton, L. (2009), Journal of Stat. Education, 17(2)|
    
</details>

### Discuss the following

1. **[8 of the 20 minutes]** What is the **Null** (and **Alternative**) **Hypothesis** and what is the definition of (and using **simulation** how do you estimate) a **p-value**? 

> First answer this question specifically for the context of "Stella McStat's Wheel of Destiny", but then see if you can give an answer that is more abstract and to some degree "context free" (in terms of **parameters** and [observed versus simulated] **statistics**)

2. **[12 of the 20 minutes]** Examine the theoretical **Simple Linear Regression** model below and consider what a **Null** (and **Alternative**) **Hypothesis** and **p-value** could be for this context? 

$$\Large Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

> **Hints**
> 
> 1. What is the data and how many data points are there?
>  
>  
> 2. What is $\epsilon_i$ and how many of them are there?
>  
>  
> 3. What values can the **slope** $\beta_0$ and **intercept** $\beta_1$ can have?  
>  
> 
> 4. Are **Null** and **Alternative Hypotheses** about **samples** like $x_i$ or $Y_i$ or **sample stastics** like $\bar Y$ or $\bar x$, or **population parameters** like $\mu$?
>  
>  
> 5. There's not a **Null** and **Alternative Hypotheses** regarding $\epsilon_i$, but there are plenty of assumptions about it (which technically are *a part* of the **Null hypothesis**)... what are those assumptions about $\epsilon_i$? 
>  
>  
> 6. Do you have any intuition of how to think about the conceptual meaning of a **p-value** (defined as "the probability that a test statistic is as or more extreme than the observed test statistic if the null hypothesis is true") in terms of **simulation** in the context of **Simple Linear Regression**?
>  
>  
> To be discussed in more detail shortly, the **fitted model** $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ corresponding to the theoretical model above is based on observed **sample data**. So, e.g., the **fitted slope** $\hat \beta_1$ is a **statistic** (which has a **sampling distribution**) that corresponds to the theoretical **slope** parameter $\beta_1$...
   

## Demo (of Model Fitting and Hypothesis Testing for the Simple Linear Regression Model)  [45 minutes]

### Terminology [12 of the 45 minutes]

$$\LARGE \text{Based on data we get} \quad \hat y_i = \hat \beta_0 + \hat \beta_1 x_i \quad \text{from}$$

$$\LARGE Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

<br>

The $\hat y_i = \hat \beta_0 + \hat \beta_1 x_i$ **fitted model** equation distinctly contrasts with the $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$ **theoretical model** specification. To emphasize and clarify the difference, we augment our **simple linear regression** model nomenclature (as given in the "**Review / Questions**" section above) with the contrasting alternative notations and terminology: 

- **Fitted intercept** $\hat \beta_0$ and **slope** $\hat \beta_1$ ***coefficients*** are given "hats" to distinguish that they **estimate** (based on observed **sample data**), respectively, the **intercept** $\beta_0$ and **slope** $\beta_1$ ***parameters***<br><br>

- **Fitted (predicted) values** $\hat y_i$ are made lower case and also given "hats" to distinguish them from the (upper case) **theoretical random variable** $Y_i$ implied by the **theoretical simple linear regression model**
  
> Technically, the **error** $\epsilon_i$ is the **random variable** specified by the **simple linear regression model** specification, and this implies the **random variable** nature of $Y_i$ 

- The **residuals** $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i = y_i - \hat \beta_0 + \hat \beta_1 x_i $ also distinctly contrast with the **errors** (or **noises**) $\epsilon_i$
    
> The **residuals** $\text{e}_i = \hat \epsilon_i$ are actually available, while the **error** (or **noises**) $\epsilon_i$ are just a theoretical concept
> 
> The **residuals** $\text{e}_i = \hat \epsilon_i$ nonetheless are therefore used to diagnostically assess the theoretical modeling assumptions of the  **errors** $\epsilon_i$, such as the **normality**, **homoskedasticity**, and **linear form** assumptions; and, <u>while this is a not necessarily beyond the scope of STA130 and would certainly be a relevant consideration for the course project, this will not be addressed here at this time</u>
    

### Observed Data Setup [5 of the 45 minutes]

Imagine you noticed that the prices of shuttlecocks in the nearby store have increased. At the same time, suppose you are also aware that there has been a recent surge in bird flu cases. You suddenly wonder if there might be a connection between these two events. So you get some historical data as given in the format below.

|Bird Flu Cases |Shuttlecock Price ($)|
|:-------------:|:----------------------------:|
|1000           |3.0                           |
|1522           |4.2                           |
|$$\vdots$$     |$$\vdots$$                    |
|1200           |3.2                           |

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> Actually, this data and all the related (analysis and plotting) code was made by just giving instructions to a ChatBot, and tweaking things a little bit. As long as you know what to ask for and what you're looking for a ChatBot can carry out **Simple Linear Regression** and related tasks (and follow any specific analyses adjustment directions you  request).

</details>

#### In the visual representation of the data below, what are we considering the (dependent) outcome $Y_i$ and what are we considering the (independent) predictor $x_i$? Does this seem sensible given the framing of our inquiry? 

|<img src="https://www.mumbailive.com/images/news/bird-flu1_151660804012.jpg?w=1368" alt="Bird Flu" style="width: 500px; height: 300px;"/>|<img src="https://www.badmintonskills.net/wp-content/uploads/2015/09/Badminton-004.jpg?x83573" alt="Shuttlecock" style="width: 250px; height: 300px;"/>|
|-:|:-|
|Assess a possible association between bird flu prevalence and the price of shuttlecocks| using **Simple Linear Regression**|



```python
import pandas as pd
import plotly.express as px

# Here's the data
data = {
    'Bird Flu Cases': [1000, 1522, 1300, 
        1450, 1550, 1350, 1250, 1500, 1150, 1650, 1300, 1400, 1750, 
        1800, 900, 1100, 1700, 1400, 1600, 1200],
    'Shuttlecock Price': [3.0, 4.3, 3.7, 
        3.9, 4.0, 3.5, 3.4, 4.0, 3.2, 4.4, 3.6, 3.7, 4.6, 
        4.9, 2.8, 3.1, 4.6, 3.9, 4.4, 3.2]}
df = pd.DataFrame(data)

# Here's the data visually
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', 
                 title='Shuttlecock Price vs. Bird Flu Cases',
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
#https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()
```

### *Model Fitting* via `statsmodels.formula.api as smf` and `smf.ols(...).fit()`<br>[16 of the 45 minutes]

- First we'll demonstrate getting a fitted Simple Linear Regression Model using `statsmodels`  and working with the key elements of the fitted model [10 of these 15 minutes]  


- Then we'll visually demonstrate the fitted Simple Linear Regression model based on its estimated intercept and slope parameters [5 of these 15 minutes]


```python
import statsmodels.formula.api as smf

# And here's how to do "Model Fitting" for Simple Linear Regression with `statsmodels`

# Use "Y~x" R-style formulas: https://www.statsmodels.org/stable/example_formulas.html
linear_specification = 'Q("Shuttlecock Price") ~ Q("Bird Flu Cases")'
# The notation above is admittidly starnge, but it's because 
# there are spaces in my column (variable) names in the data

# Put the data into the a `statsmodels` "model" object
model_data_specification = smf.ols(linear_specification, data=df)

# Fit the model
fitted_model = model_data_specification.fit()
```


```python
# See the results...
fitted_model.summary()
```


```python
# There's too much in the full `.summary()`: just focus on this table for now
fitted_model.summary().tables[1] # does this indexing make sense?
```

$$\LARGE \text{For the data above} \quad \hat y_i = 0.5361+0.0023 \times x_i$$

$$\LARGE \text{is the} \quad \hat y_i = \hat \beta_0 + \hat \beta_1 x_i \quad \text{estimating the model}$$

$$\LARGE Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

#### What are $\hat \beta_0$ and $\hat \beta_1$? What are $Y_i, x_i$, and $\hat y_i$ and $\text{e}_i = \hat \epsilon_i$? How do you interpret the fitted model?<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> When we create a **fitted Simple Linear Regression model** for an observed data dataset, we obtain **estimates** of what the data suggests the **intercept** and **slope** could be to form an equation of the (predicted) values for the data
> 
> Model fitting is typically based on the "ordinary least squares" (`ols`) concept (maximizing **R-squared**), although there are analytical closed form solutions for the **intercept** **slope coefficient estimates** for **Simple Linear Regression models**; but, these considerations will not be detailed further here (as they'll be the focus of a HW question) and <u>we'll now instead just focus on fitting models using the `python` `statsmodels` package</u>
    
</details>







```python
Y = df["Shuttlecock Price"]
x = df["Bird Flu Cases"]

# The printout coeficient values are rounded
# pd.DataFrame({"formula": 0.4291+0.0024*x,
#               "model": fitted_model.fittedvalues})

# So we use the exact fitted coefficient values using `fitted_model.params`# (actually, `fitted_model.params.values[0]`)
y_hat = fitted_model.fittedvalues
df['y-hat (from model)'] = y_hat 
df['y-hat (from formula)'] = 0.536107332688726+0.0023492341183347265*x
df

# `fitted_model.fittedvalues` is the same as `fitted_model.predict(df)`
# but the latter is more genereal 
# as it could be used to predict new values based on a different data frame...
```


```python
# residuals
e = Y - fitted_model.fittedvalues  # df['Shuttlecock Price'] - fitted_model.fittedvalues
df['e (Residuals)'] = e
# or you can just use `fitted_model.resid`
df['e (Residuals) v2'] = fitted_model.resid
df
```

- The **intercept** $\hat \beta_0$ is `0.5361`
- The **slope** $\hat \beta_1$  is `0.0023` and is labeled `Q("Bird Flu Cases")` in the output
    - The **outcome** $Y_i$ is `Q("Shuttlecock Price")`
    - The **predictor** $x_i$ is `Q("Bird Flu Cases")`
    - The **slope** is the "on average" change in $Y_i$ per "single unit" change in $x_i$
   
- A **fitted (predicted) value** $\hat y_i$ is found by "plugging in" $x_i$ and calculating $0.5361+0.0023 \times x_i$
- A **residual** is calculated as $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i = y_i - \hat \beta_0 + \hat \beta_1 x_i $



```python
# Code is commented to indicate its visualization/demonstration purpose:
# students may study smaller details beyond the "big picture" later 
# in a ChatBot session if so inclined

import numpy as np

# Here's the model fit visually
df['Original Data'] = 'Original Data' # hack to add legend item for data points
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', color='Original Data',
                 title='Shuttlecock Price vs. Bird Flu Cases',
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'},
                 trendline='ols')
fig.update_traces(marker=dict(size=10))
              
# This is what `trendline='ols'` does
fig.add_scatter(x=df['Bird Flu Cases'], y=fitted_model.fittedvalues,
                line=dict(color='blue', width=3), name="trendline='ols'")
    
# Adding the line of the math expression
x_range = np.array([df['Bird Flu Cases'].min(), df['Bird Flu Cases'].max()])
y_line = 0.536107332688726+0.0023492341183347265 * x_range
fig.add_scatter(x=x_range, y=y_line, mode='lines', name='0.5361 + 0.0023 * x', 
                line=dict(dash='dot', color='orange'))

# Adding predicted values as points
fig.add_scatter(x=df['Bird Flu Cases'], y=df['y-hat (from model)'], mode='markers', 
                name='Fitted (Predicted) Values', 
                marker=dict(color='black', symbol='cross', size=10))

fig.update_layout(legend_title=None)
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### [Omitted] Model Diagnostics: evaluating the assumptions of Simple Linear Regression [0 of the 45 minutes]


#### To evaluate if the assumptions of "normality" and "heteroskedasticity" ($x_i$ agnostic variance) of the theoretical distribution of the error (noise) terms we see if the residuals appear to be normally distributed... this is not really enough data to determine this convincingly one way or another at this point

#### In the context of Simple Linear Regression (as opposed to Multiple Linear Regression), we could examine the scatter plot to see if the assumption of a "linear form of the model" appears "true" plot... in the original scatter plot of the data there appears to be some potential "curve" in the relationship, but again there's really not enough data to determine this convincingly "by eye" at this point


```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

from scipy import stats

n = len(df['e (Residuals)'])
normality_heteroskedasticity_diagnostic_judgement = \
'<br>[Seems to plausibly be a (n='+str(n)+') sample from a "normal" distribution]'
df['Observed Residuals'] = 'Observed Residuals' # hack to add legend item for data points
fig = px.histogram(df, x='e (Residuals)', color='Observed Residuals',
                   title='Histogram of Residuals'+normality_heteroskedasticity_diagnostic_judgement)

# rerun this cell to see repeated examples
random_normal_sample = stats.norm(loc=0, scale=df['e (Residuals)'].std()).rvs(size=n) 

fig.add_histogram(x=random_normal_sample, name='Random Normal Sample', 
                  histfunc='count', opacity=0.5, marker=dict(color='orange'))
fig.update_layout(barmode='overlay', legend_title=None)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```


```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

linearity_diagnostic_judgement = '"<br>[Straight line" fit appears "reasonable"]'
# uncomment/comment `trendline='ols'` to toggle the "straight line fit" on and off
fig = px.scatter(df, x='Bird Flu Cases',  y='Shuttlecock Price', color='Original Data',
                 #trendline='ols',
                 title='Shuttlecock Price vs. Bird Flu Cases'+linearity_diagnostic_judgement,
                 labels={'Bird Flu Cases': 'Bird Flu Cases',
                         'Shuttlecock Price': 'Shuttlecock Price ($)'})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### Hypothesis Testing (for Simple Linear Regression) [12 of the 45 minutes]

We can use **Simple Linear Regression** to test

$\large
\begin{align}
H_0: {}& \beta_1=0 \quad \text{ (there is no linear assocation between $Y_i$ and $x_i$ "on average")}\\
H_A: {}& H_0 \text{ is false}
\end{align}$

That is, we can assess the evidence of a linear association in the data based on a **null hypothesis** that the **slope** (the "on average" change in $Y_i$ per "single unit" change in $x_i$) is zero

#### Did your group get $H_0$ correct in your answers for *Communication Activity #1 question 2*?<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> We are essentially never (or only very rarely in very special circumstances) interested in a **null hypothesis** concerning the **intercept** $\beta_0$ (as opposed to $\beta_1$)
> 
> $\Large
\begin{align}
H_0: {}& \beta_0=0\\
H_A: {}& H_0 \text{ is false}
\end{align}$
>
> This is because the assumption that $\beta_0$ is zero essentially never (or only very rarely in very special circumstances) has any meaning, whereas the assumption that $\beta_1$ is zero has the very practically useful interpretation of "no linear association" which allows us to evaluate the  evidence of a linear association based on observed data
    
</details>

#### How do we use the fitted Simple Linear Regression model to assess $H_0$ regarding $\beta_1$? Where do we find the p-value we use to make our assessment of $H_0$ and how do we interpret the p-value to make a decision?<br>
 
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
 
Remember, the **p-value** is "the probability that a test statistic is as or more extreme than the observed test statistic if the null hypothesis is true"

- We do not prove $H_0$ false, we instead give evidence against the $H_0$
     - "We reject the null hypothesis with a p-value of abc, meaning we have xyz evidence against the null hypothesis"
- We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$
     - "We fail to reject the null hypothesis with a p-value of abc"
|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
   
</details>
    


```python
fitted_model.summary().tables[1] # does this indexing make sense?
```

## 📢 👂 Communication Activity #2 [final 15 minutes]

### If you don't complete this NOW in TUT students will have to complete it ON THEIR OWN for the <u>EXTREMELY VERY BIG AND VERY DIFFERENT</u> Homework 06 <sub>*HW06_Week07ate09_DueNov07STA130_HW06_Week07ate09_DueNov07STA130_HW06_Week07ate09_DueNov07STA130*</sub>

In order to follow up and explain the answers to **Communication Activity #1 question 2**, each of the <u>**FIVE**</u> groups from **Communication Activity #1** will sequentially volunteer to present answers to these questions (**taking average three minutes per group**) to the class (in order as quickly as possible, with groups dynamically helping each other answer their question if needed) 

1. Explain how the "uncertainty band" in the `seaborn.regplot` of the **Further Illustrations** below represents a **bootstrapped sampling distribution** (or slighly more accurately something like a "95% confidence interval") for "lines of best fit"  


2. Explain how this so-called "**sampling distribution**" of the "line of best fit" could be "sampled"
    1. by making a **bootstrapped sampling distribution** 
    2. by assuming the **population model** and creating **simulations** 


3. Explain how the **sampling distribution** of (just) the **estimated slope** $\hat \beta_1$ could be **simulated** and the **p-value** for a **null hypothesis** of "no linear association" created using **simulation** and used to assess the evidence against the **null hypothesis**  
    1. Also explain how a **95% bootstrapped confidence interval** of the **slope coefficient** $\hat \beta_1$ could be constructed


4. Find the "R-squared" in the `fitted_model.summary()` table (or accessible via `fitted_model.rsquared`) and compare this value with 
    1. `np.corrcoef(Y,x)[0,1]**2`,  
    2. `np.corrcoef(Y,y_hat)[0,1]**2`,   
    3. and `1-((Y-y_hat)**2).sum()/((Y-Y.mean())**2).sum()` (where `Y`,`x`, and `y_hat` have been defined in the notebook above for the orignal data); 
    4. then, explain (a) what the two `np.corrcoef...` expressions capture, (b) why the final expression can be interpreted as "the proportion of variation in (outcome) Y explained by the model (y_hat)", and (c) therefore why `fitted_model.rsquared` can be interpreted as a measure of the accuracy of the model  


5. Explain what our likely judgement about the **Model Diagnostics** in the <u>**Omitted**</u> section above would be for data **simulated** based on an assumed **population model** (as opposed to using **bootstrapping**), and what would cause an analogous judgement them to fail for observed data (and what this means about the appropriateness of the theoretical **Simple Linear Regression** model for this data)


### Further Illustration 

> The `seaborn` plotting library has a function which shows the uncertainty of the (trendline) "straight line fit" based on the available data: anywhere a straight line can be drawn through the "uncertainty band" is plausible as far as the evidence in the observed data is concerned

#### How does the vertical spread of the $Y_i$ outcomes (tightness around the "line of best fit") affect the evidence against the null hypothesis? What does this mean in terms of our belief about the evidence against a *null hypothesis* of no linear association "on average" between Bird Flu Cases and Shuttlecock Price? 



```python
# Figure for demonstration/visualization purposes only:
# students may study smaller details later in a ChatBot session if so inclined

import seaborn as sns
import matplotlib.pyplot as plt

spread = 20 # increase this to vertically spread the data (1 recreated the original data)
df["Synthetically Spread y"] = fitted_model.fittedvalues + spread*df['e (Residuals)']

linear_specification_ = 'Q("Synthetically Spread y") ~ Q("Bird Flu Cases")'
model_data_specification_ = smf.ols(linear_specification_, data=df)
fitted_model_ = model_data_specification_.fit()
print(fitted_model_.summary().tables[1])

sns.regplot(x='Bird Flu Cases', y='Synthetically Spread y', data=df) #, line_kws={'color': 'red'})
plt.title('Bird Flu Cases vs. Shuttlecock Price ($)')
plt.xlabel('Bird Flu Cases')
plt.ylabel('Shuttlecock Price ($)')
plt.show()
```


```python

```
# STA130 Homework 06

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
    
1. Code and write all your answers (for both the "Pre-lecture" and "Post-lecture" HW) in a python notebook (in code and markdown cells) 
    
> It is *suggested but not mandatory* that you complete the "Pre-lecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
    
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
- [0.2 points]: Evaluation of correctness and clarity in written communication for Question "3"
- [0.2 points]: Evaluation of correctness and clarity in written communication for Question "4"
- [0.2 points]: Evaluation of submitted work and conclusions for Question "9"
- [0.3 points]: Evaluation of written communication of the "big picture" differences and correct evidence assessement for Question "11"


## "Pre-lecture" versus "Post-lecture" HW? 

- _**Your HW submission is due prior to the Nov08 TUT on Friday after you return from Reading Week; however,**_
- _**this homework assignment is longer since it covers material from both the Oct21 and Nov04 LEC (rather than a single LEC); so,**_
- _**we'll brake the assignment into "Week of Oct21" and "Week of Nov04" HW and/but ALL of it will be DUE prior to the Nov08 TUT**_


## "Week of Oct21" HW [*due prior to the Nov08 TUT*]

### 1. Explain the theoretical Simple Linear Regression model in your own words by describing its components (of predictor and outcome variables, slope and intercept coefficients, and an error term) and how they combine to form a sample from normal distribution; then, create *python* code explicitly demonstrating your explanation using *numpy* and *scipy.stats* <br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Your answer can be quite concise and will likely just address the "mathematical" and "statistical" aspects of the process of a **Simple Linear Model** specification, perhaps giving an intuitive interpretation summary of the result as a whole_
>   
> - _Your code could be based on values for `n`, `x`, `beta0`, `beta1`, and `sigma`; and, then create the `errors` and `Y`_
> 
> - _The predictors $x_i$ can be fixed arbitrarily to start the process (perhaps sampled using `stats.uniform`), and they are conceptually different from the creation of **error** (or **noise**) terms $\epsilon_i$ which are sampled from a **normal distribution** (with some aribtrarily *a priori* chosen **standard deviation** `scale` parameter $\sigma$) which are then combined with $x_i$ through the **Simple Linear Model** equation (based on aribtrarily *a priori* chosen **slope** and **intercept coefficients**) to produce the $Y_i$ outcomes_
> 
> - _It should be fairly easy to visualize the "a + bx" line defined by the **Simple Linear Model** equation, and some **simulated** data points around the line in a `plotly` figure using the help of a ChatBot_
> 
> _If you use a ChatBot (as expected for this problem), **don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)**_
>
> 
> _**Question Scope Warning:** Be careful when using a ChatBot to help you with creating an example dataset and coding up a visualization though, **because it might suggest creating (and visualizing) a fitted model for to your data (rather than the theoretical model); but, this is not what this question is asking you to demonstrate**. This question is not asking about how to produce a fitted **Simple Linear Regression** model or explain how model **slope** and **intercept coefficients** are calculated (e.g., using "ordinary least squares" or analytical equations to estimate the **coefficients**  for an observed dataset)._
> 
> ```python
> # There are two distinct ways to use `plotly` here
>
> import plotly.express as px
> px.scatter(df, x='x',  y='Y', color='Data', 
>            trendline='ols', title='Y vs. x')
>        
> import plotly.graph_objects as go
> fig = go.Figure()
> fig.add_trace(go.Scatter(x=x, y=Y, mode='markers', name='Data'))
> 
> # The latter is preferable since `trendline='ols'` in the former 
> # creates a fitted model for the data and adds it to the figure
> # and, again, THAT IS NOT what this problem is asking for right now
> ```
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>


### 2. Use a dataset simulated from your theoretical Simple Linear Regression model to demonstrate how to create and visualize a fitted Simple Linear Regression model using *pandas* and *import statsmodels.formula.api as smf*<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> - _Combine the **simulated** `x` and `Y` into a `pandas` data frame object named `df` with the column names "x" and "Y"_
> 
> - _Replace the inline question comments below with their answers (working with a ChatBot if needed)_
>
> ```python
> import statsmodels.formula.api as smf  # what is this library for?
> import plotly.express as px  # this is a ploting library
>
> # what are the following two steps doing?
> model_data_specification = smf.ols("Y~x", data=df) 
> fitted_model = model_data_specification.fit() 
>
> # what do each of the following provide?
> fitted_model.summary()  # simple explanation? 
> fitted_model.summary().tables[1]  # simple explanation?
> fitted_model.params  # simple explanation?
> fitted_model.params.values  # simple explanation?
> fitted_model.rsquared  # simple explanation?
>
> # what two things does this add onto the figure?
> df['Data'] = 'Data' # hack to add data to legend 
> fig = px.scatter(df, x='x',  y='Y', color='Data', 
>                  trendline='ols', title='Y vs. x')
>
> # This is essentially what above `trendline='ols'` does
> fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
>                 line=dict(color='blue'), name="trendline='ols'")
> 
> fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
> ```
>
> _The plotting here uses the `plotly.express` form `fig.add_scatter(x=x, y=Y)` rather than the `plotly.graph_objects` form `fig.add_trace(go.Scatter(x=x, y=Y))`. The difference between these two was noted in the "Further Guidance" comments in the previous question; but, the preference for the former in this case is because `px` allows us to access `trendline='ols'` through `px.scatter(df, x='x',  y='Y', trendline='ols')`_
>
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_      

</details>

### 3. Add the line from Question 1 on the figure of Question 2 and explain the difference between the nature of the two lines in your own words; *but, hint though: simulation of random sampling variation*<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _This question is effectively asking you to explain what the combined code you produced for Questions 1 and 2 is trying to demonstrate overall. If you're working with a ChatBot (as expected), giving these two sets of code as context, and asking what the purpose of comparing these lines could be would be a way to get some help in formulating your answer_
> 
> _The graphical visualization aspect of this question could be accomplished by appending the following code to the code provided in Question 2._
> 
> ```python
> # what does this add onto the figure in constrast to `trendline='ols'`?
> x_range = np.array([df['x'].min(), df['x'].max()])
> # beta0 and beta1 are assumed to be defined
> y_line = beta0 + beta1 * x_range
> fig.add_scatter(x=x_range, y=y_line, mode='lines',
>                 name=str(beta0)+' + '+str(beta1)+' * x', 
>                 line=dict(dash='dot', color='orange'))
>
> fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
> ```
> 
> _The right way to interactively "see" the answer to this question is to repeatedly create different dataset **simulations** using your theoretical model and the corresponding fitted models, and repeatedly visualize the data and the two lines over and over... this would be as easy as rerunning a single cell containing your simulation and visualization code..._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>

### 4. Explain how *fitted_model.fittedvalues* are derived on the basis of *fitted_model.summary().tables[1]* (or more specifically  *fitted_model.params* or *fitted_model.params.values*)<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _The previous questions used code to explore the distinction between theoretical (true) $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i \;[\text{where } \epsilon_i \sim \mathcal{N}(0, \sigma)]\;$ and fitted (estimated) $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ **Simple Linear Regression** models_
>
> _This question asks you to explicitly illustrate how the the latter "in sample predictions" of the fitted **Simple Linear Regression** model $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ are made (in contrast to the linear equation of the theoretical model)_
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>

### 5. Explain concisely in your own words what line is chosen for the fitted model based on observed data using the "ordinary least squares" method (as is done by *trendline='ols'* and *smf.ols(...).fit()*) and why it requires "squares"<br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _This question addresses the use of **residuals** $\text{e}_i = \hat \epsilon_i = Y_i - \hat y_i$ (in contrast to the **error** terms $\epsilon_i$ of the theoretical model), and particularly, asks for an explanation based on the following visualization_
>
> ```python 
> import scipy.stats as stats
> import numpy as np
> import pandas as pd
> import statsmodels.formula.api as smf
> import plotly.express as px
> 
> n,x_min,x_range,beta0,beta1,sigma = 20,5,5,2,3,5
> x = stats.uniform(x_min, x_range).rvs(size=n)
> errors = stats.norm(loc=0, scale=sigma).rvs(size=n)
> Y = beta0 + beta1 * x + errors
> 
> df = pd.DataFrame({'x': x, 'y': Y})
> model_data_specification = smf.ols("Y~x", data=df) 
> fitted_model = model_data_specification.fit() 
> 
> df['Data'] = 'Data' # hack to add data to legend 
> fig = px.scatter(df, x='x',  y='Y', color='Data', 
>                  trendline='ols', title='Y vs. x')
> 
> # This is what `trendline='ols'` is
> fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
>                 line=dict(color='blue'), name="trendline='ols'")
> 
> x_range = np.array([df['x'].min(), df['x'].max()])
> y_line = beta0 + beta1 * x_range
> fig.add_scatter(x=x_range, y=y_line, mode='lines',
>                 name=str(beta0)+' + '+str(beta1)+' * x', 
>                 line=dict(dash='dot', color='orange'))
> 
> # Add vertical lines for residuals
> for i in range(len(df)):
>     fig.add_scatter(x=[df['x'][i], df['x'][i]],
>                     y=[fitted_model.fittedvalues[i], df['Y'][i]],
>                     mode='lines',
>                     line=dict(color='red', dash='dash'),
>                     showlegend=False)
>     
> # Add horizontal line at y-bar
> fig.add_scatter(x=x_range, y=[df['Y'].mean()]*2, mode='lines',
>                 line=dict(color='black', dash='dot'), name='y-bar')
> 
> fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
> ```
>
> _**Question Scope Warning**: we are not looking for any explanation realted to the mathematical equations for the line chosen for the **Simple Linear Regression** model by the "ordinary least squares" method, which happen to be_
> 
> _$$\hat \beta_1 = r_{xy}\frac{s_y}{s_x} \quad \text{ and } \quad  \hat\beta_0 = \bar {y}-\hat \beta_1\bar {x}$$_
>
> _where $r_{xy}$ is the **correlation** between $x$ and $Y$ and $s_x$ and $s_Y$ are the **sample standard deviations** of $x$ and $y$_
>
> ---
> 
> ```python 
> # Use this if you need it    
> # https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
> import plotly.offline as pyo
> # Set notebook mode to work in offline
> pyo.init_notebook_mode()    
> ```
>
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
    
</details>

### 6. Explain why the first expression below can be interpreted as "the proportion of variation in (outcome) Y explained by the model (i.e. _fitted_model.fittedvalues_)"; and therefore, why _fitted_model.rsquared_ can be interpreted as a measure of the accuracy of the model; and, therefore what the two _np.corrcoef(...)[0,1]\*\*2_ expressions capture in the context of _Simple Linear Regression models_.

1. `1-((Y-fitted_model.fittedvalues)**2).sum()/((Y-Y.mean())**2).sum()`
2. `fitted_model.rsquared`
3. `np.corrcoef(Y,fitted_model.fittedvalues)[0,1]**2`
4. `np.corrcoef(Y,x)[0,1]**2`<br><br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _**R-squared** is the "the proportion of variation in (outcome) $Y$ explained by the model ($\hat y_i$)" and is defined as_
>
> _$R^2 = 1 - \frac{\sum_{i=1}^n(Y_i-\hat y)^2}{\sum_{i=1}^n(Y_i-\bar Y)^2}$_
>
> _The visuzation provided in the previous problem can be used to consider $(Y_i-\bar Y)^2$ as the squared distance of the $Y_i$ to their sample average $\bar Y$ as opposed to the squared **residuals** $(Y_i-\hat y)^2$ which is the squared distance of the $Y_i$ to their fitted (predicted) values $Y_i$._
>    
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>

### 7. Indicate a couple of the assumptions of the *Simple Linear Regression* model specification that do not seem compatible with the example data below<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Hint: What even ARE the assumptions of the  **Simple Linear Regression** model, you ask? Have a look at the mathematical specification and see if what it seems to be assuming._
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>


```python
import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots

# This data shows the relationship between the amount of fertilizer used and crop yield
data = {'Amount of Fertilizer (kg) (x)': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 
                                          2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 
                                          4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 
                                          6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 
                                          8.2, 8.4, 8.6, 8.8,9, 9.2, 9.4, 9.6],
        'Crop Yield (tons) (Y)': [18.7, 16.9, 16.1, 13.4, 48.4, 51.9, 31.8, 51.3, 
                                  63.9, 50.6, 58.7, 82.4, 66.7, 81.2, 96.5, 112.2, 
                                  132.5, 119.8, 127.7, 136.3, 148.5, 169.4, 177.9, 
                                  186.7, 198.1, 215.7, 230.7, 250.4, 258. , 267.8, 
                                  320.4, 302. , 307.2, 331.5, 375.3, 403.4, 393.5,
                                  434.9, 431.9, 451.1, 491.2, 546.8, 546.4, 558.9]}
df = pd.DataFrame(data)
fig1 = px.scatter(df, x='Amount of Fertilizer (kg) (x)', y='Crop Yield (tons) (Y)',
                  trendline='ols', title='Crop Yield vs. Amount of Fertilizer')

# Perform linear regression using scipy.stats
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(df['Amount of Fertilizer (kg) (x)'], df['Crop Yield (tons) (Y)'])
# Predict the values and calculate residuals
y_hat = intercept + slope * df['Amount of Fertilizer (kg) (x)']
residuals = df['Crop Yield (tons) (Y)'] - y_hat
df['Residuals'] = residuals
fig2 = px.histogram(df, x='Residuals', nbins=10, title='Histogram of Residuals',
                    labels={'Residuals': 'Residuals'})

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Crop Yield vs. Amount of Fertilizer', 
                                    'Histogram of Residuals'))
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(title='Scatter Plot and Histogram of Residuals',
    xaxis_title='Amount of Fertilizer (kg)', yaxis_title='Crop Yield (tons)',
    xaxis2_title='Residuals', yaxis2_title='Frequency', showlegend=False)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

## "Week of Nov04" HW [due prior to the Nov08 TUT]

_**In place of the "Data Analysis Assignment" format we introduced for the previous weeks' HW, the remaining questions will be a collection of exercises based around the following data**_

> The details of the "LOWESS Trendline" shown below are not a part of the intended scope of the activities here, but it is included since it is suggestive of the questions we will consider and address here



```python
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

# The "Classic" Old Faithful Geyser dataset: ask a ChatBot for more details if desired
old_faithful = sns.load_dataset('geyser')

# Create a scatter plot with a Simple Linear Regression trendline
fig = px.scatter(old_faithful, x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions", 
                 trendline='ols')#'lowess'

# Add a smoothed LOWESS Trendline to the scatter plot
lowess = sm.nonparametric.lowess  # Adjust 'frac' to change "smoothness bandwidth"
smoothed = lowess(old_faithful['duration'], old_faithful['waiting'], frac=0.25)  
smoothed_df = pd.DataFrame(smoothed, columns=['waiting', 'smoothed_duration'])
fig.add_scatter(x=smoothed_df['waiting'], y=smoothed_df['smoothed_duration'], 
                mode='lines', name='LOWESS Trendline')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### 8. Specify a *null hypothesis* of "no linear association (on average)" in terms of the relevant *parameter* of the *Simple Linear Regression* model, and use the code below to characterize the evidence in the data relative to the *null hypothesis* and interpret your subsequent beliefs regarding the Old Faithful Geyser dataset.<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Remember that **Hypothesis Testing** is not a "mathematical proof"_
>
> - _We do not prove $H_0$ false, we instead give evidence against the $H_0$: "We reject the null hypothesis with a p-value of XYZ, meaning we have ABC evidence against the null hypothesis"_
> - _We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$: "We fail to reject the null hypothesis with a p-value of XYZ"_

|p-value|Evidence|
|-|-|
|$$p > 0.1$$|No evidence against the null hypothesis|
|$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
|$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
|$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
|$$0.001 \ge p$$|Very strong evidence against the null hypothesis|

</details>    

> ```python
> import seaborn as sns
> import statsmodels.formula.api as smf
>
> # The "Classic" Old Faithful Geyser dataset
> old_faithful = sns.load_dataset('geyser')
> 
> linear_for_specification = 'duration ~ waiting'
> model = smf.ols(linear_for_specification, data=old_faithful)
> fitted_model = model.fit()
> fitted_model.summary()
> ```


### 9. As seen in the introductory figure above, if the delay of the geyser eruption since the previous geyser eruption exceeds approximately 63 minutes, there is a notable increase in the duration of the geyser eruption itself. In the figure below we therefore restrict the dataset to only short wait times. Within the context of only short wait times, is there evidence in the data for a relationship between duration and wait time in the same manner as in the full data set? Using the following code, characterize the evidence against the *null hypothesis* in the context of short wait times which are less than  *short_wait_limit* values of *62*, *64*, *66*.<br>



```python
import plotly.express as px
import statsmodels.formula.api as smf


short_wait_limit = 62 # 64 # 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### 10. Let's now consider just the (*n=160*) long wait times (as specified in the code below), and write code to do the following:

1. create fitted **Simple Linear Regression** models for **boostrap samples** and collect and visualize the **bootstrapped sampling distribution** of the **fitted slope coefficients** of the fitted models;  


2. **simulate** samples (of size `n=160`) from a **Simple Linear Regression** model that uses $\beta_0 = 1.65$, $\beta_1 = 0$, $\sigma = 0.37$ along with the values of `waiting` for $x$ to create **simuations** of $Y$ and use these collect and visualize the **sampling distribution** of the **fitted slope coefficient** under a **null hypothesis** assumption of "no linear association (on average)"; then,  


3. report if $0$ is contained within a 95\% **bootstrapped confidence interval**; and if the **simulated p-value** matches `smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1]`?<br><br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _You'll need to create `for` loops to repeatedly create fitted **Simple Linear Regression** models using different samples, collecting the **fitted slope coeffient** created in each `for` loop "step" in order to visualize the **simulated sampling distributions**_
> 
> - _A **bootstrapped sample** of the "long wait times" dataset can be created with `old_faithful[long_wait].sample(n=long_wait.sum(), replace=True)`_
>
>
> - _A **simulated** version of the "long wait times under a null hypothesis assumption of **no linear association (on average)**" dataset can be created by first creating `old_faithful_simulation = old_faithful[long_wait].copy()` and then assigning the **simulated** it values with `old_faithful_simulation['duration'] = 1.65 + 0*old_faithful_simulation.waiting + stats.norm(loc=0, scale=0.37).rvs(size=long_wait.sum())`_ 
>
>  _The values $\beta_0 = 1.65$ and $\sigma = 0.37$ are chosen to match what is actually observed in the data, while $\beta_1 = 0$ is chosen to reflect a **null hypothesis** assumption of "no linear assocaition (on average)"; and, make sure that you understand why it is that_
>
>
> - _if `bootstrapped_slope_coefficients` is the `np.array` of your **bootstrapped slope coefficients** then `np.quantile(bootstrapped_slope_coefficients, [0.025, 0.975])` is a 95\% **bootstrapped confidence interval**_
> 
>
> - _if `simulated_slope_coefficients` is the `np.array` of your **fitted slope coefficients** **simulated** under a **null hypothesis** "no linear association (on average)" then `(np.abs(simulated_slope_coefficients) >= smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().params[1]).mean()` is the **p-value** for the **simulated** **simulated sampling distribution of the slope coeficients** under a **null hypothesis** "no linear association (on average)"_

</details>
<br>


```python
import plotly.express as px

long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[long_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (>"+str(long_wait_limit)+")", 
                 trendline='ols')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### 11. Since we've considered wait times of around <64  "short" and wait times of >71 "long", let's instead just divide the data and insead call wait times of <68 "short" and otherwise just call them "long". Consider the *Simple Linear Regression* model specification using an *indicator variable* of the wait time length<br>

$$\large Y_i = \beta_{\text{intercept}} + 1_{[\text{"long"}]}(\text{k_i})\beta_{\text{contrast}} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$

### where we use $k_i$ (rather than $x_i$) (to refer to the "kind" or "katagory" or "kontrast") column (that you may have noticed was already a part) of the original dataset; and, explain the "big picture" differences between this model specification and the previously considered model specifications<br>

1. `smf.ols('duration ~ waiting', data=old_faithful)`
2. `smf.ols('duration ~ waiting', data=old_faithful[short_wait])`
3. `smf.ols('duration ~ waiting', data=old_faithful[long_wait])`

### and report the evidence against a *null hypothesis* of "no difference between groups "on average") for the new *indicator variable* based model<br>



```python
from IPython.display import display

display(smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().summary().tables[1])

fig = px.box(old_faithful, x='kind', y='duration', 
             title='duration ~ kind',
             category_orders={'kind': ['short', 'long']})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_  
</details>

### 12. Identify which of the histograms suggests the plausibility of the assumption that the distribution of *error* terms is normal for each of the models, and explain why the other three do not support this assumption.

> Hint: Question 5 of the *Communication Activity #2* of the Oct25 TUT (addressing an *omitted* section of the TUT) discusses how the assumption in *Simple Linear Regression* that the *error* terms $\epsilon_i \sim \mathcal N\left(0, \sigma\right)$ is diagnostically assessed by evaluating distributional shape of the *residuals* $\text{e}_i = \hat \epsilon_i = Y_i - \hat y_i$



```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
import numpy as np

model_residuals = {
    '<br>Model 1:<br>All Data using slope': smf.ols('duration ~ waiting', data=old_faithful).fit().resid,
    '<br>Model 2:<br>Short Wait Data': smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().resid,
    '<br>Model 3:<br>Long Wait Data': smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().resid,
    '<br>Model 4:<br>All Data using indicator': smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().resid
}

fig = make_subplots(rows=2, cols=2, subplot_titles=list(model_residuals.keys()))
for i, (title, resid) in enumerate(model_residuals.items()):

    if i == 1:  # Apply different bins only to the second histogram (index 1)
        bin_size = dict(start=-1.9, end=1.9, size=0.2)
    else:
        bin_size = dict(start=-1.95, end=1.95, size=0.3)

    fig.add_trace(go.Histogram(x=resid, name=title, xbins=bin_size, histnorm='probability density'), 
                  row=int(i/2)+1, col=(i%2)+1)
    fig.update_xaxes(title_text="n="+str(len(resid)), row=int(i/2)+1, col=(i%2)+1)    
    
    normal_range = np.arange(-3*resid.std(),3*resid.std(),0.01)
    fig.add_trace(go.Scatter(x=normal_range, mode='lines', opacity=0.5,
                             y=stats.norm(loc=0, scale=resid.std()).pdf(normal_range),
                             line=dict(color='black', dash='dot', width=2),
                             name='Normal Distribution<br>(99.7% of its area)'), 
                  row=int(i/2)+1, col=(i%2)+1)
    
fig.update_layout(title_text='Histograms of Residuals from Different Models')
fig.update_xaxes(range=[-2,2])
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
```

### 13. The "short" and "long" wait times are not "before and after" measurements so there are not natural pairs on which to base differences on which to do a "one sample" (paired differences) *hypothesis test*; but, we can do "two sample" hypothesis testing using a *permuation test*, or create a 95% *bootstrap confidence interval* for the difference in means of the two populations. 

### (A) Do a permuation test $\;H_0: \mu_{\text{short}}=\mu_{\text{long}} \; \text{ no difference in duration between short and long groups}$ by "shuffling" the labels
### (B) Create a 95% bootstrap confidence interval  by repeatedly bootstrapping within each group and applying *np.quantile(bootstrapped_mean_differences, [0.025, 0.975])* to the collection of differences between the sample means.    
### (a) Explain how the sampling approaches work for the two simulations.
### (b) Compare and contrast these two methods with the *indicator variable* based model approach used in Question 10, explaining how they're similar and different.<br>
    
<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> _You'll need to create `for` loops for repeated (shuffling simulation) **permutation** and (subgroup) **bootstrapping**, where_
>
> - _"shuffling" for **permutation testing** is done like this `old_faithful.assign(kind_shuffled=old_faithful['kind'].sample(n=len(old_faithful), replace=False).values)#.groupby('kind').size()`; then, the **mean difference statistic** is then calculated using `.groupby('kind_shuffled')['duration'].mean().iloc[::-1].diff().values[1]` (so the **observed statistic** is `old_faithful.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]`_
> 
>
> - _"two sample" **bootstrapping** is done like this `old_faithful.groupby('kind').apply(lambda x: x.sample(n=len(x), replace=True)).reset_index(drop=True)#.groupby('kind').size()`; then, the **bootstrapped mean difference statistic** is then calculated using `.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]` (like the **observed statistic** except this is applied to the **bootstrapped** resampling of `old_faithful`)_
> ---
> 
> _Don't forget to ask for summaries of all your different ChatBot sessions and organize and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatBot) But if you're using the STA130 custom NBLM ChatBot, you'll only be able to ask for summaries, of course!_
</details>

### 14. Have you reviewed the course wiki-textbook and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

>  _Here is the link of [wiki-textbook](https://github.com/pointOfive/stat130chat130/wiki) in case it gets lost among all the information you need to keep track of_  : )
>
> _Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)_
</details>    

## Recommended Additional Useful Activities [Optional]

The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 

<br>
<details class="details-example"><summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>

### Ethical Professionalism Considerations
    
The TUT and HW both addressed some of the assumptions used in **Simple Linear Regression**. The **p-values** provided by `statsmodels` via `smf.ols(...).fit()` depend on these assumptions, so if they are not (at least approximately) correct, the **p-values** (and any subsequent claims regarding the "evidience against" the **null hypothesis**) are not reliable. In light of this consideration, describe how you could diagnostically check the first three assumptions (given below) when using analyses based on **Simple Linear regression** model. From an Ethical and Professional perspective, do you think doing diagnostic checks on the assumptions of a **Simple Linear regression** model is something you can and should do whenever you're doing this kind of analysis? 
            
> The first three assumptions associated with the **Simple Linear regression** model are that
> 
> - the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
> - the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
> - the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
> 
>    and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that<br><br>
> - the $x_i$ **predictor variable** is **measured without error**
> - and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
> - and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)
    
</details>

<details class="details-example"><summary style="color:blue"><u>Current Course Project Capability Level</u></summary>

**Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times.**

Information about the course project is available on the course github repo [here](https://github.com/pointOfive/stat130chat130/tree/main/CP), including a draft [course project specfication](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F23_course_project_specification.ipynb) (subject to change). 
- The Week 01 HW introduced [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb), and the [available variables](https://drive.google.com/file/d/1ISVymGn-WR1lcRs4psIym2N3or5onNBi/view). 
- Please do not download the [data](https://drive.google.com/file/d/1mbUQlMTrNYA7Ly5eImVRBn16Ehy9Lggo/view) accessible at the bottom of the [CSCS](https://casch.org/cscs) webpage (or the course github repo) multiple times.
    
> ### NEW DEVELOPMENT<br>New Abilities Achieved and New Levels Unlocked!!!    
> **As noted, the Week 01 HW introduced the [STA130F24_CourseProject.ipynb](https://github.com/pointOfive/stat130chat130/blob/main/CP/STA130F24_CourseProject.ipynb) notebook.** _And there it instructed students to explore the notebook through the first 16 cells of the notebook._ The following cell in that notebook (there marked as "run cell 17") is preceded by an introductory section titled, "**Now for some comparisons...**", _**and all material from that point on provides an example to allow you to start applying what you're learning about Hypothesis Testing to the CSCS data**_ **using a paired samples ("one sample") framework.**
>
> **NOW, HOWEVER, YOU CAN DO MORE.** 
> - _**Now you can do "two sample" hypothesis testing without the need for paired samples.**_ All you need are two groups.
> - _**And now you can do simple linear regression modeling.**_ All you need are two columns.

### Current Course Project Capability Level

At this point in the course you should be able to do a **Simple Linear Regression** analysis for data from the Canadian Social Connection Survey data
    
1. Create and test a **null hypothesis** of no linear association "on average" for a couple of columns of interest in the Canadian Social Connection Survey data using **Simple Linear Regression**

2. Use the **residuals** of a fitted **Simple Linear Regression** model to diagnostically assess some of the assumptions of the analysis

3. Use an **indicator variable** based **Simple Linear Regression** model to compare two groups from the Canadian Social Connection Survey data

4. Compare and contrast the results of an **indicator variable** based **Simple Linear Regression** model to analyses based on a **permutation test** and a **bootstrapped confidence interval**   
    
</details>    


```python

```
