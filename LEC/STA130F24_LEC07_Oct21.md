# STA130 LEC 07 (Oct 21)

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

