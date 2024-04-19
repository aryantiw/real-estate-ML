# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
"""
## Loading Data
"""

# %%
df1 = pd.read_csv('Bengaluru_House_Data.xls')

# %%
df1.head()

# %%
df1.columns

# %%
df2 = df1[['location', 'size', 'total_sqft', 'bath', 'price']]
df2.shape

# %%
"""
## Data Cleaning
"""

# %%
df2.isnull().sum()

# %%
df2.shape

# %%
df3 = df2.dropna()
df3.isnull().sum()

# %%
"""
df3.shape
"""

# %%
"""
## Feature Engineering
"""

# %%
"""
**Add a new feature(int) for bhk**
"""

# %%
df3.head(10)

# %%
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# %%
df3.head()

# %%
"""
**Explore total sqft area**
"""

# %%
df3['total_sqft'].unique()

# %%
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# %%
df3[~df3['total_sqft'].apply(is_float)]

# %%
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

# %%
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4.head()

# %%
df4.loc[30]

# %%
df4.isnull().sum()

# %%
df4 = df4[~df4['total_sqft'].isnull()]
df4.shape

# %%
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()

# %%
df5_stats = df5['price_per_sqft'].describe()
df5_stats

# %%
df5.to_csv('bhp.csv', index = False)

# %%
"""
**Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations**
"""

# %%


# %%
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.location.value_counts(ascending = False)
location_stats

# %%
len(location_stats[location_stats > 10])

# %%
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

# %%
len(df5.location.unique())

# %%
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())

# %%
df5.head(10)

# %%
"""
## Outlier Removal
"""

# %%
df5[df5.total_sqft / df5.bhk < 300].head()

# %%
df5.shape

# %%
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape

# %%
df6.price_per_sqft.describe()

# %%
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape

# %%
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")

# %%
plot_scatter_chart(df7,"Hebbal")

# %%
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape

# %%
plot_scatter_chart(df8,"Rajaji Nagar")

# %%
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

# %%
df8.bath.unique()

# %%
plt.hist(df8.bath, rwidth=0.8)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')

# %%
df8[df8.bath>10]

# %%
df8[df8.bath>df8.bhk+2]

# %%
df9 = df8[df8.bath<df8.bhk+2]
df9.shape

# %%
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)

# %%
"""
## Use One Hot Encoding For Location
"""

# %%
dummies = pd.get_dummies(df10.location)
dummies

# %%
df11 = pd.concat([df10, dummies.drop('other', axis=1)], axis = 1)
df11.head()

# %%
df12 = df11.drop('location', axis=1)
df12.head(2)

# %%
"""
## Build a Model
"""

# %%
df12.shape

# %%
X = df12.drop('price', axis=1)
X.head(2)

# %%
Y = df12.price
print(X.shape)
print(Y.shape)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# %%
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)

# %%
from sklearn.model_selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 0)
cross_val = cross_val_score(LinearRegression(), X, Y, cv=cv)
np.mean(cross_val)

# %%
"""
## Find best models using Grid Search CV
"""

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model(X, y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'fit_intercept' : [True, False]
            }
        },
        'lasso' :{
            'model' : Lasso(),
            'params' : {
                'alpha':[1,2],
                'selection':['random', 'cyclic']
            }
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


find_best_model(X, Y)

# %%
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

# %%
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

# %%
