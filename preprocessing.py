
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[286]:

train = pd.read_csv('train.csv')
train['type'] = 'train'

test = pd.read_csv('test.csv')
test['type'] = 'test'

df = (train.append(test)
         .rename(columns=str.lower))


# In[287]:

# functions to get new parameters from the column
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'

df['sex'] = df.sexuponoutcome.apply(get_sex)
df['neutered'] = df.sexuponoutcome.apply(get_neutered)


# In[288]:

def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'mix'
    return 'not'

df['mix'] = df.breed.apply(get_mix)


# In[289]:

def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return np.nan
    age = int(x.split()[0])
    if x.find('year') > -1: return age 
    if x.find('month')> -1: return age / 12.
    if x.find('week')> -1: return age / 52.
    if x.find('day')> -1: return age / 365.
    else: return np.nan
    
df['ageinyears'] = df.ageuponoutcome.apply(calc_age_in_years)


# In[290]:

# Creating some more date variables

from datetime import datetime

df['datetime'] = pd.to_datetime(df.datetime)
df['year'] = df['datetime'].map(lambda x: x.year).astype(str)
df['wday'] = df['datetime'].map(lambda x: x.dayofweek).astype(str)


# In[291]:

drop_cols = ['animalid', 'datetime', 'name', 'ageuponoutcome', 'sexuponoutcome', 'id', 'outcomesubtype']

df.drop(drop_cols, axis=1, inplace=True)


# In[292]:

df['mix'] = df['breed'].str.contains('Mix').astype(int)


# In[293]:

df['color_simple'] = df.color.str.split('/| ').str.get(0)
df.drop(['breed', 'color'], axis = 1 , inplace = True)


# In[294]:

y = df['outcometype']
X = df
X.drop(['outcometype', 'type'], axis=1, inplace=True)


# In[295]:

X.head()
text_vars = ['animal_type', 'sex', 'neutered', 'color_simple']
X = pd.get_dummies(X)


# In[296]:

pd.isnull(X['ageinyears']).value_counts()

# Wanting to predict those 24 null values using a random forest


# In[297]:

X['age_null'] = pd.isnull(X['ageinyears'])

map_vect = {True:1, False:0}

X['age_null'] = X['age_null'].map(map_vect)


# In[298]:

X_NN = X.query('age_null == 0')
y_NN = X_NN['ageinyears']
X_NN.drop(['ageinyears'], axis = 1, inplace = True)


# In[299]:

X_NN.head()


# In[301]:

y = y_NN.as_matrix()
X = X_NN.as_matrix()

from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE = 123

clf = RandomForestClassifier()
clf = clf.fit(X, y)

