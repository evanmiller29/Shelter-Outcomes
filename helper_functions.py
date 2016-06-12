def data_import(df1, df2):
    
    import numpy as np
    
    print ('Running feature extraction process..')
    df = (df1.append(df2)
         .rename(columns=str.lower))
    
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
    
    def get_mix(x):
        x = str(x)
        if x.find('Mix') >= 0: return 'mix'
        return 'not'

    df['mix'] = df.breed.apply(get_mix)
    
    
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
    
    # Creating some more date variables

    from datetime import datetime

    df['datetime'] = pd.to_datetime(df.datetime)
    df['year'] = df['datetime'].map(lambda x: x.year).astype(str)
    df['month'] = df['datetime'].map(lambda x: x.month).astype(str)
    df['wday'] = df['datetime'].map(lambda x: x.dayofweek).astype(str)
    df['hour'] = df['datetime'].map(lambda x: x.hour).astype(str)
    
    def has_name(x):
        if x == 'Nameless': return 0
        else: return 1
    
    df['hasname'] = df['name'].map(has_name)
    
    print ('Dropping unused variables..')
    
    drop_cols = ['animalid', 'datetime', 'name', 'ageuponoutcome', 'sexuponoutcome', 'outcomesubtype']

    df.drop(drop_cols, axis=1, inplace=True)

    df['mix'] = df['breed'].str.contains('Mix').astype(int)

    df['color_simple'] = df.color.str.split('/| ').str.get(0)
    df.drop(['breed', 'color'], axis = 1 , inplace = True)
    
    # Using mean imputation of missing values. Can build on if necessary
    
    df['ageinyears'] = df.ageinyears.fillna(df.ageinyears.mean())
    
    # Just using training data for model building
    
    return(df)
    
def prep_data(dataframe, type):
    
    df = dataframe.copy()
    df.drop('id', axis = 1, inplace = True)
    df = df.loc[df.type == type,:]
    df.drop('type', axis = 1, inplace = True)
    
    # Encoding labels
    print ('Encoding labels of the outcome variable..')
    
    y = df['outcometype'].values
    
    if type == 'test':
        df['color_simple_Ruddy'] = 0
        df['hour_5'] = 0
    
    if type == 'train':
        df['hour_3'] = 0
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    
    le.fit(y)
    
    y = le.transform(y)
    X = df
    X.drop(['outcometype'], axis=1, inplace=True)
    
    print ('Using one hot encoding for predictor variables..')
    X = pd.get_dummies(X)
    
    X_cols = X.columns
    from sklearn.preprocessing import Imputer

    #Imputing missing values
    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(X)

    X = imp.transform(X)
    
    return(X, y, le, X_cols)

def col_check(train_cols, test_cols):
    cols_equal = list(set(train_cols) - set(test_cols)) 

    if not cols_equal:
        print ('Columns are the same!!')
    else:
        print ('Columns are not the same..')
           
def predict_output(clf, file):

    print ('Predicting outcomes..')
    y_pred = clf.predict_proba(X_test)

    df = pd.DataFrame(y_pred)
    df.columns = le_train.classes_
    ID = pd.Series(range(1, y_pred.shape[0] + 1))
    submission = pd.concat([ID, df], axis=1)
    submission.rename(columns={0:'ID'}, inplace=True)
    submission['ID'] = submission['ID'].astype(int)
    
    if (y_pred.shape[0] == 11456):
        print ('Correct number of rows')
        print ('Saving to CSV..')
        
        import time
        
        file_path = './Submissions/'
        current_time  = time.strftime("%d_%m_%Y_%H_%M")
        file_name = file_path + file + '_' + current_time + '.csv'
        
        submission.to_csv(file_name, index=False)
        print ('Done!')
        
    else:
        print ('Incorrect number of rows, please check')