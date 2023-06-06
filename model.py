import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Cirrhosis.csv")
columns=df.columns
#Filling the null values with median
df.select_dtypes(include=(['int64', 'float64'])).isna().sum()
df_num_col = df.select_dtypes(include=(['int64', 'float64'])).columns
for c in df_num_col:
    df[c].fillna(df[c].median(), inplace=True)
#filling the string values with mode
df_cat_col = df.select_dtypes(include=('object')).columns
for c in df_cat_col:
    df[c].fillna(df[c].mode().values[0], inplace=True)
df['Stage'] = np.where(df['Stage'] ==4,1,0)
# replacing catagorical data with intgers.
df['Sex'] = df['Sex'].replace({'M':0, 'F':1})                                # Male : 0 , Female :1
df['Ascites'] = df['Ascites'].replace({'N':0, 'Y':1})                        # N : 0, Y : 1
df['Drug'] = df['Drug'].replace({'D-penicillamine':0, 'Placebo':1})          # D-penicillamine : 0, Placebo : 1
df['Hepatomegaly'] = df['Hepatomegaly'].replace({'N':0, 'Y':1})              # N : 0, Y : 1
df['Spiders'] = df['Spiders'].replace({'N':0, 'Y':1})                        # N : 0, Y : 1
df['Edema'] = df['Edema'].replace({'N':0, 'Y':1, 'S':-1})                    # N : 0, Y : 1, S : -1
df['Status'] = df['Status'].replace({'C':0, 'CL':1, 'D':-1})                 # 'C':0, 'CL':1, 'D':-1
df=df.drop(['ID'],axis=1)
columns=['Sex','Ascites','Spiders','Edema','Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin']

def drop_outliers_zscore(dfcopy: pd.DataFrame, cols, threshold: int = 3, inplace: bool = False):
    '''
    input:  dfcopy            ==> the dataframe that contains outliers
            cols  list / str  ==> list of strings (names of columns that have outliers)
            inplace           ==> if True,  method will edit the original dataframe
                                if False, method will return the new dataframe
            threshold         ==> maximum and minimun threshold of zscore

    output: df:        ==>clean dataframe
    this method drops outliers from data using zscore method
    '''
    if inplace:
        global df
    else:
        df = dfcopy.copy()

    def drop_col(df_, col):

        mean, std = np.mean(df_[col]), np.std(df_[col])
        df_['is_outlier'] = df_[col].apply(lambda x: np.abs((x - mean) / std) > threshold)
        outliers_idx = df_.loc[df_['is_outlier']].index
        df_ = df_.drop(outliers_idx, axis=0)

        df = df_.drop('is_outlier', axis=1)
        return df

    if type(cols) == str:
        df = drop_col(df, cols)
    elif type(cols) == list:
        for col in cols:
            df = drop_col(df, col)
    else:
        raise ValueError('Pass neither list nor string in {Cols}')

    if inplace:

        dfcopy = df
    else:
        return df


THRESHOLD = 4
num_outlier_records = df.shape[0] - drop_outliers_zscore(df, columns, threshold=THRESHOLD).shape[0]
print(f'Number of Outliers {num_outlier_records}, with threshold, with threshold {THRESHOLD}')

drop_outliers_zscore(df, columns, threshold=THRESHOLD, inplace=True)
df=df.drop(['Ascites'], axis=1)
x = df.drop(['Status','N_Days','Stage'], axis=1)
y = df.pop('Stage')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/5,random_state=0)
RF = RandomForestClassifier(n_estimators = 100, random_state = 0)
RF.fit(x_train,y_train)
pickle.dump(RF,open("model.pkl","wb"))
