from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _encode_curfew(X):
    X = X.copy()

    cf1 = ((X['date'] >= pd.to_datetime('2020/10/17')) & (X['date'] <= pd.to_datetime('2020/10/29'))) & ((X['date'].dt.hour >= 21) | (X['date'].dt.hour <= 6))
    cf2 = ((X['date'] >= pd.to_datetime('2020/12/16')) & (X['date'] <= pd.to_datetime('2021/01/15'))) & ((X['date'].dt.hour >= 20) | (X['date'].dt.hour <= 6))
    cf3 = ((X['date'] >= pd.to_datetime('2021/01/16')) & (X['date'] <= pd.to_datetime('2021/03/20'))) & ((X['date'].dt.hour >= 18) | (X['date'].dt.hour <= 6))
    cf4 = ((X['date'] >= pd.to_datetime('2021/03/21')) & (X['date'] <= pd.to_datetime('2021/04/02'))) & ((X['date'].dt.hour >= 19) | (X['date'].dt.hour <= 6))
    cf5 = ((X['date'] >= pd.to_datetime('2021/05/19')) & (X['date'] <= pd.to_datetime('2021/06/08'))) & ((X['date'].dt.hour >= 21) | (X['date'].dt.hour <= 6))
    cf6 = ((X['date'] >= pd.to_datetime('2021/06/09')) & (X['date'] <= pd.to_datetime('2021/06/20'))) & ((X['date'].dt.hour >= 23) | (X['date'].dt.hour <= 6))

    mask = cf1 | cf2 | cf3 | cf4 | cf5 | cf6
    
    
    X['curfew'] =np.where(mask, 1, 0)
    return X

def _encode_lockdown(X):
    X = X.copy()

    lk1 = (X['date'] >= pd.to_datetime('2020/10/30')) & (X['date'] <= pd.to_datetime('2020/12/15'))
    lk2 = (X['date'] >= pd.to_datetime('2021/4/3')) & (X['date'] <= pd.to_datetime('2021/5/3'))
    mask = lk1 | lk2

    X['lockdown'] =np.where(mask, 1, 0)
    return(X)

def _encode_schools_holidays(X):
    is_holidays = (
        ((X['date'] >= pd.to_datetime('2020/10/17'))
        & (X['date'] <= pd.to_datetime('2020/11/2'))) |
        ((X['date'] >= pd.to_datetime('2020/12/19'))
        & (X['date'] <= pd.to_datetime('2021/1/4'))) |
        ((X['date'] >= pd.to_datetime('2021/2/13'))
        & (X['date'] <= pd.to_datetime('2021/3/1'))) |
        ((X['date'] >= pd.to_datetime('2021/4/10'))
        & (X['date'] <= pd.to_datetime('2021/4/26'))) |
        ((X['date'] >= pd.to_datetime('2021/7/7'))
        & (X['date'] <= pd.to_datetime('2021/9/4'))) 
    )

    X['holidays'] = np.where(is_holidays, 1, 0)
    return X

def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", "u", "rr3", "ff"]].sort_values("date"), on="date"
    )

    X['t'] = X['t'].fillna(0)
    X['rr3'] = X['rr3'].fillna(0)
    X['u'] = X['u'].fillna(0)
    X['ff'] = X['ff'].fillna(0)
    
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _encode_broken(X):
    X = X.copy()

    is_broken = (
        ((X['site_name'] == '152 boulevard du Montparnasse')
        & (X['date'] >= pd.to_datetime('2021/01/26'))
        & (X['date'] <= pd.to_datetime('2021/02/24'))) 
        |
        ((X['site_name'] == '20 Avenue de Clichy')
        & (X['date'] >= pd.to_datetime('2021/05/06'))
        & (X['date'] <= pd.to_datetime('2021/07/21')))
    )

    X['broken'] = np.where(is_broken, 1, 0)
    return X

def _cyclic_hour(X):
    X = X.copy()# modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "hour_c"] = np.cos(np.pi/12 * X["hour"])
    X.loc[:, "hour_s"] = np.sin(np.pi/12 * X["hour"])
    # a way to show the cyclical ways of the hours
    return(X.drop(columns="hour"))

def get_estimator():
    # Call the merge_external_data function
    merge_external = FunctionTransformer(_merge_external_data, validate=False)

    # Call the special events functions : 
    # All need : ["date"], last one needs ["date", "site_name"]
    # This creates columns : ["curfew", "lockdown", "holiday", "broken"]
    curfew = FunctionTransformer(_encode_curfew, validate=False)
    lockdown = FunctionTransformer(_encode_lockdown, validate=False)
    holidays = FunctionTransformer(_encode_schools_holidays, validate=False)
    broken = FunctionTransformer(_encode_broken, validate=False)
    

    # Call the _encode_dates function to split the date column to several columns 
    # !! delete "date"
    date_encoder = FunctionTransformer(_encode_dates)
    
    # Call the _cyclic_hour function to split the hour column to several columns
    # !! delete "hour"
    # This creates columns : ['hour_c', 'hour_s']
    cyclic_hours = FunctionTransformer(_cyclic_hour)


    # Encode the final columns
    numeric_encoder = StandardScaler()
    numeric_cols = ['t', 'u','rr3', 'ff', 'year', 'month', 'day', 'weekday', 'hour_c', 'hour_s', 'latitude', 'longitude']
   

    evts = ["curfew", "lockdown", "holidays", "broken"]


    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_encoder, categorical_cols),
            ("numeric", numeric_encoder, numeric_cols),
            ("events", 'passthrough', evts),

        #, remainder='passthrough'
        ]
    )
    
    # Set the best parameters values for our LightBGM model

    params = {
        'colsample_bytree': 0.7, 
        'learning_rate': 0.1,
        'max_depth': 11,
        'min_child_samples': 198,
        'min_child_weight': 0.1,
        'n_estimators': 3000,
        'num_leaves': 99,
        'reg_alpha': 1, 
        'reg_lambda': 0.1,
        'subsample': 0.5
    }
    
    # Create the regressor object 
    regressor = LGBMRegressor(**params)
    
    
# Create a ColumnTransformer object to perform all encodings
    pipe = make_pipeline(
        merge_external,
        curfew,
        lockdown, 
        holidays, 
        broken,
        date_encoder,
        cyclic_hours,
        preprocessor, 
        regressor
    )

    return pipe
