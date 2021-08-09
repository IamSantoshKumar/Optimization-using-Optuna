from sklearn import model_selection
import xgboost
from functools import partial
import optuna
from sklearn.metrics import mean_squared_error

def optimize(trial,x,y):
    #criterion=trial.suggest_categorical('criterion',['gini','entropy'])
    n_estimators=trial.suggest_int('n_estimators',1000,10000)
    max_depth=trial.suggest_int('max_depth',2,15)
    min_child_weight=trial.suggest_int('min_child_weight',10,500)
    learning_rate=trial.suggest_loguniform('learning_rate',0.01, 1.0)
    gamma_=trial.suggest_loguniform('gamma_',0.01, 1.0)
    alpha=trial.suggest_loguniform('alpha',0.01, 1.0)
    subsample=trial.suggest_float('subsample',0.01,1.0,log=True)
    colsample_bytree=trial.suggest_float('colsample_bytree',0.01,1.0,log=True)
    
    
    
    xgb_params= {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "reg_alpha" : alpha,
        "gamma" :gamma_,
        "min_child_weight": min_child_weight,
        "n_jobs": 4,
        "seed": 2001,
        'tree_method': "gpu_hist",
        "gpu_id": 0,
    }
 
    rmse_score = []
        
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
    oof = np.zeros((x.shape[0],))
    test_preds = 0

    for f, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y))):
        df_train, df_val = x.iloc[train_idx][columns], x.iloc[val_idx][columns]
        train_target, val_target = y[train_idx], y[val_idx]
        test_ =xgb.DMatrix(test_df[columns])
        df_train_ = xgb.DMatrix(df_train, label=train_target)
        df_val_ = xgb.DMatrix(df_val, label=val_target)
            
        model = xgb.train(xgb_params, df_train_, n_estimators)
            
        oof_tmp = model.predict(df_val_)
        test_tmp = model.predict(test_)
        oof[val_idx] = oof_tmp
        test_preds += test_tmp/10
        rmse  = mean_squared_error(oof_tmp, val_target, squared=False)
        rmse_score.append(rmse)
      
    return np.mean(rmse_score)

 

if __name__=='__main__':
    optimize_func=partial(optimize,x=train_df, y=train_df['target'].values)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimize_func,n_trials=250)