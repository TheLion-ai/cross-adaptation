from typing import Union, List, Optional

import pandas as pd
import numpy as np
from adapt.instance_based import KMM, TrAdaBoost, KLIEP
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier  


def adapt_domains(data: List[pd.DataFrame],
                  da_method: Union[KMM, TrAdaBoost, KLIEP],
                  da_parms: dict,
                  classifier: Union[LogisticRegression, GaussianNB, KNeighborsClassifier, SVC, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier, XGBClassifier],
                  classifier_params: dict,
                  random_state: int=42) -> List[pd.DataFrame]:
    dfs = pd.concat(data, join='outer', axis=0)
    dfx = []
    scaler = StandardScaler()
    scaler.fit(dfs)
    dump(scaler, 'std_scaler.bin', compress=True)

    for idx, dfs in enumerate(data):
        Xs, ys = dfs.loc[:, dfs.columns != 'target'], dfs['target']
        Xs = scaler.transform(Xs)

        target = data.copy()
        target.pop(idx)
        dft = pd.concat(target, join='outer', axis=0)
        dft = dft.sample(frac=1).reset_index(drop=True)
        Xt, yt = dft.loc[:, dft.columns != 'target'], dft['target']
        Xt = scaler.transform(Xt)

        estimator = classifier(**classifier_params)
        adapt_model = da_method(estimator=estimator, Xt=Xt, yt=yt, **da_params, random_state=random_state)
        adapt_model.fit(Xs, ys)

        Xs = pd.DataFrame(Xs, columns=dfs.columns)
        ys = ys.reset_index(drop=True)
        weights = adapt_model.predict_weights()
        debiasing_idx = np.random.choice(Xs.index, len(Xs), p=weights/weights.sum())
        tf_Xs = Xs.iloc[debiasing_idx]
        tf_ys = ys.iloc[debiasing_idx]
        tf_dfs = pd.concat([tf_Xs, tf_ys], join='outer', axis=1)
        dfx.append(tf_dfs)

    return dfx