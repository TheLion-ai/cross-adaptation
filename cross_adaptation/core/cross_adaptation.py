from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler


class Adapter:
    def __init__(
        self,
        train_data: Dict[str, pd.DataFrame],
        adapt_model: object,
        estimator: object,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        """Cross-adaptation method for domain adaptation

        Args:
            train_data (Dict[str, pd.DataFrame]): the data to be adapted, each dataframe should have a column called target
            adapt_model (object): the method for domain adaptation
            estimator (object): estimator used for the task
            scaler (Optional[StandardScaler], optional): Scaler used for scaling the data. Defaults to None.

        Returns:
            List[pd.DataFrame]: a list of adapted dataframes
        """
        self.train_data = train_data
        self.adapt_model = adapt_model
        self.estimator = estimator
        self.scaler = scaler or self.init_scaler()

        self.__validate_input()

    def __validate_input(self):
        # Validate adapt_model
        if (
            getattr(self.adapt_model, "fit", None) is None
            or getattr(self.adapt_model, "predict_weights", None) is None
        ):
            raise ValueError("da_method must implement fit and predict_weights method")
        # Validate estimator
        if (
            getattr(self.estimator, "fit", None) is None
            or getattr(self.estimator, "predict", None) is None
        ):
            raise ValueError("estimator must implement fit and predict method")
        # validate train_data
        for dataset_name, data in self.train_data.items():
            if "target" not in data.columns:
                raise ValueError(
                    f"dataset {dataset_name} must have a column called target"
                )
        # validate scaler
        if (
            getattr(self.scaler, "fit", None) is None
            or getattr(self.scaler, "transform", None) is None
        ):
            raise ValueError("scaler must implement fit and transform method")

    def init_scaler(self) -> StandardScaler:
        """Create a scaler for the data if no scaler is provided.
        The scaler is saved in a file called scaler.bin"""
        scaler = StandardScaler()
        dfs = pd.concat(self.train_data.values(), join="outer", axis=0)
        X = dfs.loc[:, dfs.columns != "target"]
        scaler.fit(X)
        dump(scaler, "scaler.bin", compress=True)
        return scaler

    def adapt(self):
        """Cross-adapt the data using the adapt_model"""
        self.train_data_x = {}
        # Adapt each dataset based on the others
        for dataset_name, dfs in self.train_data.items():
            Xs, ys = dfs.loc[:, dfs.columns != "target"], dfs["target"]
            Xs = self.scaler.transform(Xs)

            # Adapt source dataset based on the target (all other datasets)
            target = self.train_data.copy()
            del target[dataset_name]
            dft = pd.concat(list(target.values()), join="outer", axis=0)
            dft = dft.sample(frac=1).reset_index(drop=True)
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            self.adapt_model.Xt = Xt
            self.adapt_model.yt = np.array(yt)

            self.adapt_model.fit(Xs, np.array(ys))

            Xs = pd.DataFrame(Xs, columns=dfs.columns.drop("target"))
            ys = ys.reset_index(drop=True)
            weights = self.adapt_model.predict_weights()
            # The adaptation is done by sampling the source dataset based on the weights
            debiasing_idx = np.random.choice(
                Xs.index, len(Xs), p=weights / weights.sum()
            )
            tf_Xs = Xs.iloc[debiasing_idx]
            tf_ys = ys.iloc[debiasing_idx]
            tf_dfs = pd.concat([tf_Xs, tf_ys], join="outer", axis=1)
            self.train_data_x[dataset_name] = tf_dfs
        return self.train_data_x

    def calc_baseline(
        self, test_data: Dict[str, pd.DataFrame], metrics: List[Callable]
    ) -> Dict[str, float]:
        """Calculate the baseline training the model on not adapted data"""
        dfs = pd.concat(list(self.train_data.values()), join="outer", axis=0)
        Xs, ys = dfs.loc[:, dfs.columns != "target"], dfs["target"]
        Xs = self.scaler.transform(Xs)
        self.estimator.fit(Xs, np.array(ys))
        results = {}
        for dataset_name, dft in test_data.items():
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            y_pred = self.estimator.predict(Xt)
            for metric in metrics:
                results[f"baseline_{dataset_name}_{metric.func.__name__}"] = metric(
                    yt, y_pred
                )
        dump(self.estimator, "baseline_estimator.bin", compress=True)
        return results

    def test(
        self, test_data: Dict[str, pd.DataFrame], metrics: List[Callable]
    ) -> Dict[str, float]:
        """Test the estimator on the adapted data

        Args:
            dfx (List[pd.DataFrame]): the adapted data
            test_data (List[pd.DataFrame]): the test data
            metrics (List[Callable], optional): the metrics to be used for evaluation. Each metric should ba a callable that accepts 2 array like objects of true and pred values ad return a float.
        return:
            Dict[str, float]: dictionary of metrics and values"""
        dfx = pd.concat(list(self.train_data_x.values()), join="outer", axis=0)
        dfx = dfx.sample(frac=1).reset_index(drop=True)
        Xs, ys = dfx.loc[:, dfx.columns != "target"], dfx["target"]
        Xs = self.scaler.transform(Xs)
        self.estimator.fit(Xs, np.array(ys))
        results = {}
        for dataset_name, dft in test_data.items():
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            y_pred = self.estimator.predict(Xt)
            for metric in metrics:
                results[f"{dataset_name}_{metric.func.__name__}"] = metric(yt, y_pred)
        dump(self.estimator, "estimator.bin", compress=True)
        return results

    def compare(
        self,
        results: Dict[str, float],
        baseline_results: Dict[str, float],
        metrics: List[object],
        test_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compare the results of the baseline and the adapted model

        Args:
            baseline_results (Dict[str, float]): the results of the baseline model
            metrics (List[Callable], optional): the metrics to be used for evaluation. Each metric should ba a callable that accepts 2 array like objects of true and pred values ad return a float.
            test_data (Dict[str, pd.DataFrame]): the test data
        return:
            Dict[str, float]: dictionary of metrics and values
        """

        compared_results = {}
        for metric in metrics:
            for dataset_name in test_data.keys():
                metric_name = metric.func.__name__
                key = f"{dataset_name}_{metric_name}"
                results_metric = list(
                    filter(lambda item: key in item[0], results.items())
                )[0][1]
                baseline_results_metric = list(
                    filter(lambda item: key in item[0], baseline_results.items())
                )[0][1]
                compared_results[f"compared_{key}"] = (
                    results_metric - baseline_results_metric
                )
        return compared_results
