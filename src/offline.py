import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("/home/nick/Time-Series-Regression/src")
from base import Base


def batching(df, datetime, output, lags, forecasts, resolution, test_frac):
    model = Batching()
    model.time_features(df, datetime, resolution)
    model.lag_features(df, output, lags)
    model.forecast_features(df, output, forecasts)
    model.additional_features(df, datetime, output)
    model.train(test_frac)
    model.predict()

    return model


class Batching(Base):
    def train(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        for out in self.output:
            X = train.copy().drop(columns=self.output)
            Y = train.copy()[[out]]

            model = XGBRegressor(
                booster="gbtree",
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X, Y)

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.output:
            X = test.copy().drop(columns=self.output)
            Y = test.copy()[[out]]

            model = self.model[out]
            y_pred = model.predict(X)
            y_true = Y.to_numpy().ravel()

            metric = mean_squared_error(
                y_true=y_true, 
                y_pred=y_pred, 
                squared=False,
            )
            metric = f"RMSE: {round(metric, 6)}"

            predictions = pd.DataFrame({
                "Actual": y_true,
                "Predicted": y_pred,
            })

            self.metric[out] = metric
            self.predictions[out] = predictions
