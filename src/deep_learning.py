import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import sys
sys.path.append("/home/nick/Time-Series-Regression/src")
from base import Base


def nnet(df, datetime, output, lags, forecasts, resolution, test_frac, deep):
    model = NNet()
    model.time_features(df, datetime, resolution)
    model.lag_features(df, output, lags)
    model.forecast_features(df, output, forecasts)
    model.additional_features(df, datetime, output)
    model.scale(test_frac)
    model.train(deep)
    model.predict()

    return model


class NNet(Base):
    def scale(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))
        X = train.copy().drop(columns=self.output)

        # standardize the inputs to take on values between 0 and 1
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = self.data.copy().drop(columns=self.output)
        columns = X.columns
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=columns)

        Y = self.data.copy()[self.output]
        self.data = pd.concat([Y, X], axis="columns")

    def train(self, deep):
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        if deep:
            layer = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
        else:
            layer = [128, 128]

        for out in self.output:
            X = train.copy().drop(columns=self.output)
            Y = train.copy()[[out]]

            model = MLPRegressor(
                max_iter=1000,
                hidden_layer_sizes=layer,
                learning_rate_init=0.001,
                learning_rate="adaptive",
                batch_size=16,
                random_state=42,
            )
            model.fit(X, Y.to_numpy().ravel())

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
