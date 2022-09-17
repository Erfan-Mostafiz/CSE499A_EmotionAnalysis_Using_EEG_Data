import numpy as np
import joblib
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostRegressor


with open('data_training.npy', 'rb') as fileTrain:
    # with open('data\s' + sub + '.dat', 'rb') as file:
    data_train = np.load(fileTrain)

with open('label_training.npy', 'rb') as fileTrainL:
    data_train_label = np.load(fileTrainL)

data_train = normalize(data_train)

Arousal_Train = np.ravel(data_train_label[:, [0]])
Valence_Train = np.ravel(data_train_label[:, [1]])
Domain_Train = np.ravel(data_train_label[:, [2]])
Like_Train = np.ravel(data_train_label[:, [3]])


# start emotion recognition
Val_R = AdaBoostRegressor(n_estimators=5000, learning_rate=0.01)
Val_R.fit(data_train[0:data_train.shape[0]:32], Valence_Train[0:data_train.shape[0]:32])
joblib.dump(Val_R, "model/adaboost_val_model.pkl")


Aro_R = AdaBoostRegressor(n_estimators=5000, learning_rate=0.01)
Aro_R.fit(data_train[0:data_train.shape[0]:32], Arousal_Train[0:data_train.shape[0]:32])
joblib.dump(Aro_R, "model/adaboost_aro_model.pkl")


Dom_R = AdaBoostRegressor(n_estimators=5000, learning_rate=0.01)
Dom_R.fit(data_train[0:data_train.shape[0]:32], Domain_Train[0:data_train.shape[0]:32])
joblib.dump(Dom_R, "model/adaboost_dom_model.pkl")


Lik_R = AdaBoostRegressor(n_estimators=5000, learning_rate=0.01)
Lik_R.fit(data_train[0:data_train.shape[0]:32], Like_Train[0:data_train.shape[0]:32])
joblib.dump(Lik_R, "model/adaboost_lik_model.pkl")

