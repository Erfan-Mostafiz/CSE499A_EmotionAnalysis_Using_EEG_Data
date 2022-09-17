import numpy as np
import joblib
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression

with open('data_training.npy', 'rb') as fileTrain:
    data_train = np.load(fileTrain)

with open('label_training.npy', 'rb') as fileTrainL:
    data_train_label = np.load(fileTrainL)

data_train = normalize(data_train)

Arousal_Train = np.ravel(data_train_label[:, [0]])
Valence_Train = np.ravel(data_train_label[:, [1]])
Domain_Train = np.ravel(data_train_label[:, [2]])
Like_Train = np.ravel(data_train_label[:, [3]])

print("running")


Val_R = LinearRegression()
Val_R.fit(data_train[0:data_train.shape[0]:32], Valence_Train[0:data_train.shape[0]:32])
joblib.dump(Val_R, "model/LRR_val_model.pkl")


Aro_R = LinearRegression()
Aro_R.fit(data_train[0:data_train.shape[0]:32], Arousal_Train[0:data_train.shape[0]:32])
joblib.dump(Aro_R, "model/LRR_aro_model.pkl")


Dom_R = LinearRegression()
Dom_R.fit(data_train[0:data_train.shape[0]:32], Domain_Train[0:data_train.shape[0]:32])
joblib.dump(Dom_R, "model/LRR_dom_model.pkl")


Lik_R = LinearRegression(n_jobs=5)
Lik_R.fit(data_train[0:data_train.shape[0]:32], Like_Train[0:data_train.shape[0]:32])
joblib.dump(Lik_R, "model/LRR_lik_model.pkl")
