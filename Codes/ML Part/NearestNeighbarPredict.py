import joblib
import numpy as np
from sklearn.preprocessing import normalize


def predict(data_test, data_test_label, model):
    """
    arguments:  data_test: testing dataset
                data_test_label: testing dataset label
                model: scikit-learn model

    return:     void
    """
    output = model.predict(data_test[0:data_test.shape[0]:32])
    label = data_test_label[0:data_test.shape[0]:32]

    success = 0

    for i in range(len(label)):
        # a good guess
        if output[i] > 5 and label[i] > 5:
            success = success + 1
        elif output[i] < 5 and label[i] < 5:
            success = success + 1

    print("classification accuracy: ", success / len(label), success, len(label))


# load testing dataset
with open('data_testing.npy', 'rb') as fileTest:
    data_test = np.load(fileTest)

with open('label_testing.npy', 'rb') as fileTestL:
    data_test_label = np.load(fileTestL)

data_test = normalize(data_test)

Arousal_Test = np.ravel(data_test_label[:, [0]])
Valence_Test = np.ravel(data_test_label[:, [1]])
Domain_Test = np.ravel(data_test_label[:, [2]])
Like_Test = np.ravel(data_test_label[:, [3]])


# load trained model
Val_R = joblib.load("model/KNR_val_model.pkl")
Aro_R = joblib.load("model/KNR_aro_model.pkl")
Dom_R = joblib.load("model/KNR_dom_model.pkl")
Lik_R = joblib.load("model/KNR_lik_model.pkl")


# start classification
print("-----Valence-----")
predict(data_test, Valence_Test, Val_R)

print("-----Arousal-----")
predict(data_test, Arousal_Test, Aro_R)

print("-----Domain-----")
predict(data_test, Domain_Test, Dom_R)

print("-----Like-----")
predict(data_test, Like_Test, Lik_R)
