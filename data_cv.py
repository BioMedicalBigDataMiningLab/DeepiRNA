import data_parameters as par

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# K fold cross-validation
cv_X_train = []
cv_X_validation = []
cv_y_train = []
cv_y_validation = []


# Split train data and test data
def split_data(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=par.test_size,
                                                        random_state=seed,
                                                        shuffle=True)

    return X_train, X_test, y_train, y_test


# CV
def cv(X_train, y_train):
    kf = KFold(n_splits=par.cv)
    for train, test in kf.split(X_train):
        cv_X_train.append(X_train[train])
        cv_X_validation.append(X_train[test])
        cv_y_train.append(y_train[train])
        cv_y_validation.append(y_train[test])

    return cv_X_train, cv_X_validation, cv_y_train, cv_y_validation
