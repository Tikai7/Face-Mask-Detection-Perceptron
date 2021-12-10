import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def model(X, W, b):
    Z = X.dot(W)+b  # --- Z = X*W + b notre model
    A = 1/(1+np.exp(-Z))  # --- sigmoid
    return A


def init(X):
    # --vecteur W qui contient autants de params que de variables X
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def cost_Log_Loss(A, y, epsilon=1e-15):
    return 1/len(y) * np.sum(-y*np.log(A+epsilon)-(1-y)*np.log(1-A+epsilon))


def gradients(A, X, y):
    dw = 1/len(y) * np.dot(X.T, A-y)
    db = 1/len(y) * np.sum(A-y)
    return (dw, db)


def update(dw, db, W, b, learning_rate):
    W = W-learning_rate*dw
    b = b-learning_rate*db
    return (W, b)


def predict(X, W, b):
    A = model(X, W, b)
    return (A >= 0.5, A)


def Ann(X, y, X_test, y_test, learning_rate=0.1, n_iteration=10000):

    W, b = init(X)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for i in tqdm(range(n_iteration)):
        A = model(X, W, b)
        if i % 10:
            train_loss.append(cost_Log_Loss(A, y))
            y_pred_, proba = predict(X, W, b)
            train_acc.append(accuracy_score(y, y_pred_))

            A_test = model(X_test, W, b)
            test_loss.append(cost_Log_Loss(A_test, y_test))
            y_pred_test, proba_ = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred_test))

        dw, db = gradients(A, X, y)
        W, b = update(dw, db, W, b, learning_rate)

    y_pred, proba = predict(X, W, b)
    print(accuracy_score(y, y_pred))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, c='red', label="train_acc")
    plt.plot(test_acc, c='blue', label="test_acc")
    plt.legend()
    plt.savefig('train_test_result.png')
    plt.show()

    return (W, b)  # -- les parametres que le ANN a appris
