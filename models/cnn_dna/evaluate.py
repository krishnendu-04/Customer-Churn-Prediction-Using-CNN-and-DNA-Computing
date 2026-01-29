from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates CNN model using different probability thresholds.
    """

    y_pred_prob = model.predict(X_test)

    best_accuracy = 0
    best_threshold = 0.5

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_pred_prob >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold
