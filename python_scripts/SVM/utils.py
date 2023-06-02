
def SVM(x, y, kernel = "linear", C=1, cv=5):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import  balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score

    ## Model
    clf = svm.SVC(kernel = kernel, C=C, random_state = 42, class_weight="balanced", probability=True)

    # Fit the classifier and perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, x, y, cv=cv, method='predict_proba')[:, 1]

    # Find optimal threshold using F1 score and the predicted probabilities
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y, y_pred_cv >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Evaluate the classifier using the optimal threshold
    y_pred = (y_pred_cv >= optimal_threshold).astype(int)

    print(f"Optimal decision threshold: {optimal_threshold:.2f}")
    print(f"Precision: {precision_score(y, y_pred):.2f}")
    print(f"Recall: {recall_score(y, y_pred):.2f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.2f}")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y, y_pred):.2f}")


    # Compute precision, recall and thresholds
    precision, recall, thresholds = precision_recall_curve(y, y_pred_cv)

    # Plot the precision-recall curve
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y, y_pred_cv)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
