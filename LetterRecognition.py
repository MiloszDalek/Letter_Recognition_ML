import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_SET_PATH = "./letter+recognition/letter-recognition.data"
OUTPUT_DIR = "./Plots"

SAVE = False


def generate_confusion_matrix(y_predict, name):
    cm = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(df["letter"].unique()),
                yticklabels=sorted(df["letter"].unique()))
    plt.title(f"Confusion matrix {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    if SAVE:
        plt.savefig(os.path.join(OUTPUT_DIR, f"Confusion matrix {name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_importance_graph(name, model):
    result = permutation_importance(model, X_test, y_test, scoring='accuracy', n_repeats=10, random_state=42)

    # Sorting feature according to their importance
    feature_importance = result.importances_mean
    sorted_indices = feature_importance.argsort()[::-1]

    feature_names = getattr(X_train, "columns", [f"{columns[i + 1]}" for i in range(X_train.shape[1])])

    plt.figure(figsize=(10, 5))
    plt.barh(range(len(feature_importance)), feature_importance[sorted_indices], align='center')
    plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance for {name}")
    plt.gca().invert_yaxis()

    if SAVE:
        plt.savefig(os.path.join(OUTPUT_DIR, f"Feature Importance for {name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    columns = [
        "letter", "x-box", "y-box", "width", "height", "onpix", "x-bar", "y-bar",
        "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
    ]

    df = pd.read_csv(DATA_SET_PATH, header=None, names=columns)

    plt.figure(figsize=(12, 5))
    sns.countplot(x=df["letter"], order=sorted(df["letter"].unique()))
    plt.title("Letters distribution in data set")

    if SAVE:
        plt.savefig(os.path.join(OUTPUT_DIR, "Letters distribution in data set.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    X = df.drop("letter", axis=1)  # All columns except "letter"
    y = df["letter"]               # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN
    modelKNN = KNeighborsClassifier(n_neighbors=5)
    modelKNN.fit(X_train, y_train)

    y_knn_predict = modelKNN.predict(X_test)

    knn_accuracy = accuracy_score(y_test, y_knn_predict)
    print("KNN model accuracy:", knn_accuracy)

    generate_importance_graph("KNN", modelKNN)

    # Decision Tree
    modelDT = DecisionTreeClassifier()
    modelDT.fit(X_train, y_train)

    y_dt_predict = modelDT.predict(X_test)

    dt_accuracy = accuracy_score(y_test, y_dt_predict)
    print("Decision Tree model accuracy:", dt_accuracy)

    generate_importance_graph("Decision Tree", modelDT)

    # Random Forest
    modelRF = RandomForestClassifier(n_estimators=100, random_state=42)
    modelRF.fit(X_train, y_train)

    y_rf_predict = modelRF.predict(X_test)

    rf_accuracy = accuracy_score(y_test, y_rf_predict)
    print("Random Forest model accuracy:", rf_accuracy)

    generate_importance_graph("Random Forest", modelRF)

    # Naive Bayes
    modelNB = GaussianNB()
    modelNB.fit(X_train, y_train)

    y_nb_predict = modelNB.predict(X_test)

    nb_accuracy = accuracy_score(y_test, y_nb_predict)
    print("Naive Bayes model accuracy:", nb_accuracy)

    generate_importance_graph("Naive Bayes", modelNB)

    generate_confusion_matrix(y_knn_predict, "KNN")
    generate_confusion_matrix(y_dt_predict, "Decision Tree")
    generate_confusion_matrix(y_rf_predict, "Random Forest")
    generate_confusion_matrix(y_nb_predict, "Naive Bayes")

    # Comparison
    model_names = ["Random Forest", "Naive Bayes", "Decision Tree", "KNN"]
    accuracies = [rf_accuracy, nb_accuracy, dt_accuracy, knn_accuracy]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Comparison of model classifiers")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center", fontsize=12, fontweight="bold")

    if SAVE:
        plt.savefig(os.path.join(OUTPUT_DIR, "Comparison of model classifiers.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
