import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from .load_mitbih import load_mitbih
from .balance_mitbih import balance_by_upsampling


def split_xy(df, label_col: int = 187):
    X = df.iloc[:, :186].values
    y = df[label_col].values
    return X, y


def train_knn(train_csv: str, test_csv: str, k: int = 3, balance: bool = True, out_model: str = "models/knn_model.joblib"):
    train_df, test_df = load_mitbih(train_csv, test_csv)

    if balance:
        train_df = balance_by_upsampling(train_df, label_col=187, n_per_class=20000)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, out_model)
    print(f"Saved model: {out_model}")

    return acc


if __name__ == "__main__":
    # Example usage (edit paths)
    train_knn("data/raw/mitbih_train.csv", "data/raw/mitbih_test.csv")
