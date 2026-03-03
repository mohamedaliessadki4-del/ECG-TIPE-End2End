import argparse
from src.ml.train_knn import train_knn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--out-model", default="models/knn_model.joblib")
    args = ap.parse_args()

    train_knn(args.train, args.test, k=args.k, balance=not args.no_balance, out_model=args.out_model)


if __name__ == "__main__":
    main()
