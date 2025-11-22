import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from utils import (
    load_dataset,
    get_model_path,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_mae_vs_training_size,
)


def main():

    # LOAD FULL DATASET

    df = load_dataset("perovskite_bandgap_5000rows.csv")

    X = df.drop("BandGap", axis=1)
    y = df["BandGap"]


    # TRAIN–TEST SPLIT (MAIN MODEL)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # MAIN MODEL DEFINITION

    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )


    # TRAIN MAIN MODEL

    model.fit(X_train, y_train)


    # PREDICTIONS + METRICS

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Baseline model: always predicts mean of training target
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    improvement_percent = ((baseline_mae - mae) / baseline_mae) * 100


    print("       MAIN MODEL PERFORMANCE      ")
    print("===================================")
    print(f"Dataset Size            : {len(df)} rows")
    print(f"MAE                     : {mae:.4f}")
    print(f"RMSE                    : {rmse:.4f}")
    print(f"R² Score                : {r2:.4f}")
    print(f"Baseline MAE            : {baseline_mae:.4f}")
    print(f"Improvement vs Baseline : {improvement_percent:.2f}%")


    # -----------------------------
    # SAVE MAIN MODEL
    # -----------------------------
    model_path = get_model_path("model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[SAVED] Model saved at → {model_path}")


    # GRAPH 1: REAL vs PREDICTED

    plot_feature_importance(model, X.columns)
    plot_pred_vs_actual(y_test, y_pred)


    # GRAPH 2: MAE vs TRAINING SIZE

    train_sizes = [200, 500, 1000, 2000, 3000, 4000]
    mae_values = []

    for size in train_sizes:
        # Take the first `size` samples
        X_sub = X.iloc[:size]
        y_sub = y.iloc[:size]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )

        temp_model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_tr, y_tr)

        y_temp_pred = temp_model.predict(X_te)
        temp_mae = mean_absolute_error(y_te, y_temp_pred)

        mae_values.append(temp_mae)

    plot_mae_vs_training_size(train_sizes, mae_values)

    print("[DONE] All graphs generated inside /results/")


if __name__ == "__main__":
    main()
