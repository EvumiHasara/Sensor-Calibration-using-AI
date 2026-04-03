import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def generate_time_windows(data_in, target_in, step_count=24):
    """Transforms flat data into temporal windows (3D) for Recurrent Networks."""
    windowed_x, windowed_y = [], []
    for i in range(len(data_in) - step_count):
        windowed_x.append(data_in[i:(i + step_count)])
        windowed_y.append(target_in[i + step_count])
    return np.array(windowed_x), np.array(windowed_y)


def execute_model_cycle(tag, model_obj, x_train, y_train, x_test, y_test, out_scaler, run_epochs=100, cb_list=None):
    print(f"Executing training for: {tag}...")
    model_obj.compile(optimizer='adam', loss='mse')

    tracking = model_obj.fit(
        x_train, y_train,
        epochs=run_epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=cb_list,
        verbose=0
    )

    raw_forecast = model_obj.predict(x_test, verbose=0)

    final_forecast = np.expm1(out_scaler.inverse_transform(raw_forecast))
    actual_values = np.expm1(out_scaler.inverse_transform(y_test))

    fit_score = r2_score(actual_values, final_forecast)
    print(f"--> {tag} achieved R2: {fit_score:.4f}\n")

    return tracking, actual_values, final_forecast, fit_score


def main():
    # 1. Load Source
    csv_source = os.path.join('data', 'Measurement_info.csv')
    if not os.path.exists(csv_source):
        print(f"Error: {csv_source} not found.")
        return

    raw_data = pd.read_csv(csv_source)

    # 2. Engineering & Feature Extraction
    pm_subset = raw_data[raw_data['Item code'] == 9].copy()
    pm_subset['Measurement date'] = pd.to_datetime(pm_subset['Measurement date'])


    # Align Ground Truth and Drift Sensor
    ref_truth = pm_subset[pm_subset['Instrument status'] == 0] \
        .groupby('Measurement date')['Average value'].mean().reset_index()

    drift_obs = pm_subset[pm_subset['Instrument status'] == 1][
        ['Measurement date', 'Average value']
    ]

    merged_df = pd.merge(
        ref_truth, drift_obs,
        on='Measurement date',
        suffixes=('_GT', '_DR')
    ).dropna()

    # --- EXTRA VISUALIZATIONS ---

    # 1. Distribution of PM2.5 values
    plt.figure(figsize=(12, 5))
    plt.hist(merged_df['Average value_GT'], bins=50, alpha=0.7)
    plt.title('Distribution of PM2.5 Values (Ground Truth)')
    plt.xlabel('PM2.5 Concentration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 2. Time Series (First 500 Hours)
    ts_sample = merged_df.sort_values('Measurement date').head(500)

    plt.figure(figsize=(14, 6))
    plt.plot(ts_sample['Measurement date'], ts_sample['Average value_GT'],
             label='Ground Truth', linewidth=1)

    plt.plot(ts_sample['Measurement date'], ts_sample['Average value_DR'],
             label='Drift Sensor', linewidth=1, alpha=0.7)

    plt.title('Time Series Structure (First 500 Hours)')
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Concentration')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Normalization
    target_log = np.log1p(merged_df[['Average value_GT']].values)

    input_scaler, output_scaler = MinMaxScaler(), MinMaxScaler()
    inputs_scaled = input_scaler.fit_transform(
        merged_df[['Average value_DR']].values
    )
    target_scaled = output_scaler.fit_transform(target_log)

    # 4. Sequence Generation
    LOOKBACK = 12
    features_3d, target_seq = generate_time_windows(
        inputs_scaled, target_scaled, step_count=LOOKBACK
    )

    # 5. Data Partitioning
    train_x_3d, test_x_3d, train_y, test_y = train_test_split(
        features_3d, target_seq, test_size=0.2, random_state=42
    )

    train_x_2d = train_x_3d.reshape((train_x_3d.shape[0], -1))
    test_x_2d = test_x_3d.reshape((test_x_3d.shape[0], -1))

    performance_log = {}
    halt_logic = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # --- ITERATION 1 ---
    print("\n" + "="*40 + "\nITERATION 1: Baseline\n" + "="*40)

    flat_net_v1 = tf.keras.Sequential([
        layers.Input(shape=(train_x_2d.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    temporal_net_v1 = tf.keras.Sequential([
        layers.Input(shape=(LOOKBACK, train_x_3d.shape[2])),
        layers.LSTM(32),
        layers.Dense(1)
    ])

    performance_log['Iter1_MLP'] = execute_model_cycle(
        "Base_MLP", flat_net_v1,
        train_x_2d, train_y,
        test_x_2d, test_y,
        output_scaler, run_epochs=50
    )

    performance_log['Iter1_LSTM'] = execute_model_cycle(
        "Base_LSTM", temporal_net_v1,
        train_x_3d, train_y,
        test_x_3d, test_y,
        output_scaler, run_epochs=50
    )

    # --- ITERATION 2 ---
    print("\n" + "="*40 + "\nITERATION 2: Advanced\n" + "="*40)

    flat_net_v2 = tf.keras.Sequential([
        layers.Input(shape=(train_x_2d.shape[1],)),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    temporal_net_v2 = tf.keras.Sequential([
        layers.Input(shape=(LOOKBACK, train_x_3d.shape[2])),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])

    performance_log['Iter2_MLP'] = execute_model_cycle(
        "Deep_MLP", flat_net_v2,
        train_x_2d, train_y,
        test_x_2d, test_y,
        output_scaler, run_epochs=100,
        cb_list=[halt_logic]
    )

    performance_log['Iter2_LSTM'] = execute_model_cycle(
        "Deep_LSTM", temporal_net_v2,
        train_x_3d, train_y,
        test_x_3d, test_y,
        output_scaler, run_epochs=100,
        cb_list=[halt_logic]
    )

    # 6. Visualization
    metrics_list = []
    plt.figure(figsize=(20, 10))

    for count, (label, (hist, actual, pred, r2)) in enumerate(performance_log.items(), 1):
        err_rmse = np.sqrt(mean_squared_error(actual, pred))

        metrics_list.append({
            "Architecture": label,
            "R2_Score": round(r2, 4),
            "RMSE": round(err_rmse, 4)
        })

        # Loss curves
        plt.subplot(2, 4, count)
        plt.plot(hist.history['loss'], label='Train')
        plt.plot(hist.history['val_loss'], label='Valid')
        plt.title(f'{label} Loss')
        plt.legend()

        # Prediction vs Actual
        plt.subplot(2, 4, count + 4)
        plt.scatter(actual, pred, alpha=0.3, s=8)
        plt.plot([actual.min(), actual.max()],
                 [actual.min(), actual.max()], 'r--')
        plt.title(f'R2: {r2:.2f}')

    plt.tight_layout()
    plt.show()

    # Summary
    print("\nCOMPARATIVE ANALYSIS:")
    print(pd.DataFrame(metrics_list).to_string(index=False))


if __name__ == '__main__':
    main()