import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_daily_dataset(df: pd.DataFrame, lookback_hours: int = 72) -> tuple[pd.DataFrame, list[str]]:
    """Create daily samples with 72-hour lag features (6 variables) and daily mean target.

    - Requires columns: 'timestamp','Utilization','HYPE_price_usd','FundingRate','TVL_USD','total_borrow','total_supply'
    - Each day t uses lags 1..lookback_hours at the first timestamp of day t (00:00)
    - Target: mean(Utilization) over day t
    - Per-day z-score normalization across each feature's 72-hour window
    """

    df = df.sort_values('timestamp').reset_index(drop=True)
    base_cols = ['Utilization', 'HYPE_price_usd', 'FundingRate', 'TVL_USD', 'total_borrow', 'total_supply']

    # Lag features
    lag_features: dict[str, pd.Series] = {}
    for col in base_cols:
        for k in range(1, lookback_hours + 1):
            lag_features[f'{col}_lag_{k}'] = df[col].shift(k)
    lag_df = pd.DataFrame(lag_features)
    df_lagged = pd.concat([df, lag_df], axis=1)

    # Daily samples using exact midnight rows
    df_lagged['date_day'] = df_lagged['timestamp'].dt.floor('D')
    day_mean = df_lagged.groupby('date_day')['Utilization'].mean().rename('target')
    midnight_rows = df_lagged[df_lagged['timestamp'].dt.hour == 0].sort_values('timestamp')
    daily = midnight_rows.merge(day_mean, on='date_day', how='inner')

    # Keep complete windows
    raw_feature_cols = [f'{col}_lag_{k}' for col in base_cols for k in range(1, lookback_hours + 1)]
    daily = daily.dropna(subset=raw_feature_cols + ['target']).reset_index(drop=True)

    # Per-day z-score across lags for each feature
    norm_blocks: list[pd.DataFrame] = []
    for col in base_cols:
        cols = [f'{col}_lag_{k}' for k in range(1, lookback_hours + 1)]
        mat = daily[cols].to_numpy(dtype='float64')
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1)
        row_std[row_std == 0] = 1.0
        norm = (mat - row_mean[:, None]) / row_std[:, None]
        col_names = [f'{col}_zn_{k}' for k in range(1, lookback_hours + 1)]
        norm_blocks.append(pd.DataFrame(norm, columns=col_names, index=daily.index))
    daily = pd.concat([daily] + norm_blocks, axis=1)

    feature_cols = [f'{col}_zn_{k}' for col in base_cols for k in range(1, lookback_hours + 1)]
    return daily, feature_cols


def train_and_predict_random_forest(
    csv_path: str,
    start_date: str = '2025-07-15',
    end_date: str = '2025-08-15',
    lookback_hours: int = 72,
    model_suffix: str = 'rf'
) -> pd.DataFrame:
    """Train RandomForest on daily samples and predict daily targets in a given window."""

    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise KeyError('Missing required column: date')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    daily, feature_cols = build_daily_dataset(df, lookback_hours=lookback_hours)
    # Merge supply_rate from IRM daily CSVs by date (use total_supply as supply_rate)
    base_dir = os.path.dirname(csv_path)
    fname = os.path.basename(csv_path).lower()
    irm_filename = 'Hypurrfi_Daily_with_IRM.csv' if 'hyperfi' in fname else ('Hyperlend_Daily_with_IRM.csv' if 'hyperlend' in fname else None)
    if irm_filename is not None:
        irm_path = os.path.join(base_dir, irm_filename)
        if os.path.exists(irm_path):
            irm_df = pd.read_csv(irm_path)
            if 'timestamp' in irm_df.columns and 'total_supply' in irm_df.columns:
                irm_df['timestamp'] = pd.to_datetime(irm_df['timestamp'])
                irm_df['date_day'] = irm_df['timestamp'].dt.floor('D')
                supply_map = irm_df[['date_day', 'total_supply']].rename(columns={'total_supply': 'supply_rate'})
                daily = daily.merge(supply_map, on='date_day', how='left')
    if len(daily) < 30:
        print(f'❌ Not enough daily samples in {os.path.basename(csv_path)}')
        return pd.DataFrame()

    # Time-based split: 70% train, 15% val (no test here)
    n = len(daily)
    i_train = max(1, int(n * 0.7))
    i_val = max(i_train + 1, int(n * 0.85))
    train_df = daily.iloc[:i_train]
    val_df = daily.iloc[i_train:i_val]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']

    print(f'   Train (daily): {len(X_train)} | Val (daily): {len(X_val)}')

    # RandomForest model (robust default)
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        oob_score=False,
        bootstrap=True,
    )
    model.fit(X_train, y_train)

    if len(X_val) > 0:
        y_val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = mean_squared_error(y_val, y_val_pred)
        print(f'   Val MAE: {mae:.6f} | Val RMSE: {rmse:.6f}')

    # Predict for the requested window (one per day)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (daily['date_day'] >= start_dt) & (daily['date_day'] <= end_dt)
    predict_df = daily.loc[mask].copy()
    if len(predict_df) == 0:
        print(f'   ❌ No daily rows to predict in window {start_date}..{end_date}')
        return pd.DataFrame()

    y_hat = model.predict(predict_df[feature_cols])

    # Align outputs so that for each day t we return:
    # - utilization: predicted utilization for day t+1
    # - supply_rate: IRM total_supply of day t
    # - borrow_rate_prediction: utilization(t+1) * supply_rate(t)
    util_next = pd.Series(y_hat, index=predict_df.index).shift(-1)
    supply_rate_t = predict_df['supply_rate']
    borrow_rate_pred = util_next * supply_rate_t

    out_df = pd.DataFrame({
        'timestamp': predict_df['date_day'].values,
        'utilization': util_next.values,
        'supply_rate': supply_rate_t.values,
        'borrow_rate_prediction': borrow_rate_pred.values,
    })
    out_df = out_df.dropna(subset=['utilization']).reset_index(drop=True)
    return out_df


def main():
    print('🔥 RandomForest daily predictor (72×6 lags, per-day z-score)')
    base_dir = os.path.dirname(__file__)

    protocols = {
        '1_hyperfi.csv': 'hyperfi',
        '1_hyperlend.csv': 'hyperlend',
    }

    for filename, proto in protocols.items():
        print(f'\n📊 Processing {proto.upper()}')
        csv_path = os.path.join(base_dir, filename)
        preds = train_and_predict_random_forest(csv_path)
        if len(preds) == 0:
            continue
        out_path = os.path.join(base_dir, f'predictions_{proto}_rf.csv')
        preds.to_csv(out_path, index=False)
        print(f'   ✅ Saved {len(preds)} rows to {out_path}')
        print(preds.head().to_string(index=False))


if __name__ == '__main__':
    main()


