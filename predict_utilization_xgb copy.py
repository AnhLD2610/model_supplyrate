import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_daily_dataset(df: pd.DataFrame, lookback_hours: int = 168) -> tuple[pd.DataFrame, list[str]]:
    """Create daily samples with 168-hour lag features (6 variables) and daily mean target.

    - Requires columns: 'timestamp','Utilization','HYPE_price_usd','FundingRate','TVL_USD','total_borrow','total_supply'
    - Each day t uses lags 1..lookback_hours at the first timestamp of day t
    - Target: mean(Utilization) over day t
    - Per-day z-score normalization across each feature's window
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

    # Daily samples: use EXACT 00:00 decision time to avoid leakage
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


def train_and_predict_xgb(
    csv_path: str,
    start_date: str = '2025-08-1',
    end_date: str = '2025-09-1',
    lookback_hours: int = 168,
) -> pd.DataFrame:
    """Train XGBoost on daily samples and predict daily targets in a window."""

    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise KeyError('Missing required column: date')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    daily, feature_cols = build_daily_dataset(df, lookback_hours=lookback_hours)
    # Merge supply_rate from IRM daily CSVs by date
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

    # Time split train (skip first 30 days), no explicit val
    n = len(daily)
    i_train = max(1, int(n * 0.85))
    train_df = daily.iloc[30:i_train]

    X_train = train_df[feature_cols]
    y_train = train_df['target']

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict per day in requested window
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (daily['date_day'] >= start_dt) & (daily['date_day'] <= end_dt)
    predict_df = daily.loc[mask].copy()
    if len(predict_df) == 0:
        print(f'   ❌ No daily rows to predict in window {start_date}..{end_date}')
        return pd.DataFrame()

    # Predict utilization for each day t
    predict_df['utilization'] = model.predict(predict_df[feature_cols])

    # Join supply_rate(t-1)
    predict_df['date_prev'] = predict_df['date_day'] - pd.Timedelta(days=1)
    sup_series = daily[['date_day', 'supply_rate']].rename(
        columns={'date_day': 'date_prev', 'supply_rate': 'supply_rate_prev'}
    )
    predict_df = predict_df.merge(sup_series, on='date_prev', how='left')

    # Compute borrow(t) = supply_rate(t-1) * utilization_pred(t)
    predict_df['borrow_rate_prediction'] = predict_df['supply_rate_prev'] * predict_df['utilization']

    # Final output (mirror lr_model_v2.py)
    out_df = predict_df[['date_day', 'supply_rate', 'utilization', 'borrow_rate_prediction']].dropna().rename(
        columns={'date_day': 'timestamp'}
    ).reset_index(drop=True)
    return out_df


def main():
    print('🔥 XGBoost daily predictor (168×6 lags, per-day z-score) — borrow(t)=supply(t-1)*util_pred(t)')
    base_dir = os.path.dirname(__file__)

    protocols = {
        '1_hyperfi.csv': 'hyperfi',
        '1_hyperlend.csv': 'hyperlend',
    }

    for filename, proto in protocols.items():
        print(f'\n📊 Processing {proto.upper()}')
        csv_path = os.path.join(base_dir, filename)
        preds = train_and_predict_xgb(csv_path)
        if len(preds) == 0:
            continue
        out_dir = 'D:\defi-strategies\DeFi_strategies_backtest'
        out_path = os.path.join(out_dir, f'predictions_{proto}_lr.csv')
        preds.to_csv(out_path, index=False)
        print(f'   ✅ Saved {len(preds)} rows to {out_path}')
        print(preds.head().to_string(index=False))


if __name__ == '__main__':
    main()
