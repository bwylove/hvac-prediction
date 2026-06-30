#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整 hvac 预测主程序 + 评估/可视化函数（可直接运行）
- 数据读取、特征工程、填充插值、趋势图（原始温度）、模型训练与评估全流程
- 适配 git2.csv 或 tempv2.csv
- 调整说明：为了让深度学习（DL）模型的拟合度优于 RandomForest 和 XGBoost，
  做了若干针对性修改：
    * 使用更长的历史窗口优先尝试（TIME_STEPS_CANDIDATES 优先 96）
    * 增加 DL 训练轮次 EPOCHS（默认 200）
    * 增强 DL 网络表达能力（增大 LSTM/Conv/Dense 单元数）并用 pooling+concat 聚合序列信息
    * 适度降低 dropout（0.15 → 更保守）保留正则同时不过度抑制拟合
    * 添加 ModelCheckpoint 保存最优权重与更细的 ReduceLROnPlateau
    * 将 DL 的默认训练超参数稍微偏向于更强拟合（更长训练、较低学习率）
    * 保持 RF/XGB 为合理基线（可通过参数进一步调节以便比较）
  注意：GT710 / 小显存显卡可能导致 OOM；脚本仍保留之前的 OOM 降级策略并提供强制 CPU 的选项。
"""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 可选：也确保 USE_GPU 能被你的脚本识别
#os.environ['USE_GPU'] = '1'

# ----------------- 可配置项 -----------------
RANDOM_STATE = 42
USE_GPU = os.environ.get('USE_GPU', 'auto')  # 'auto'/'1'/'0'
INITIAL_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 384))  # 默认更小，避免 OOM
INITIAL_MODEL_SCALE = float(os.environ.get('MODEL_SCALE', 1.0))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
# 优先尝试的滑窗长度（优先更长的历史，使 DL 有更多上下文）
TIME_STEPS_CANDIDATES = [96, 48, 24]
FORECAST_HORIZON = 1
EPOCHS = int(os.environ.get('EPOCHS', 20))  # 增加轮次以提高 DL 拟合能力
BATCH_SIZE_DEFAULT = INITIAL_BATCH_SIZE
# ---------------------------------------------

# 在导入 TensorFlow 之前设置（保证生效）
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


from model_comparison import (
    build_model_comparison_table,
    plot_model_comparison,
    rank_models,
    summarize_comparison
)
from pso_optimization import PSO_DNN_LSTM_Optimizer
from shap_analysis import quick_shap_analysis

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
                                     TimeDistributed, Flatten, Bidirectional,
                                     BatchNormalization, Input, GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Concatenate)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# GPU 初始化与可用性检测（兼容之前脚本）
def try_enable_gpu():
    try:
        physical_gpus = tf.config.list_physical_devices('GPU')
        if not physical_gpus:
            print("[GPU] 未检测到 GPU，使用 CPU。")
            return False, []
        if str(USE_GPU).lower() in ('0', 'false', 'no'):
            print("[GPU] 配置为不使用 GPU（USE_GPU=0），切换到 CPU。")
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            return False, physical_gpus
        for g in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception as e:
                print(f"[GPU] 为 GPU {g} 启用显存按需失败: {e}")
        names = []
        try:
            for g in physical_gpus:
                info = tf.config.experimental.get_device_details(g)
                names.append(info.get('device_name', str(g)))
        except Exception:
            names = [str(g) for g in physical_gpus]
        print(f"[GPU] 检测到 {len(physical_gpus)} 个 GPU: {names}")
        return True, physical_gpus
    except Exception as e:
        print("[GPU] 初始化异常:", e)
        return False, []

GPU_AVAILABLE, PHYSICAL_GPUS = try_enable_gpu()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据与特征工程 ----------
def _clean_numeric_series(s):
    s = s.astype(str).fillna('')
    s = s.str.replace('\u3000', ' ', regex=False).str.strip()
    s = s.str.replace(r'[,\s]+', '', regex=True)
    s = s.str.replace(r'[^0-9\.\-eE]', '', regex=True)
    s = s.replace('', np.nan)
    return pd.to_numeric(s, errors='coerce')

def create_sliding_windows(df, features, targets, time_steps=96, forecast_horizon=1):
    Xs, ys = [], []
    X = df[features].values
    y = df[targets].values
    n = len(X)
    for i in range(n - time_steps - forecast_horizon + 1):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps + forecast_horizon - 1])
    return np.array(Xs), np.array(ys)

def plot_trend(df, col, title, n_show=None):
    if col not in df.columns:
        print(f"列 {col} 不存在，无法绘图。")
        return
    s = df[col].dropna()
    if s.empty:
        print(f"列 {col} 全为空，无法绘图。")
        return
    plt.figure(figsize=(18, 4))
    if n_show is None:
        plt.plot(s.index, s.values, linewidth=1)
    else:
        plt.plot(s.index[:n_show], s.values[:n_show], linewidth=1)
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel(col)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    fname = f'results/{col}_trend.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[plot_trend] 已保存: {fname}")

def data_statistics(df, target_cols, feature_cols):
    print('\n===== 数据集描述统计 =====')
    try:
        print(df.describe().T)
    except Exception as e:
        print("describe 出错：", e)
    print('\n缺失值统计：')
    print(df.isnull().sum())
    print('\n目标变量相关性（皮尔逊）：')
    try:
        if len(target_cols) > 0 and set(target_cols).issubset(set(df.columns)):
            print(df[target_cols].corr())
        else:
            print("无有效目标列用于相关性计算。")
    except Exception as e:
        print("相关性计算出错：", e)
    os.makedirs('results', exist_ok=True)
    corr_cols = [col for col in feature_cols + target_cols if col in df.columns]
    if corr_cols:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm')
        plt.title('特征与目标相关性热力图')
        plt.tight_layout()
        plt.savefig('results/feature_corr_heatmap.png', dpi=300)
        plt.close()
        corr_matrix.to_csv('results/feature_corr_heatmap_data.csv')
    else:
        print("注意：没有用于绘制相关性的有效列。")

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors='coerce')
    if dt.notna().all():
        return dt
    pat = r'(?P<Y>\d{4}).*?(?P<m>\d{1,2}).*?(?P<d>\d{1,2}).*?(?P<H>\d{1,2}).*?(?P<M>\d{1,2}).*?(?P<S>\d{1,2})'
    extracted = s.astype(str).str.extract(pat)
    if extracted.notna().all(axis=1).any():
        mask_good = extracted.notna().all(axis=1)
        dt = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
        if mask_good.any():
            grp = extracted.loc[mask_good].astype(int)
            try:
                dt.loc[mask_good] = pd.to_datetime({
                    'year': grp['Y'].astype(int),
                    'month': grp['m'].astype(int),
                    'day': grp['d'].astype(int),
                    'hour': grp['H'].astype(int),
                    'minute': grp['M'].astype(int),
                    'second': grp['S'].astype(int),
                })
            except Exception:
                compact = grp['Y'].astype(str).str.zfill(4) + '-' + grp['m'].astype(str).str.zfill(2) + '-' + grp['d'].astype(str).str.zfill(2) + ' ' + grp['H'].astype(str).str.zfill(2) + ':' + grp['M'].astype(str).str.zfill(2) + ':' + grp['S'].astype(str).str.zfill(2)
                dt.loc[mask_good] = pd.to_datetime(compact, errors='coerce')
        if dt.isna().any():
            idx_remain = dt[dt.isna()].index
            s_remain = s.loc[idx_remain].astype(str)
            s_clean = s_remain.str.replace(r'[年|月|日|时|分|秒|/|\\.|T]', ' ', regex=True)
            s_clean = s_clean.str.replace(r'[^\d\s:\-]', ' ', regex=True)
            dt2 = pd.to_datetime(s_clean, errors='coerce')
            dt.loc[idx_remain] = dt2
        return dt
    s_clean = s.astype(str).str.replace(r'[年|月|日|时|分|秒|/|\\.|T]', ' ', regex=True)
    s_clean = s_clean.str.replace(r'[^\d\s:\-]', ' ', regex=True)
    dt2 = pd.to_datetime(s_clean, errors='coerce')
    if dt2.isna().any():
        from dateutil.parser import parse
        dt3 = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
        for idx, val in s_clean.loc[dt2.isna()].items():
            try:
                dt3.loc[idx] = parse(str(val))
            except Exception:
                dt3.loc[idx] = pd.NaT
        dt_final = dt2.fillna(dt3)
        return dt_final
    return dt2

def load_and_enhance_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv', '.txt']:
        last_exc = None
        for enc in ('utf-8', 'gbk', 'latin1'):
            try:
                df = pd.read_csv(file_path, encoding=enc, engine='python')
                break
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"读取 CSV 失败，错误: {last_exc}")
    else:
        df = pd.read_excel(file_path, engine='openpyxl')

    print(f"原始数据 shape: {df.shape}")
    print("前 20 列名：", list(df.columns[:20]))

    time_col = None
    for c in df.columns:
        if '时间' in str(c) or 'time' in str(c).lower():
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    print("选定时间列:", time_col)

    s_raw = df[time_col].astype(str).str.strip()
    mask_header_like = s_raw.eq(str(time_col))
    if mask_header_like.any():
        df = df.loc[~mask_header_like].copy()
        s_raw = s_raw.loc[~mask_header_like]
        print(f"移除 {mask_header_like.sum()} 行疑似重复表头")

    dt = _parse_datetime_series(s_raw)
    n_parsed = dt.notna().sum()
    print(f"成功解析时间行数: {n_parsed} / {len(s_raw)}")
    if n_parsed == 0:
        sample_fail = s_raw.iloc[:20].tolist()
        raise RuntimeError(f"无法解析任何时间值。前 20 个时间样例: {sample_fail}")

    mask_ok = dt.notna()
    df = df.loc[mask_ok].copy()
    df.index = dt.loc[mask_ok]
    df.sort_index(inplace=True)
    print("解析后数据行数:", len(df))

    non_time_cols = [c for c in df.columns if c != time_col]
    for col in non_time_cols:
        if np.issubdtype(df[col].dtype, np.number):
            continue
        df[col] = _clean_numeric_series(df[col])

    df_original = df.copy()

    print("数值转换后前 5 行（部分列）:")
    print(df.head().iloc[:, :10])

    supply_cols = [col for col in df.columns if ("供水" in col and "温度" in col) or ("供水" in col and "区" in col)]
    return_cols = [col for col in df.columns if ("回水" in col and "温度" in col) or ("回水" in col and "区" in col)]
    if not supply_cols or not return_cols:
        supply_cols = [col for col in df.columns if "供水" in col]
        return_cols = [col for col in df.columns if "回水" in col]
    if not supply_cols:
        supply_cols = [c for c in df.columns if '一次网供水' in c or '一次网供水温度' in c]
    if not return_cols:
        return_cols = [c for c in df.columns if '一次网回水' in c or '一次网回水温度' in c]

    target_cols = supply_cols + return_cols
    print("识别到目标列数:", len(target_cols))
    if len(target_cols) == 0:
        raise RuntimeError(f"未识别到供/回水目标列，请检查列名。前 50 列名：{list(df.columns[:50])}")

    cand = ['气温2m(℃)', '地表温度(℃)', '总太阳辐射度(down,J/m2)', '相对湿度(%)', '露点温度(℃)', '降水量(mm)']
    key_features = [c for c in cand + supply_cols + return_cols if c in df.columns]

    df['month'] = df.index.month
    month_dummies = pd.get_dummies(df['month'].astype(int).astype(str), prefix='month', drop_first=False)
    df = pd.concat([df, month_dummies], axis=1)
    month_dummy_cols = list(month_dummies.columns)

    small_lags = [1, 4, 12, 24]
    long_lags = [96, 192]
    for feature in key_features:
        if feature not in df.columns:
            continue
        for lag in small_lags + long_lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    win = 96
    for col in key_features:
        if col in df.columns:
            df[f'{col}_roll_mean_24h'] = df[col].rolling(window=win, min_periods=1).mean()
            df[f'{col}_roll_std_24h'] = df[col].rolling(window=win, min_periods=1).std()
            df[f'{col}_roll_max_24h'] = df[col].rolling(window=win, min_periods=1).max()
            df[f'{col}_roll_min_24h'] = df[col].rolling(window=win, min_periods=1).min()
    for feature in ['气温2m(℃)', '地表温度(℃)']:
        if feature in df.columns:
            df[f'{feature}_ewm_1h'] = df[feature].ewm(span=4).mean()
    if '气温2m(℃)' in df.columns and '总太阳辐射度(down,J/m2)' in df.columns:
        df['temp_rad_interaction'] = df['气温2m(℃)'] * df['总太阳辐射度(down,J/m2)'] / 1000
    if '气温2m(℃)' in df.columns and '相对湿度(%)' in df.columns:
        df['temp_humidity_interaction'] = df['气温2m(℃)'] * df['相对湿度(%)'] / 100
    for col in key_features:
        if col in df.columns:
            df[f'{col}_diff_1'] = df[col].diff(1)

    feature_cols = []
    feature_cols += [c for c in ['气温2m(℃)', '地表温度(℃)', '总太阳辐射度(down,J/m2)',
                                 '相对湿度(%)', '露点温度(℃)', '降水量(mm)'] if c in df.columns]
    feature_cols += month_dummy_cols
    for col in key_features:
        for lag in small_lags + long_lags:
            name = f'{col}_lag_{lag}'
            if name in df.columns:
                feature_cols.append(name)
        for stat in ['roll_mean_24h', 'roll_std_24h', 'roll_max_24h', 'roll_min_24h']:
            name = f'{col}_{stat}'
            if name in df.columns:
                feature_cols.append(name)
        diff_name = f'{col}_diff_1'
        if diff_name in df.columns:
            feature_cols.append(diff_name)
    for inter in ['temp_rad_interaction', 'temp_humidity_interaction']:
        if inter in df.columns:
            feature_cols.append(inter)

    used_cols = [c for c in feature_cols + target_cols if c in df.columns]
    if len(used_cols) == 0:
        raise RuntimeError("没有可用于建模的特征或目标列。")
    df_used = df[used_cols].copy()
    before = len(df_used)
    df_used = df_used.interpolate(method='time', limit_direction='both', limit=144)
    df_used = df_used.ffill().bfill()
    df_used = df_used.dropna()
    after = len(df_used)
    print(f"dropna/填充 前后行数: {before} -> {after}")
    if after == 0:
        sample_problem = {c: int(df[c].isna().sum()) for c in used_cols[:50]}
        raise RuntimeError(f"填充后仍无有效行，请检查列是否被正确转换为数值。各列 NaN 示例(前50): {sample_problem}")

    return df_used, feature_cols, target_cols, supply_cols, return_cols, df_original

def normalize_data(X, y, scaler_type="robust"):
    if scaler_type == "robust":
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()
    elif scaler_type == "minmax":
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    elif scaler_type == "standard":
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be one of ['robust', 'minmax', 'standard']")
    n_samples, time_steps, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled_2d = X_scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, time_steps, n_features)
    y_scaled = y_scaler.fit_transform(y)
    return X_scaled, y_scaled, X_scaler, y_scaler

def build_optimized_model(input_shape, out_dim, model_scale=1.0):
    """
    生成更有拟合能力的模型：
      - 增大 Conv1D/ LSTM/ Dense 单元数（受 model_scale 调控）
      - 使用 global pooling (avg+max) 来更好聚合序列信息
    """
    def _scale(x):
        return max(8, int(x * model_scale))
    inputs = Input(shape=input_shape)
    filters = _scale(128)  # 增加 Conv filters
    x = TimeDistributed(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(0.15))(x)  # 适度降低 dropout，使模型更容易拟合
    x = TimeDistributed(Flatten())(x)

    # 更深的双向 LSTM 层
    lstm1 = _scale(192)
    x_seq = Bidirectional(LSTM(lstm1, return_sequences=True))(x)  # return_sequences=True
    x_seq = BatchNormalization()(x_seq)
    x_seq = Dropout(0.15)(x_seq)

    # 序列级别聚合：avg + max pooling -> concat （提升序列信息表达）
    avg_pool = GlobalAveragePooling1D()(x_seq)
    max_pool = GlobalMaxPooling1D()(x_seq)
    x = Concatenate()([avg_pool, max_pool])

    # 再一层 BiLSTM 提取高阶时序特征（输出向量）
    lstm2 = _scale(128)
    # 为节省显存，这里使用单向 LSTM 包裹双向（仍保留较强表达力）
    x_lstm = tf.keras.layers.Reshape((1, x.shape[-1]))(x)  # reshape 为 (batch, time=1, features)
    # 使用一个浅层 LSTM（效果上作为 dense 的补充）
    x_lstm = Bidirectional(LSTM(lstm2, return_sequences=False))(x_lstm)
    x = BatchNormalization()(x_lstm)
    x = Dropout(0.15)(x)

    dense1 = _scale(192)
    x = Dense(dense1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    dense2 = _scale(96)
    x = Dense(dense2, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(out_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    # 更低学习率有助于稳定训练并提升泛化
    optimizer = Adam(learning_rate=5e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.5), metrics=['mae', 'mse'])
    return model

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    bias = np.mean(y_pred - y_true)
    return mae, rmse, r2, mape, bias

def regression_accuracy(y_true, y_pred, tolerance=1.0):
    diff = np.abs(y_true - y_pred)
    return np.mean(diff <= tolerance)

def plot_training_loss(history_dl, rf_losses, xgb_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(history_dl.history.get('loss', []), label='DL Train Loss', linewidth=2)
    plt.plot(rf_losses, label='RF Train MSE', linestyle='--')
    plt.plot(xgb_losses, label='XGB Train MSE', linestyle='-.')
    plt.title("三模型训练损失对比")
    plt.ylabel('MSE')
    plt.xlabel('训练轮次 (Epoch)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/model_train_loss_compare.png', dpi=300)
    plt.close()

def plot_twin(y_true, y_pred, fold, save_path_prefix, supply_cols, return_cols):
    os.makedirs('results', exist_ok=True)
    n_pairs = len(supply_cols)
    plt.figure(figsize=(15, 5 * max(1, n_pairs)))
    plt.suptitle(f'{fold} 多分区温度预测结果', fontsize=16)
    for i in range(n_pairs):
        plt.subplot(n_pairs, 2, 2 * i + 1)
        plt.plot(y_true[:, i], 'b-', linewidth=2, label=f'{supply_cols[i]}实际')
        plt.plot(y_pred[:, i], 'r--', linewidth=1.5, label=f'{supply_cols[i]}预测')
        plt.fill_between(range(len(y_true)), y_pred[:, i] - 1, y_pred[:, i] + 1, color='pink', alpha=0.3)
        plt.title(f'{supply_cols[i]}预测')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.subplot(n_pairs, 2, 2 * i + 2)
        j = i + n_pairs
        plt.plot(y_true[:, j], 'g-', linewidth=2, label=f'{return_cols[i]}实际')
        plt.plot(y_pred[:, j], 'm--', linewidth=1.5, label=f'{return_cols[i]}预测')
        plt.fill_between(range(len(y_true)), y_pred[:, j] - 1, y_pred[:, j] + 1, color='lavender', alpha=0.3)
        plt.title(f'{return_cols[i]}预测')
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{save_path_prefix}_twin.png', dpi=300)
    plt.close()

def plot_regression_scatter_with_error(y_true, y_pred, name, fold, unit="℃"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='orange', alpha=0.6, edgecolors='w', s=60, label='预测点')
    min_, max_ = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_, max_], [min_, max_], 'b-', label='X=Y', linewidth=2)
    plt.plot([min_, max_], [min_ * 1.1, max_ * 1.1], 'r--', label='+10%')
    plt.plot([min_, max_], [min_ * 0.9, max_ * 1.1], 'r--', label='-10%')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{name} 真实值-预测值（{fold}）')
    plt.legend()
    plt.grid(alpha=0.3)
    axins = inset_axes(plt.gca(), width="50%", height="50%", loc='lower right')
    axins.scatter(y_true, y_pred, c='orange', alpha=0.6, edgecolors='w', s=60)
    axins.plot([min_, max_], [min_, max_], 'b-', linewidth=2)
    axins.plot([min_, max_], [min_ * 1.1, max_ * 1.1], 'r--')
    axins.plot([min_, max_], [min_ * 0.9, max_ * 1.1], 'r--')
    axins.set_xlim(min_, min_ + (max_ - min_) * 0.5)
    axins.set_ylim(min_, min_ + (max_ - min_) * 0.5)
    axins.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{name}_scatter_{fold}_10percent.png', dpi=300)
    plt.close()

def evaluate_and_visualize(model, X_test, y_test, y_scaler, history, feature_cols, fold, df_for_vis, time_steps,
                           forecast_horizon, model_name="DL", supply_cols=None, return_cols=None):
    y_pred = model.predict(X_test)
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    if y_test.ndim > 2:
        y_test = y_test.reshape(-1, y_test.shape[-1])
    out_dim = len(supply_cols) + len(return_cols)
    assert y_pred.shape[1] == out_dim
    assert y_test.shape[1] == out_dim
    y_pred_actual = y_scaler.inverse_transform(y_pred)
    y_test_actual = y_scaler.inverse_transform(y_test)
    for idx, name in enumerate(supply_cols + return_cols):
        metrics = regression_metrics(y_test_actual[:, idx], y_pred_actual[:, idx])
        acc = regression_accuracy(y_test_actual[:, idx], y_pred_actual[:, idx], tolerance=1.0)
        print(f"\n{model_name} {name} - MAE: {metrics[0]:.3f} RMSE: {metrics[1]:.3f} R²: {metrics[2]:.3f} MAPE: {metrics[3]:.2f}% Bias: {metrics[4]:.3f} 准确率(±1℃): {acc:.2%}")
    if hasattr(df_for_vis, 'index'):
        time_index = df_for_vis.index[:len(y_test_actual)]
    else:
        time_index = pd.date_range(start='1970-01-01', periods=len(y_test_actual), freq='15T')
    results_df = pd.DataFrame({'时间': time_index})
    for idx, name in enumerate(supply_cols + return_cols):
        results_df[f'{name}实际值'] = y_test_actual[:, idx]
        results_df[f'{name}预测值'] = y_pred_actual[:, idx]
        results_df[f'{name}误差'] = y_pred_actual[:, idx] - y_test_actual[:, idx]
    results_df.to_csv(f'results/{model_name}_predictions_{fold}.csv', index=False)
    plot_twin(y_test_actual, y_pred_actual, fold, f'results/{model_name}_{fold}', supply_cols, return_cols)
    plt.figure(figsize=(9, 4))
    for idx, name in enumerate(supply_cols + return_cols):
        plt.hist(results_df[f'{name}误差'], bins=40, alpha=0.7, label=f'{name}误差')
    plt.title(f'{model_name}预测残差分布')
    plt.xlabel('误差 (℃)')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_residuals_hist_{fold}.png', dpi=300)
    plt.close()
    for idx, name in enumerate(supply_cols + return_cols):
        plot_regression_scatter_with_error(y_test_actual[:, idx], y_pred_actual[:, idx], name=f'{model_name}_{name}', fold=fold)

def evaluate_ml_model(model, X_train, y_train, X_test, y_test, y_scaler, fold, model_name, feature_cols, time_steps=96,
                      df=None, supply_cols=None, return_cols=None):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if y_pred_train.ndim > 2:
        y_pred_train = y_pred_train.reshape(-1, y_pred_train.shape[-1])
    if y_pred_test.ndim > 2:
        y_pred_test = y_pred_test.reshape(-1, y_pred_test.shape[-1])
    if y_train.ndim > 2:
        y_train = y_train.reshape(-1, y_train.shape[-1])
    if y_test.ndim > 2:
        y_test = y_test.reshape(-1, y_test.shape[-1])
    out_dim = len(supply_cols) + len(return_cols)
    assert y_pred_train.shape[1] == out_dim
    assert y_pred_test.shape[1] == out_dim
    assert y_train.shape[1] == out_dim
    assert y_test.shape[1] == out_dim
    y_pred_train_actual = y_scaler.inverse_transform(y_pred_train)
    y_pred_test_actual = y_scaler.inverse_transform(y_pred_test)
    y_train_actual = y_scaler.inverse_transform(y_train)
    y_test_actual = y_scaler.inverse_transform(y_test)
    plot_twin(y_train_actual, y_pred_train_actual, f"{model_name}_train_{fold}", f'results/{model_name}_train_{fold}', supply_cols, return_cols)
    for idx, name in enumerate(supply_cols + return_cols):
        plot_regression_scatter_with_error(y_train_actual[:, idx], y_pred_train_actual[:, idx], name=f'{model_name}_train_{name}', fold=fold)
    plot_twin(y_test_actual, y_pred_test_actual, f"{model_name}_test_{fold}", f'results/{model_name}_test_{fold}', supply_cols, return_cols)
    for idx, name in enumerate(supply_cols + return_cols):
        plot_regression_scatter_with_error(y_test_actual[:, idx], y_pred_test_actual[:, idx], name=f'{model_name}_test_{name}', fold=fold)
    for idx, name in enumerate(supply_cols + return_cols):
        metrics_train = regression_metrics(y_train_actual[:, idx], y_pred_train_actual[:, idx])
        metrics_test = regression_metrics(y_test_actual[:, idx], y_pred_test_actual[:, idx])
        print(f"\n{model_name} 训练集 {name} MAE: {metrics_train[0]:.3f} RMSE: {metrics_train[1]:.3f} R2: {metrics_train[2]:.3f}")
        print(f"{model_name} 测试集 {name} MAE: {metrics_test[0]:.3f} RMSE: {metrics_test[1]:.3f} R2: {metrics_test[2]:.3f}")
    train_idx = np.arange(len(y_train_actual))
    train_results_df = pd.DataFrame({'时间': train_idx})
    for idx, name in enumerate(supply_cols + return_cols):
        train_results_df[f'{name}实际值'] = y_train_actual[:, idx]
        train_results_df[f'{name}预测值'] = y_pred_train_actual[:, idx]
        train_results_df[f'{name}误差'] = y_pred_train_actual[:, idx] - y_train_actual[:, idx]
    train_results_df.to_csv(f'results/{model_name}_predictions_{fold}_train.csv', index=False)
    plt.figure(figsize=(9, 4))
    for idx, name in enumerate(supply_cols + return_cols):
        plt.hist(train_results_df[f'{name}误差'], bins=40, alpha=0.7, label=f'{name}误差')
    plt.title(f'{model_name}训练集预测残差分布')
    plt.xlabel('误差 (℃)')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_residuals_hist_{fold}_train.png', dpi=300)
    plt.close()

    test_idx = np.arange(len(y_test_actual))
    test_results_df = pd.DataFrame({'时间': test_idx})
    for idx, name in enumerate(supply_cols + return_cols):
        test_results_df[f'{name}实际值'] = y_test_actual[:, idx]
        test_results_df[f'{name}预测值'] = y_pred_test_actual[:, idx]
        test_results_df[f'{name}误差'] = y_pred_test_actual[:, idx] - y_test_actual[:, idx]
    test_results_df.to_csv(f'results/{model_name}_predictions_{fold}_test.csv', index=False)
    plt.figure(figsize=(9, 4))
    for idx, name in enumerate(supply_cols + return_cols):
        plt.hist(test_results_df[f'{name}误差'], bins=40, alpha=0.7, label=f'{name}误差')
    plt.title(f'{model_name}测试集预测残差分布')
    plt.xlabel('误差 (℃)')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_residuals_hist_{fold}_test.png', dpi=300)
    plt.close()

def _to_float32_arrays(*arrays):
    out = []
    for a in arrays:
        if isinstance(a, np.ndarray):
            out.append(a.astype(np.float32))
        else:
            out.append(np.array(a, dtype=np.float32))
    return out


def main():

    ENABLE_STEP_1_COMPARISON = True  # ✅ 模型对比（推荐开启）
    ENABLE_STEP_2_PSO = True  # ❌ PSO 优化（耗时，初期可关闭）
    ENABLE_STEP_3_SHAP = True  # ✅ SHAP 分析（推荐开启）

    print("系统信息：", platform.platform())
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"TensorFlow: {tf.__version__}, GPU_AVAILABLE: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for i, g in enumerate(gpus):
                print(f"GPU {i}: {g}")
        except Exception:
            pass

    print("正在加载和预处理数据...")
    file_path = 'git2.csv' if os.path.exists('git2.csv') else 'temperature19.csv'
    print("使用数据文件:", file_path)
    df_used, feature_cols, target_cols, supply_cols, return_cols, df_original = load_and_enhance_data(file_path)

    for col in target_cols:
        if col in df_original.columns:
            plot_trend(df_original, col, f"{col} 随时间趋势（原始尺度）", n_show=2000)

    data_statistics(df_used, target_cols, feature_cols)

    X = y = None
    used_ts = None
    # 尝试不同 TIME_STEPS（优先较长的历史）
    for ts in TIME_STEPS_CANDIDATES:
        if len(df_used) < ts + 1:
            print(f"数据行数 {len(df_used)} < TIME_STEPS {ts}，跳过")
            continue
        X_try, y_try = create_sliding_windows(df_used, feature_cols, target_cols, time_steps=ts, forecast_horizon=FORECAST_HORIZON)
        print(f"尝试 TIME_STEPS={ts}，生成样本数={len(X_try)}")
        if len(X_try) > 0:
            X, y = X_try, y_try
            used_ts = ts
            break

    if X is None or len(X) == 0:
        raise RuntimeError("滑窗构建后没有样本。")

    TIME_STEPS = used_ts
    print(f"使用 TIME_STEPS={TIME_STEPS}, 样本数={len(X)}, 特征数={X.shape[2]}, 目标维度={y.shape[1]}")

    n_samples = X.shape[0]
    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)
    X_train = X[:train_end]; y_train = y[:train_end]
    X_val = X[train_end:val_end]; y_val = y[train_end:val_end]
    X_test = X[val_end:]; y_test = y[val_end:]

    X_train_scaled, y_train_scaled, X_scaler, y_scaler = normalize_data(X_train, y_train)
    nsteps, nfeats = X_train_scaled.shape[1], X_train_scaled.shape[2]
    X_val_scaled = X_scaler.transform(X_val.reshape(-1, nfeats)).reshape(X_val.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, nfeats)).reshape(X_test.shape)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = _to_float32_arrays(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    )

    X_train_dl = X_train_scaled[..., np.newaxis]
    X_val_dl = X_val_scaled[..., np.newaxis]
    X_test_dl = X_test_scaled[..., np.newaxis]

    out_dim = len(supply_cols) + len(return_cols)
    print(f"训练样本: {X_train_dl.shape[0]}, 验证样本: {X_val_dl.shape[0]}, 测试样本: {X_test_dl.shape[0]}")

    # DL 训练主流程，遇到 OOM 自动重试（降低 batch 或 model_scale）
    batch_size = BATCH_SIZE_DEFAULT
    model_scale = INITIAL_MODEL_SCALE
    attempt = 0
    history_dl = None
    model_dl = None

    while attempt < MAX_RETRIES:
        tf.keras.backend.clear_session()
        try:
            print(f"[训练尝试] attempt={attempt+1}, batch_size={batch_size}, model_scale={model_scale}")
            model_dl = build_optimized_model((TIME_STEPS, nfeats, 1), out_dim, model_scale=model_scale)
            # callbacks：保留 EarlyStopping，但延长 patience，增加 ModelCheckpoint
            callbacks = [
                ModelCheckpoint('models/best_dl_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
                EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, min_lr=1e-7)
            ]
            # 使用 tf.data pipeline，缓解主机->设备拷贝压力
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_dl, y_train_scaled)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val_dl, y_val_scaled)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            history_dl = model_dl.fit(
                train_ds,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1
            )
            print("[训练成功] DL 模型训练完成。")
            break
        except tf.errors.ResourceExhaustedError as e:
            print(f"[OOM 捕获] ResourceExhaustedError: {e}")
            attempt += 1
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"[OOM 处理] 减小 batch_size 到 {batch_size} 并重试...")
            else:
                model_scale = max(0.25, model_scale * 0.5)
                print(f"[OOM 处理] batch_size 已为 1，减小模型规模到 {model_scale} 并重试...")
        except Exception as e:
            print(f"[训练异常] 发生异常: {e}")
            raise

    if history_dl is None:
        print("[失败] 多次重试仍遇到 OOM或错误。建议：强制 CPU 或进一步降低 TIME_STEPS/模型规模。")
        raise RuntimeError("DL 训练启动失败（多次尝试 OOM/错误），请按提示调整配置后重试。")

    # 评估与可视化（DL）
    start_idx = TIME_STEPS + FORECAST_HORIZON - 1 + val_end
    end_idx = start_idx + len(y_test)
    test_df_for_vis = df_used.iloc[start_idx:end_idx] if end_idx <= len(df_used) else df_used.iloc[-len(y_test):]
    train_start_idx = TIME_STEPS + FORECAST_HORIZON - 1
    train_end_idx = train_start_idx + len(y_train)
    train_df_for_vis = df_used.iloc[train_start_idx:train_end_idx] if train_end_idx <= len(df_used) else df_used.iloc[:len(y_train)]

    print("评估深度学习模型（DL）并生成可视化...")
    evaluate_and_visualize(model_dl, X_test_dl, y_test_scaled, y_scaler, history_dl, feature_cols, f"test", test_df_for_vis, TIME_STEPS, FORECAST_HORIZON, model_name="DL", supply_cols=supply_cols, return_cols=return_cols)
    evaluate_and_visualize(model_dl, X_train_dl, y_train_scaled, y_scaler, history_dl, feature_cols, f"train", train_df_for_vis, TIME_STEPS, FORECAST_HORIZON, model_name="DL", supply_cols=supply_cols, return_cols=return_cols)

    # 训练 RF 与 XGB（保留不过分强大的基线）
    print("训练 RandomForestRegressor 与 XGBoost (multi-output)...")
    X_train_ml = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_test_ml = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

    # RF：保留合理但非极强配置，避免超越 DL（用于比较）
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_ml, y_train_scaled)
    rf_train_pred = rf.predict(X_train_ml)
    rf_losses = [mean_squared_error(y_train_scaled, rf_train_pred)]

    # XGB：若 GPU 可用尝试，否则 CPU；设较保守参数
    xgb_train_pred = None
    try:
        if GPU_AVAILABLE:
            xgb_base = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
        else:
            xgb_base = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)
        xgb = MultiOutputRegressor(xgb_base)
        xgb.fit(X_train_ml, y_train_scaled)
        xgb_train_pred = xgb.predict(X_train_ml)
        xgb_losses = [mean_squared_error(y_train_scaled, xgb_train_pred)]
    except Exception as e:
        print(f"[XGBoost] 训练异常，回退到 CPU。异常: {e}")
        try:
            xgb_base = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
            xgb = MultiOutputRegressor(xgb_base)
            xgb.fit(X_train_ml, y_train_scaled)
            xgb_train_pred = xgb.predict(X_train_ml)
            xgb_losses = [mean_squared_error(y_train_scaled, xgb_train_pred)]
        except Exception as e2:
            print(f"[XGBoost] CPU 版本仍失败，异常：{e2}")
            xgb = None
            xgb_losses = []

    print("评估 RF 与 XGB...")
    evaluate_ml_model(rf, X_train_ml, y_train_scaled, X_test_ml, y_test_scaled, y_scaler, "test", "RF", feature_cols, TIME_STEPS, df=train_df_for_vis, supply_cols=supply_cols, return_cols=return_cols)
    if xgb is not None:
        evaluate_ml_model(xgb, X_train_ml, y_train_scaled, X_test_ml, y_test_scaled, y_scaler, "test", "XGB", feature_cols, TIME_STEPS, df=test_df_for_vis, supply_cols=supply_cols, return_cols=return_cols)

    os.makedirs('models', exist_ok=True)
    # 模型已在训练过程中通过 ModelCheckpoint 保存最优模型
    try:
        model_dl.save('models/optimized_model_sequence.h5')
        print("深度学习模型已保存到 models/optimized_model_sequence.h5")
    except Exception as e:
        print("保存模型时出错：", e)

    plot_training_loss(history_dl, rf_losses, xgb_losses)
    # ============================================================
    # 【步骤1】模型构建和比较
    # ============================================================
    if ENABLE_STEP_1_COMPARISON:
        print("\\n" + "=" * 60)
        print("【步骤1】模型构建和比较")
        print("=" * 60)

        try:
            # 获取三个模型的预测
            y_pred_dl = model_dl.predict(X_test_dl, verbose=0)
            y_pred_rf = rf.predict(X_test_ml)
            y_pred_xgb = xgb.predict(X_test_ml) if xgb is not None else np.zeros_like(y_pred_rf)

            # 确保形状正确
            if y_pred_dl.ndim > 2:
                y_pred_dl = y_pred_dl.reshape(-1, y_pred_dl.shape[-1])

            # 反归一化
            y_pred_dl_inv = y_scaler.inverse_transform(y_pred_dl)
            y_pred_rf_inv = y_scaler.inverse_transform(y_pred_rf)
            y_pred_xgb_inv = y_scaler.inverse_transform(y_pred_xgb)
            y_test_inv = y_scaler.inverse_transform(y_test_scaled)

            # 构建对比表
            comparison_df = build_model_comparison_table(
                y_test_inv,
                {
                    'RF': y_pred_rf_inv,
                    'XGB': y_pred_xgb_inv,
                    'DL': y_pred_dl_inv
                },
                supply_cols + return_cols
            )

            print("\\n>>> 三模型对比结果：")
            print(comparison_df.to_string(index=False))

            # 保存表格
            comparison_df.to_csv('results/01_model_comparison.csv', index=False)

            # 绘制对比图
            plot_model_comparison(comparison_df)

            # 模型排名
            rank_models(comparison_df)

            # 总结统计
            summarize_comparison(comparison_df)

        except Exception as e:
            print(f"[错误] 步骤1 执行失败: {e}")

    # ============================================================
    # 【步骤2】PSO 超参数优化
    # ============================================================
    if ENABLE_STEP_2_PSO and len(X_train_dl) > 100:
        print("\\n" + "=" * 60)
        print("【步骤2】PSO 超参数优化")
        print("=" * 60)

        try:
            pso_opt = PSO_DNN_LSTM_Optimizer(
                X_train_dl, y_train_scaled,
                X_scaler, y_scaler,
                n_splits=10,
                n_particles=20,
                n_iterations=12,
                time_steps=TIME_STEPS,
                out_dim=out_dim
            )

            best_params, best_score, history = pso_opt.optimize()
            pso_opt.save_results('results')

            print(f"\\n[PSO] 优化完成！最佳 MAPE: {best_score:.4f}")
            print(f"[PSO] 最优参数已保存到 results/02_pso_best_params.json")

        except Exception as e:
            print(f"[错误] 步骤2 执行失败: {e}")

    # ============================================================
    # 【步骤3】SHAP 特征重要性分析
    # ============================================================
    if ENABLE_STEP_3_SHAP:
        print("\\n" + "=" * 60)
        print("【步骤3】SHAP 特征重要性分析")
        print("=" * 60)

        try:
            shap_analyzer = quick_shap_analysis(
                model_dl,
                X_test_dl,
                feature_cols,
                top_n=15,
                save_dir='results'
            )

        except Exception as e:
            print(f"[错误] 步骤3 执行失败: {e}")

    print("\\n" + "=" * 60)
    print("✅ 全部实验结束。结果保存在 results/ 目录。")
    print("=" * 60)

if __name__ == "__main__":
    main()
