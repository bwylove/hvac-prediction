#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多年份多时间间隔 HVAC 供回水温度预测程序（修复版）
功能：
1. 预测2021-2024年的供回水温度（1小时间隔）
2. 2024年数据分两次预测：15分钟间隔和1小时间隔，进行对比
3. 单独预测2021年和2024年，对比结果
4. 输出论文风格图表：残差直方图、散点对比图（5%误差带）、具体表格
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_GPU'] = '0'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
def get_chinese_font():
    preferred_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC',
        'Heiti SC', 'STHeiti', 'AR PL UMing CN',
        'DejaVu Sans', 'Liberation Sans'
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in preferred_fonts:
        if font in available_fonts:
            return font
    for f in available_fonts:
        if any(kw in f for kw in ['CJK', 'SC', 'CN', 'Hei', 'Song', 'Ming', 'Noto']):
            return f
    return None

chinese_font = get_chinese_font()
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    print(f"[字体] 使用中文字体: {chinese_font}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置项 ====================
RANDOM_STATE = 42
TIME_STEPS = 96
FORECAST_HORIZON = 1
EPOCHS = 50
BATCH_SIZE = 4
MODEL_PATH = 'models/optimized_model_sequence.h5'

# ==================== 数据处理函数 ====================

def clean_numeric_series(s):
    """清理数字序列"""
    s = s.astype(str).fillna('')
    s = s.str.replace('\u3000', ' ', regex=False).str.strip()
    s = s.str.replace(r'[,\s]+', '', regex=True)
    s = s.str.replace(r'[^0-9\.\-eE]', '', regex=True)
    s = s.replace('', np.nan)
    return pd.to_numeric(s, errors='coerce')

def parse_datetime_series(s):
    """解析时间序列"""
    dt = pd.to_datetime(s, errors='coerce')
    if dt.notna().all():
        return dt
    s_clean = s.astype(str).str.replace(r'[年|月|日|时|分|秒|/|\\.|T]', ' ', regex=True)
    s_clean = s_clean.str.replace(r'[^\d\s:\-]', ' ', regex=True)
    return pd.to_datetime(s_clean, errors='coerce')

def load_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    print(f"原始数据行数: {len(df)}")
    print(f"列名: {list(df.columns[:15])}")

    time_col = None
    for c in df.columns:
        if '时间' in str(c) or 'time' in str(c).lower():
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]

    print(f"时间列: {time_col}")

    s_raw = df[time_col].astype(str).str.strip()
    dt = parse_datetime_series(s_raw)

    mask_ok = dt.notna()
    df = df.loc[mask_ok].copy()
    df.index = dt.loc[mask_ok]
    df.sort_index(inplace=True)

    print(f"解析后数据行数: {len(df)}")

    non_time_cols = [c for c in df.columns if c != time_col]
    for col in non_time_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = clean_numeric_series(df[col])

    # 识别目标列
    supply_cols = [col for col in df.columns if ("供水" in col and "温度" in col) or ("一次网供水" in col)]
    return_cols = [col for col in df.columns if ("回水" in col and "温度" in col) or ("一次网回水" in col)]

    if not supply_cols:
        supply_cols = [c for c in df.columns if '供水' in c]
    if not return_cols:
        return_cols = [c for c in df.columns if '回水' in c]

    target_cols = supply_cols + return_cols
    print(f"目标列数: {len(target_cols)}, 列名: {target_cols[:4]}")

    # 特征列（包含目标滞后）
    feature_cols = [c for c in ['气温2m(℃)', '地表温度(℃)', '总太阳辐射度(down,J/m2)',
                                '相对湿度(%)', '露点温度(℃)', '降水量(mm)'] if c in df.columns]
    feature_cols += target_cols

    # 添加目标滞后特征
    for target in target_cols:
        if target in df.columns:
            for lag in [1, 4, 12, 24, 96]:
                col_name = f'{target}_lag_{lag}'
                df[col_name] = df[target].shift(lag)
                if col_name not in feature_cols:
                    feature_cols.append(col_name)

    used_cols = [c for c in feature_cols if c in df.columns]
    df_used = df[used_cols].copy()

    df_used = df_used.interpolate(method='time', limit_direction='both', limit=144)
    df_used = df_used.ffill().bfill()
    df_used = df_used.dropna()

    print(f"最终数据行数: {len(df_used)}")

    return df_used, target_cols, feature_cols

def create_sliding_windows(df, features, targets, time_steps=96, forecast_horizon=1):
    """创建滑窗"""
    Xs, ys = [], []
    X = df[features].values
    y = df[targets].values
    n = len(X)

    for i in range(n - time_steps - forecast_horizon + 1):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps + forecast_horizon - 1])

    return np.array(Xs), np.array(ys)

def normalize_data(X, y, scaler_type="robust"):
    """数据标准化"""
    if scaler_type == "robust":
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()
    else:
        raise ValueError("Only robust scaler supported")

    n_samples, time_steps, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled_2d = X_scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, time_steps, n_features)
    y_scaled = y_scaler.fit_transform(y)

    return X_scaled, y_scaled, X_scaler, y_scaler

def resample_data(df, interval_minutes):
    """重新采样数据到指定时间间隔"""
    freq_str = f"{interval_minutes}T"
    df_resampled = df.resample(freq_str).mean()
    df_resampled = df_resampled.interpolate(method='linear')
    df_resampled = df_resampled.dropna()
    return df_resampled

def filter_by_year(df, year):
    """按年份筛选数据"""
    mask = df.index.year == year
    return df[mask]

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def load_pretrained_model(model_path, input_shape, out_dim):
    """加载预训练模型或建立新模型"""
    if os.path.exists(model_path):
        print(f"[模型] 加载预训练模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print(f"[模型] 未找到预训练模型，建立新模型")
        return build_simple_model(input_shape, out_dim)

def build_simple_model(input_shape, out_dim):
    """建立简单的LSTM模型"""
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(out_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# ==================== 可视化函数 ====================

def plot_residual_histogram(results_df, target_cols, save_path, title_suffix=""):
    """
    论文风格残差直方图 - 带 Zero Reference 线
    """
    n_targets = len(target_cols)
    n_cols = 2
    n_rows = (n_targets + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, col in enumerate(target_cols):
        residuals = results_df[f'{col}_误差'].values
        ax = axes[idx]

        ax.hist(residuals, bins=30, color='cornflowerblue', edgecolor='black',
                alpha=0.8, label='Residuals')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Reference')

        ax.set_xlabel('Residuals (True - Predicted)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{col} {title_suffix}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 残差直方图已保存: {save_path}")


def plot_scatter_with_5percent_band(y_true, y_pred, title, save_path, unit="℃"):
    """
    含 5% 误差带的散点图 - 论文风格
    """
    plt.figure(figsize=(8, 8))

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    plot_min, plot_max = min_val - margin, max_val + margin

    # 散点
    plt.scatter(y_true, y_pred, c='orange', alpha=0.5, s=30, edgecolors='w', label='Data points')

    # 完美拟合线
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k-', linewidth=1.5, label='Perfect fit')

    # 5% 误差带
    plt.fill_between([plot_min, plot_max],
                     [plot_min * 0.95, plot_max * 0.95],
                     [plot_min * 1.05, plot_max * 1.05],
                     alpha=0.15, color='red', label='±5% band')

    plt.xlabel(f'True value ({unit})', fontsize=12)
    plt.ylabel(f'Predicted value ({unit})', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 散点图已保存: {save_path}")


def plot_time_series_comparison(results_df, target_cols, save_path, title_suffix=""):
    """
    时间序列对比图 - 真实值 vs 预测值
    """
    n_targets = len(target_cols)
    n_cols = 2
    n_rows = (n_targets + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, col in enumerate(target_cols):
        ax = axes[idx]
        time_index = pd.to_datetime(results_df['时间'])

        ax.plot(time_index, results_df[f'{col}_实际'], 'b-', linewidth=1, label='True data', alpha=0.7)
        ax.plot(time_index, results_df[f'{col}_预测'], 'r--', linewidth=1, label='Predicted data', alpha=0.7)

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(f'{col} (℃)', fontsize=10)
        ax.set_title(f'{col} {title_suffix}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 时间序列对比图已保存: {save_path}")


def plot_metrics_comparison(results_list, save_path):
    """
    多结果 MAPE/R² 对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    descriptions = [r['description'] for r in results_list]

    # 计算每个结果的平均 MAPE 和 R²
    avg_mapes = []
    avg_r2s = []
    for r in results_list:
        metrics = r['metrics']
        mapes = [metrics[col]['MAPE'] for col in r['target_cols']]
        r2s = [metrics[col]['R²'] for col in r['target_cols']]
        avg_mapes.append(np.mean(mapes))
        avg_r2s.append(np.mean(r2s))

    # MAPE 对比
    axes[0].bar(range(len(descriptions)), avg_mapes, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(descriptions)))
    axes[0].set_xticklabels(descriptions, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Average MAPE (%)', fontsize=11)
    axes[0].set_title('MAPE Comparison', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    # R² 对比
    axes[1].bar(range(len(descriptions)), avg_r2s, color='coral', alpha=0.8)
    axes[1].set_xticks(range(len(descriptions)))
    axes[1].set_xticklabels(descriptions, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Average R²', fontsize=11)
    axes[1].set_title('R² Comparison', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 指标对比图已保存: {save_path}")


# ==================== 预测和评估函数 ====================

def predict_and_evaluate(df_data, target_cols, feature_cols, year=None, interval_minutes=60, description=""):
    """预测并评估"""
    print(f"\n{'='*70}")
    print(f"[预测任务] {description}")
    print(f"{'='*70}")

    # 按年份筛选
    if year:
        df_work = filter_by_year(df_data, year)
        print(f"筛选年份: {year}, 数据行数: {len(df_work)}")
    else:
        df_work = df_data.copy()

    # 重新采样
    if interval_minutes != 15:
        print(f"重新采样到 {interval_minutes} 分钟间隔...")
        df_work = resample_data(df_work, interval_minutes)

    if len(df_work) < TIME_STEPS + 1:
        print(f"警告：数据行数 {len(df_work)} 不足 {TIME_STEPS + 1}，跳过此任务")
        return None

    # 创建滑窗
    X, y = create_sliding_windows(df_work, feature_cols, target_cols, TIME_STEPS, FORECAST_HORIZON)
    print(f"样本数: {len(X)}, 特征维度: {X.shape[2]}, 目标维度: {y.shape[1]}")

    # 标准化
    X_scaled, y_scaled, X_scaler, y_scaler = normalize_data(X, y)

    # 加载模型
    out_dim = len(target_cols)
    input_shape = (TIME_STEPS, X_scaled.shape[2])

    model = load_pretrained_model(MODEL_PATH, input_shape, out_dim)

    # 如果模型是新建的，进行快速训练
    if not os.path.exists(MODEL_PATH):
        print("[模型] 进行快速训练...")
        model.fit(X_scaled, y_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # 预测
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_scaled)

    # 创建结果DataFrame
    start_idx = TIME_STEPS + FORECAST_HORIZON - 1
    time_index = df_work.index[start_idx:start_idx + len(y_true)]

    results_df = pd.DataFrame({'时间': time_index})

    for idx, col in enumerate(target_cols):
        results_df[f'{col}_实际'] = y_true[:, idx]
        results_df[f'{col}_预测'] = y_pred[:, idx]
        results_df[f'{col}_误差'] = y_pred[:, idx] - y_true[:, idx]

    # 计算指标
    print("\n[评估结果]")
    print("-" * 70)

    metrics_all = {}
    for idx, col in enumerate(target_cols):
        metrics = calculate_metrics(y_true[:, idx], y_pred[:, idx])
        metrics_all[col] = metrics
        print(f"{col}:")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")
        print(f"  R²:   {metrics['R²']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.4f}°C")
        print(f"  RMSE: {metrics['RMSE']:.4f}°C")

    return {
        'results_df': results_df,
        'metrics': metrics_all,
        'y_true': y_true,
        'y_pred': y_pred,
        'target_cols': target_cols,
        'description': description
    }


def save_results(results, output_dir='results'):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    description = results['description'].replace(' ', '_').replace('(', '').replace(')', '')

    # 保存预测结果CSV
    csv_path = os.path.join(output_dir, f'predictions_{description}.csv')
    results['results_df'].to_csv(csv_path, index=False)
    print(f"[保存] CSV结果: {csv_path}")

    # 保存指标
    metrics_path = os.path.join(output_dir, f'metrics_{description}.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"预测任务: {results['description']}\n")
        f.write("="*70 + "\n")
        for col, metrics in results['metrics'].items():
            f.write(f"\n{col}:\n")
            for key, val in metrics.items():
                if key == 'MAPE':
                    f.write(f"  {key}: {val:.4f}%\n")
                elif key == 'R²':
                    f.write(f"  {key}: {val:.6f}\n")
                else:
                    f.write(f"  {key}: {val:.4f}\n")

    print(f"[保存] 指标文件: {metrics_path}")

    # 保存可视化
    plot_residual_histogram(
        results['results_df'], results['target_cols'],
        os.path.join(output_dir, f'residuals_{description}.png'),
        title_suffix=f"({results['description']})"
    )

    # 为每个目标变量绘制散点图
    for idx, col in enumerate(results['target_cols']):
        plot_scatter_with_5percent_band(
            results['y_true'][:, idx], results['y_pred'][:, idx],
            f'{col} - {results["description"]}',
            os.path.join(output_dir, f'scatter_{description}_{col}.png')
        )

    # 时间序列对比图
    plot_time_series_comparison(
        results['results_df'], results['target_cols'],
        os.path.join(output_dir, f'timeseries_{description}.png'),
        title_suffix=f"({results['description']})"
    )


# ==================== 主函数 ====================

def main():
    print("\n" + "="*70)
    print("HVAC 多年份多时间间隔预测系统")
    print("="*70)

    # 加载数据
    file_path = 'temperature19.csv'
    if not os.path.exists(file_path):
        print(f"[错误] 找不到数据文件: {file_path}")
        return

    df_data, target_cols, feature_cols = load_data(file_path)

    results_collection = []

    # 任务1: 2021-2024年预测（1小时间隔）
    print("\n[任务1] 2021-2024年预测（1小时间隔）")
    for year in [2021, 2022, 2023, 2024]:
        result = predict_and_evaluate(
            df_data, target_cols, feature_cols,
            year=year, interval_minutes=60,
            description=f"Year_{year}_1h"
        )
        if result:
            results_collection.append(result)
            save_results(result, output_dir='results/yearly_1h')

    # 任务2: 2024年预测对比（15分钟 vs 1小时）
    print("\n[任务2] 2024年预测对比（15分钟 vs 1小时）")
    result_15m = predict_and_evaluate(
        df_data, target_cols, feature_cols,
        year=2024, interval_minutes=15,
        description="Year_2024_15min"
    )
    if result_15m:
        results_collection.append(result_15m)
        save_results(result_15m, output_dir='results/2024_interval_comparison')

    result_1h = predict_and_evaluate(
        df_data, target_cols, feature_cols,
        year=2024, interval_minutes=60,
        description="Year_2024_1h"
    )
    if result_1h:
        results_collection.append(result_1h)
        save_results(result_1h, output_dir='results/2024_interval_comparison')

    # 任务3: 2021年和2024年预测对比
    print("\n[任务3] 2021年 vs 2024年预测对比")
    result_2021 = predict_and_evaluate(
        df_data, target_cols, feature_cols,
        year=2021, interval_minutes=60,
        description="Year_2021_comparison"
    )
    if result_2021:
        results_collection.append(result_2021)
        save_results(result_2021, output_dir='results/year_comparison')

    result_2024 = predict_and_evaluate(
        df_data, target_cols, feature_cols,
        year=2024, interval_minutes=60,
        description="Year_2024_comparison"
    )
    if result_2024:
        results_collection.append(result_2024)
        save_results(result_2024, output_dir='results/year_comparison')

    # 总体对比图
    print("\n[总结] 生成对比分析...")
    if len(results_collection) > 1:
        plot_metrics_comparison(
            results_collection,
            'results/metrics_comparison.png'
        )

    print("\n" + "="*70)
    print("所有预测任务完成！")
    print("结果保存位置：")
    print("  - results/yearly_1h/        : 2021-2024年每年预测（1小时）")
    print("  - results/2024_interval_comparison/  : 2024年时间间隔对比")
    print("  - results/year_comparison/  : 2021年和2024年对比")
    print("="*70 + "\n")


if __name__ == "__main__":
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    main()
