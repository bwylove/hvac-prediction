#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1：模型比较与评估
对比 RF、XGBoost、DNN+LSTM 三个模型的性能
输出纯数字指标表格：MAPE、R²、RMSE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    计算回归模型的评估指标
    
    Args:
        y_true: 真实值 (n_samples, n_targets) 或 (n_samples,)
        y_pred: 预测值，相同形状
    
    Returns:
        dict: 包含 MAE, RMSE, R², MAPE 的指标字典
    """
    # 确保是 2D 数组
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 避免除以零
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4),
        'MAPE(%)': round(mape, 2)
    }


def build_model_comparison_table(y_true, y_pred_dict, target_names):
    """
    构建三模型对比表（纯数字）
    
    Args:
        y_true: (n_samples, n_targets) 真实值
        y_pred_dict: {'RF': pred_rf, 'XGB': pred_xgb, 'DL': pred_dl}
        target_names: 目标变量名称列表
    
    Returns:
        pd.DataFrame: 模型对比表
    """
    rows = []
    
    for model_name, y_pred in y_pred_dict.items():
        # 确保形状一致
        if y_pred.ndim > 2:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        if y_true.ndim > 2:
            y_true_use = y_true.reshape(-1, y_true.shape[-1])
        else:
            y_true_use = y_true
        
        # 对每个目标变量计算指标
        for idx, target_name in enumerate(target_names):
            if y_true_use.shape[1] > idx and y_pred.shape[1] > idx:
                metrics = calculate_metrics(y_true_use[:, idx], y_pred[:, idx])
                row = {
                    '模型': model_name,
                    '目标变量': target_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R²': metrics['R²'],
                    'MAPE(%)': metrics['MAPE(%)']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_model_comparison(comparison_df, save_path='results/01_model_comparison_chart.png'):
    """
    绘制模型对比图
    
    Args:
        comparison_df: 模型对比表
        save_path: 保存路径
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    # 按目标变量分类绘图
    targets = comparison_df['目标变量'].unique()
    n_targets = len(targets)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['MAPE(%)', 'R²', 'RMSE', 'MAE']
    models = comparison_df['模型'].unique()
    
    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            ax = axes[idx]
            
            # 提取数据
            data_for_plot = comparison_df.groupby('模型')[metric].mean()
            
            # 绘制柱状图
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            data_for_plot.plot(kind='bar', ax=ax, color=colors[:len(data_for_plot)])
            
            ax.set_title(f'{metric} 对比（平均值）', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xlabel('模型')
            ax.grid(alpha=0.3, axis='y')
            ax.legend(title='模型')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 模型对比图已保存: {save_path}")


def rank_models(comparison_df, save_path='results/01_model_ranking.csv'):
    """
    对模型进行排名
    
    Args:
        comparison_df: 模型对比表
        save_path: 保存路径
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    # 计算平均排名
    ranking = pd.DataFrame()
    
    for metric in ['MAPE(%)', 'R²', 'RMSE', 'MAE']:
        # 针对 MAPE/RMSE/MAE，越小越好；R² 越大越好
        if metric in ['MAPE(%)', 'RMSE', 'MAE']:
            rank = comparison_df.groupby('模型')[metric].mean().rank()
        else:  # R²
            rank = comparison_df.groupby('模型')[metric].mean().rank(ascending=False)
        
        ranking[f'{metric}_排名'] = rank
    
    # 计算综合排名
    ranking['综合排名'] = ranking.mean(axis=1).rank()
    ranking = ranking.sort_values('综合排名')
    
    ranking.to_csv(save_path)
    print(f"[表格] 模型排名已保存: {save_path}")
    print("\n模型综合排名:")
    print(ranking)
    
    return ranking


def summarize_comparison(comparison_df, save_path='results/01_model_comparison_summary.csv'):
    """
    生成模型对比总结表
    
    Args:
        comparison_df: 模型对比表
        save_path: 保存路径
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    # 按模型计算平均指标
    summary = comparison_df.groupby('模型').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'R²': 'mean',
        'MAPE(%)': 'mean'
    }).round(4)
    
    # 添加最优指标行
    best_row = {
        '模型': '最优',
        'MAE': summary['MAE'].min(),
        'RMSE': summary['RMSE'].min(),
        'R²': summary['R²'].max(),
        'MAPE(%)': summary['MAPE(%)'].min()
    }
    
    summary.to_csv(save_path)
    print(f"\n[表格] 模型对比总结已保存: {save_path}")
    print("\n各模型平均性能指标:")
    print(summary)
    
    return summary


if __name__ == "__main__":
    # 示例使用
    print("模型对比模块已加载，可在主程序中调用")
