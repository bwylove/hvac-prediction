#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC 预测完整集成方案
将步骤1、步骤2、步骤3 集成到主程序中

集成步骤：
1. 在 main() 函数中添加以下导入和调用
2. 根据需要设置 ENABLE_STEP_1/2/3 开关
"""

# ==================== 集成指南 ====================

INTEGRATION_GUIDE = """

## 集成步骤

### Step 0: 导入新模块
在 main.py 的顶部添加：

    from model_comparison import (
        build_model_comparison_table,
        plot_model_comparison,
        summarize_comparison,
        rank_models
    )
    from pso_optimization import PSO_DNN_LSTM_Optimizer
    from shap_analysis import quick_shap_analysis

### Step 1: 设置开关（在 main() 函数开始处）

    # 步骤开关
    ENABLE_STEP_1_COMPARISON = True   # 模型对比
    ENABLE_STEP_2_PSO = False         # PSO 优化（耗时，默认关闭）
    ENABLE_STEP_3_SHAP = True         # SHAP 分析

### Step 2: 在模型训练后添加步骤1（模型对比）

在当前 evaluate_and_visualize() 和 evaluate_ml_model() 调用之后，
main() 函数末尾添加：

    # ============================================================
    # 【步骤1】三模型对比
    # ============================================================
    if ENABLE_STEP_1_COMPARISON:
        print("\n" + "="*60)
        print("【步骤1】模型构建和比较")
        print("="*60)
        
        # 获取三个模型的预测结果
        y_pred_dl = model_dl.predict(X_test_dl, verbose=0)
        y_pred_rf = rf.predict(X_test_ml)
        y_pred_xgb = xgb.predict(X_test_ml) if xgb is not None else np.zeros_like(y_pred_rf)
        
        # 反归一化
        if y_pred_dl.ndim > 2:
            y_pred_dl = y_pred_dl.reshape(-1, y_pred_dl.shape[-1])
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
        
        print("\n>>> 模型对比表：")
        print(comparison_df.to_string(index=False))
        
        # 保存对比表
        comparison_df.to_csv('results/01_model_comparison.csv', index=False)
        
        # 绘制对比图
        plot_model_comparison(comparison_df)
        
        # 生成排名
        ranking = rank_models(comparison_df)
        
        # 生成总结
        summary = summarize_comparison(comparison_df)

### Step 3: 在模型训练后添加步骤2（PSO 优化）

    # ============================================================
    # 【步骤2】PSO 超参数优化
    # ============================================================
    if ENABLE_STEP_2_PSO and len(X_train_dl) > 100:
        print("\n" + "="*60)
        print("【步骤2】模型优化 - PSO 超参数优化")
        print("="*60)
        
        # 创建 PSO 优化器
        pso_opt = PSO_DNN_LSTM_Optimizer(
            X_train_dl, y_train_scaled,
            X_scaler, y_scaler,
            n_splits=10,
            n_particles=20,
            n_iterations=12,
            time_steps=TIME_STEPS,
            out_dim=out_dim
        )
        
        # 执行优化
        best_params, best_score, history = pso_opt.optimize()
        
        # 保存结果
        pso_opt.save_results('results')
        
        print(f"\n[PSO] 优化完成！最佳 MAPE: {best_score:.4f}")
        
        # 使用最优参数重新训练（可选）
        print("\n使用最优参数训练最终模型...")
        # model_dl_final = build_model_with_pso_params(best_params)
        # ... 训练代码

### Step 4: 在模型训练后添加步骤3（SHAP 分析）

    # ============================================================
    # 【步骤3】SHAP 特征重要性分析
    # ============================================================
    if ENABLE_STEP_3_SHAP:
        print("\n" + "="*60)
        print("【步骤3】利用 SHAP 增强可解释性")
        print("="*60)
        
        # 执行 SHAP 分析
        shap_analyzer = quick_shap_analysis(
            model_dl,
            X_test_dl,
            feature_cols,
            top_n=15,
            save_dir='results'
        )
        
        # 打印特征重要性排名
        print("\n特征重要性排名 (Top 20):")
        print(shap_analyzer.shap_importance.head(20).to_string(index=False))

### Step 5: 完整的 main() 函数末尾示例

    def main():
        # ... 原有代码 ...
        
        # 设置开关
        ENABLE_STEP_1_COMPARISON = True
        ENABLE_STEP_2_PSO = False
        ENABLE_STEP_3_SHAP = True
        
        # 加载数据、构建模型等原有代码...
        # ...
        
        # 【步骤1】模型对比（见上面的代码块）
        if ENABLE_STEP_1_COMPARISON:
            # ... 对比代码 ...
        
        # 【步骤2】PSO 优化（见上面的代码块）
        if ENABLE_STEP_2_PSO:
            # ... PSO 代码 ...
        
        # 【步骤3】SHAP 分析（见上面的代码块）
        if ENABLE_STEP_3_SHAP:
            # ... SHAP 代码 ...
        
        print("\\n" + "="*60)
        print("全部实验结束。结果保存在 results/ 目录。")
        print("="*60)

### 输出文件结构

执行完整程序后，results/ 目录中将包含：

    results/
    ├── 01_model_comparison.csv              # 三模型对比表
    ├── 01_model_comparison_chart.png        # 对比图表
    ├── 01_model_ranking.csv                 # 模型排名
    ├── 01_model_comparison_summary.csv      # 对比总结
    │
    ├── 02_pso_best_params.json              # 最优超参数
    ├── 02_pso_optimization_history.csv      # 优化过程
    ├── 02_pso_optimization_curve.png        # 优化曲线图
    │
    ├── 03_shap_feature_importance.csv       # 特征重要性排名
    ├── 03_shap_feature_importance.png       # 特征重要性图
    ├── 03_shap_value_distribution.png       # SHAP 值分布
    ├── 03_shap_values_sample.csv            # SHAP 值样本
    │
    ├── DL_train_predictions.csv
    ├── DL_test_predictions.csv
    ├── RF_train_predictions.csv
    ├── RF_test_predictions.csv
    ├── XGB_train_predictions.csv
    ├── XGB_test_predictions.csv
    │
    └── ... 其他图表

"""

print(INTEGRATION_GUIDE)

# ==================== 辅助函数 ====================

def create_integrated_main_template():
    """
    生成完整的集成 main() 函数模板
    """
    template = '''
# main.py 中的 main() 函数集成模板

def main():
    """
    HVAC 预测主程序 - 完整版
    包含：步骤1（模型对比）、步骤2（PSO优化）、步骤3（SHAP分析）
    """
    # ===== 步骤开关 =====
    ENABLE_STEP_1_COMPARISON = True   # 模型对比
    ENABLE_STEP_2_PSO = False         # PSO 优化（可选，耗时）
    ENABLE_STEP_3_SHAP = True         # SHAP 分析
    
    # ===== 原有的初始化代码 =====
    print("系统信息：", platform.platform())
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"TensorFlow: {tf.__version__}, GPU_AVAILABLE: {GPU_AVAILABLE}")
    
    # 数据加载、特征工程等...
    df_used, feature_cols, target_cols, supply_cols, return_cols, df_original = load_and_enhance_data(file_path)
    
    # 构建滑窗、创建模型等...
    X, y = create_sliding_windows(df_used, feature_cols, target_cols, ...)
    
    # 数据分割、归一化、模型训练...
    # ... 原有的训练代码 ...
    
    # ===== 【步骤1】模型对比 =====
    if ENABLE_STEP_1_COMPARISON:
        print("\\n" + "="*60)
        print("【步骤1】模型构建和比较")
        print("="*60)
        
        # 获取预测
        y_pred_dl = model_dl.predict(X_test_dl, verbose=0)
        y_pred_rf = rf.predict(X_test_ml)
        y_pred_xgb = xgb.predict(X_test_ml) if xgb else None
        
        # 反归一化
        y_pred_dl_inv = y_scaler.inverse_transform(y_pred_dl.reshape(-1, y_pred_dl.shape[-1]))
        y_pred_rf_inv = y_scaler.inverse_transform(y_pred_rf)
        if y_pred_xgb is not None:
            y_pred_xgb_inv = y_scaler.inverse_transform(y_pred_xgb)
        y_test_inv = y_scaler.inverse_transform(y_test_scaled)
        
        # 对比
        comparison_df = build_model_comparison_table(
            y_test_inv,
            {
                'RF': y_pred_rf_inv,
                'XGB': y_pred_xgb_inv if y_pred_xgb is not None else None,
                'DL': y_pred_dl_inv
            },
            supply_cols + return_cols
        )
        
        print("\\n三模型对比结果：")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv('results/01_model_comparison.csv', index=False)
        plot_model_comparison(comparison_df)
        rank_models(comparison_df)
        summarize_comparison(comparison_df)
    
    # ===== 【步骤2】PSO 优化 =====
    if ENABLE_STEP_2_PSO and len(X_train_dl) > 100:
        print("\\n" + "="*60)
        print("【步骤2】PSO 超参数优化")
        print("="*60)
        
        pso_opt = PSO_DNN_LSTM_Optimizer(
            X_train_dl, y_train_scaled, X_scaler, y_scaler,
            n_splits=10, n_particles=20, n_iterations=12,
            time_steps=TIME_STEPS, out_dim=out_dim
        )
        
        best_params, best_score, history = pso_opt.optimize()
        pso_opt.save_results('results')
    
    # ===== 【步骤3】SHAP 分析 =====
    if ENABLE_STEP_3_SHAP:
        print("\\n" + "="*60)
        print("【步骤3】SHAP 特征重要性分析")
        print("="*60)
        
        shap_analyzer = quick_shap_analysis(
            model_dl, X_test_dl, feature_cols,
            top_n=15, save_dir='results'
        )
    
    print("\\n" + "="*60)
    print("全部实验结束。结果保存在 results/ 目录。")
    print("="*60)

if __name__ == "__main__":
    main()
'''
    return template


# 生成并保存集成模板
if __name__ == "__main__":
    template = create_integrated_main_template()
    
    # 保存模板
    with open('INTEGRATION_TEMPLATE.py', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("\n集成模板已生成：INTEGRATION_TEMPLATE.py")
