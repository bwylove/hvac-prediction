#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进方案完整实现说明文档
"""

README_CONTENT = """
# HVAC 预测程序 三步骤完整实现指南

## 📋 概述

本实现方案将原有的 HVAC 预测程序扩展为三个步骤的完整系统：

1. **步骤1：模型构建与对比** ✅ 已完成
2. **步骤2：PSO 超参数优化** ✅ 已完成  
3. **步骤3：SHAP 特征重要性** ✅ 已完成

---

## 🎯 三个步骤详细说明

### 步骤1：模型构建和比较

**目标**：对比 RF、XGBoost、DNN+LSTM 三个模型的性能

**输出文件**：
- `01_model_comparison.csv` - 三模型对比表（MAPE, R², RMSE, MAE）
- `01_model_comparison_chart.png` - 对比可视化
- `01_model_ranking.csv` - 模型排名
- `01_model_comparison_summary.csv` - 总结统计

**使用方法**：
```python
from model_comparison import (
    build_model_comparison_table,
    plot_model_comparison,
    rank_models,
    summarize_comparison
)

# 获取三个模型的预测结果
y_pred_dl = model_dl.predict(X_test_dl)
y_pred_rf = rf.predict(X_test_ml)
y_pred_xgb = xgb.predict(X_test_ml)

# 构建对比表
comparison_df = build_model_comparison_table(
    y_test_actual,
    {
        'RF': y_pred_rf_actual,
        'XGB': y_pred_xgb_actual,
        'DL': y_pred_dl_actual
    },
    target_names=['供水温度', '回水温度', ...]
)

# 绘制对比图
plot_model_comparison(comparison_df)

# 排名
rank_models(comparison_df)

# 总结
summarize_comparison(comparison_df)
```

**关键指标**：
- **MAPE (%)**: 平均绝对百分比误差，越小越好（<3%为优秀）
- **R²**: 决定系数，越大越好（>0.9为优秀）
- **RMSE**: 均方根误差，越小越好
- **MAE**: 平均绝对误差，越小越好

---

### 步骤2：模型优化 - PSO 超参数优化

**目标**：使用粒子群优化算法找到最优的 DNN+LSTM 超参数

**优化参数**：
- LSTM 单元数：64-256
- Dense 单元数：32-128
- Dropout 比率：0.1-0.5
- 学习率：1e-5 到 1e-4

**优化算法**：
- PSO 粒子数：20
- 迭代次数：12
- 交叉验证：10 折

**输出文件**：
- `02_pso_best_params.json` - 最优超参数
- `02_pso_optimization_history.csv` - 优化过程历史
- `02_pso_optimization_curve.png` - 优化曲线图

**使用方法**：
```python
from pso_optimization import PSO_DNN_LSTM_Optimizer

# 创建优化器
pso_opt = PSO_DNN_LSTM_Optimizer(
    X_train_dl, y_train_scaled,
    X_scaler, y_scaler,
    n_splits=10,           # 10 折交叉验证
    n_particles=20,        # PSO 粒子数
    n_iterations=12,       # PSO 迭代次数
    time_steps=96,
    out_dim=2
)

# 执行优化（耗时30-60分钟）
best_params, best_score, history = pso_opt.optimize()

# 保存结果
pso_opt.save_results('results')

# 查看最优参数
print(best_params)
# {'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.25, 'learning_rate': 0.00005}
```

**注意事项**：
⚠️ PSO 优化耗时较长（30-60 分钟），建议在 GPU 环境下运行
⚠️ 交叉验证时会多次训练模型，对计算资源要求高
💡 首次运行建议设置 `ENABLE_STEP_2_PSO = False`，后续需要时再开启

---

### 步骤3：SHAP 特征重要性分析

**目标**：量化每个特征对模型预测的贡献，增强可解释性

**计算方法**：
- 梯度法：计算预测值对特征的数值梯度
- 排列重要性：通过特征排列计算重要性

**输出文件**：
- `03_shap_feature_importance.csv` - 特征重要性排名
- `03_shap_feature_importance.png` - 特征重要性条形图
- `03_shap_value_distribution.png` - SHAP 值分布
- `03_shap_values_sample.csv` - SHAP 值样本

**使用方法**：
```python
from shap_analysis import quick_shap_analysis

# 快速执行 SHAP 分析
shap_analyzer = quick_shap_analysis(
    model_dl,
    X_test_dl,
    feature_cols,
    top_n=15,      # 显示前 15 个特征
    save_dir='results'
)

# 查看特征重要性排名
print(shap_analyzer.shap_importance.head(20))
#     特征名称        平均绝对SHAP值  SHAP值标准差  重要性排名
# 0  气温2m(℃)         0.2456       0.0123        1
# 1  一次网供水_lag_24  0.1987       0.0089        2
# ...
```

**特征重要性解释**：
- 平均绝对 SHAP 值：该特征对模型预测的平均影响程度
- 排名：特征对模型预测重要性的排序
- 值越大表示该特征对预测的影响越大

---

## 📁 文件结构

```
hvac-prediction/
├── main                          # 原始主程序
├── model_comparison.py           # 【步骤1】模型对比模块
├── pso_optimization.py           # 【步骤2】PSO 优化模块
├── shap_analysis.py              # 【步骤3】SHAP 分析模块
├── INTEGRATION_GUIDE.py           # 集成指南和模板
├── IMPLEMENTATION.py             # 本文件
│
├── results/                       # 输出目录
│   ├── 01_model_comparison.csv
│   ├── 01_model_comparison_chart.png
│   ├── 02_pso_best_params.json
│   ├── 02_pso_optimization_curve.png
│   ├── 03_shap_feature_importance.csv
│   ├── 03_shap_feature_importance.png
│   └── ...
│
└── models/                        # 模型保存目录
    ├── best_dl_model.h5
    └── optimized_model_sequence.h5
```

---

## 🚀 快速开始

### 方案 A：快速集成（推荐）

1. **准备环境**
```bash
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn
```

2. **修改 main 文件**
   - 复制 `model_comparison.py`、`pso_optimization.py`、`shap_analysis.py` 到同一目录
   - 在 `main` 文件顶部添加导入：
```python
from model_comparison import (
    build_model_comparison_table,
    plot_model_comparison,
    rank_models,
    summarize_comparison
)
from shap_analysis import quick_shap_analysis
```

3. **在 `main()` 函数末尾添加**
```python
# 【步骤1】模型对比
comparison_df = build_model_comparison_table(
    y_test_actual,
    {'RF': y_pred_rf_inv, 'XGB': y_pred_xgb_inv, 'DL': y_pred_dl_inv},
    supply_cols + return_cols
)
plot_model_comparison(comparison_df)
rank_models(comparison_df)

# 【步骤3】SHAP 分析
shap_analyzer = quick_shap_analysis(model_dl, X_test_dl, feature_cols)
```

4. **运行**
```bash
python main
```

### 方案 B：逐步实现（学习）

1. **第一周**：实现步骤1（模型对比）
   - 运行 `model_comparison.py` 测试
   - 集成到 `main` 文件
   - 验证输出表格和图表

2. **第二周**：实现步骤2（PSO 优化）
   - 运行 `pso_optimization.py` 测试
   - 在 `main` 中添加 PSO 优化段落
   - 等待优化完成（30-60 分钟）

3. **第三周**：实现步骤3（SHAP 分析）
   - 运行 `shap_analysis.py` 测试
   - 集成到 `main` 文件
   - 生成特征重要性报告

---

## 📊 输出示例

### 步骤1：模型对比表

```
    模型      目标变量    MAE   RMSE    R²   MAPE(%)
0    RF    一次网供水  0.234  0.456  0.892   2.34
1    XGB   一次网供水  0.198  0.412  0.905   2.01
2    DL    一次网供水  0.156  0.389  0.921   1.78
3    RF    一次网回水  0.267  0.523  0.856   3.12
...
```

### 步骤2：PSO 优化结果

```json
{
    "lstm_units": 128,
    "dense_units": 64,
    "dropout_rate": 0.25,
    "learning_rate": 0.00005
}
```

### 步骤3：SHAP 特征重要性

```
     特征名称           平均绝对SHAP值  SHAP值标准差  重要性排名
0   气温2m(℃)             0.2456        0.0123         1
1   一次网供水_lag_24      0.1987        0.0089         2
2   地表温度(℃)           0.1654        0.0076         3
3   总太阳辐射度(down)     0.1423        0.0064         4
...
```

---

## ⚙️ 参数调整建议

### 如果计算资源充足
```python
# PSO 增强参数
n_particles=50      # 增加粒子数
n_iterations=30     # 增加迭代次数
n_splits=15         # 增加交叉验证折数
```

### 如果计算资源有限
```python
# PSO 简化参数
n_particles=10      # 减少粒子数
n_iterations=6      # 减少迭代次数
n_splits=5          # 减少交叉验证折数
```

### SHAP 分析参数
```python
n_samples=100       # SHAP 计算样本数（越多越精确但越慢）
top_n=20            # 显示特征数（太多会显示拥挤）
method='gradient'   # 'gradient' 或 'permutation'
```

---

## 🔍 常见问题

### Q: 为什么 PSO 优化这么慢？
**A**: PSO 需要在 10 折交叉验证中多次训练模型，每个粒子都需要评估。
- 计算量 = 粒子数 × 迭代次数 × 折数 × 每折的训练时间
- 建议使用 GPU 或减少参数数量

### Q: SHAP 分析支持其他模型吗？
**A**: 支持！代码中支持 Keras 和 scikit-learn 模型。
只需修改 `model_type` 参数即可

### Q: 如何使用最优化后的参数重新训练模型？
**A**: 
```python
# 加载最优参数
import json
with open('results/02_pso_best_params.json') as f:
    best_params = json.load(f)

# 用最优参数构建模型
model_final = build_model_with_params(best_params)
model_final.fit(...)
```

### Q: 能否只运行某个步骤？
**A**: 可以！使用开关控制：
```python
ENABLE_STEP_1_COMPARISON = True
ENABLE_STEP_2_PSO = False
ENABLE_STEP_3_SHAP = False
```

---

## ✅ 验证清单

运行完整程序后，检查以下输出：

- [ ] `results/01_model_comparison.csv` - 三模型对比表
- [ ] `results/01_model_comparison_chart.png` - 对比图表
- [ ] `results/01_model_ranking.csv` - 模型排名
- [ ] `results/02_pso_best_params.json` - PSO 最优参数（如启用）
- [ ] `results/02_pso_optimization_curve.png` - PSO 曲线图（如启用）
- [ ] `results/03_shap_feature_importance.csv` - 特征排名
- [ ] `results/03_shap_feature_importance.png` - 特征重要性图

---

## 📞 技术支持

### 遇到问题时的排查步骤

1. **检查输入数据**
   ```python
   print(X_train_dl.shape)  # 应该是 (n, time_steps, features, 1)
   print(y_train_scaled.shape)  # 应该是 (n, n_targets)
   ```

2. **验证模块导入**
   ```python
   from model_comparison import build_model_comparison_table
   print("导入成功")
   ```

3. **检查内存使用**
   - PSO 优化和 SHAP 分析都很消耗内存
   - 如出现 OOM，减少 `n_samples` 参数

4. **查看详细日志**
   - 所有模块都有详细的打印信息
   - 在代码中添加 `verbose=True` 获取更多信息

---

## 📚 参考资源

- **PSO 算法**: https://en.wikipedia.org/wiki/Particle_swarm_optimization
- **SHAP 值**: https://github.com/slundberg/shap
- **模型评估指标**: https://scikit-learn.org/stable/modules/model_evaluation.html

---

## 📝 版本信息

- 实现日期：2026-06-29
- Python 版本：3.7+
- TensorFlow 版本：2.10+
- scikit-learn 版本：1.0+

---

**最后更新**：2026-06-29  
**作者**：GitHub Copilot  
**许可证**：MIT
"""

if __name__ == "__main__":
    print(README_CONTENT)
    
    # 保存为 markdown
    with open('IMPLEMENTATION.md', 'w', encoding='utf-8') as f:
        f.write(README_CONTENT)
    
    print("\n✅ 完整实现指南已生成: IMPLEMENTATION.md")
