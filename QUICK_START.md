#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始指南 - 五分钟集成三步骤
"""

QUICK_START = """
# ⚡ 快速开始指南（5分钟集成）

## 第一步：复制模块文件（1分钟）

将以下文件复制到你的项目目录：
- model_comparison.py      ← 模型对比
- pso_optimization.py      ← PSO 优化
- shap_analysis.py         ← SHAP 分析
- INTEGRATION_GUIDE.py     ← 集成指南
- IMPLEMENTATION.md        ← 完整文档

## 第二步：修改 main 文件（3分钟）

### 2.1 在文件顶部添加导入（第 50 行左右）

```python
# 新增导入
from model_comparison import (
    build_model_comparison_table,
    plot_model_comparison,
    rank_models,
    summarize_comparison
)
from pso_optimization import PSO_DNN_LSTM_Optimizer
from shap_analysis import quick_shap_analysis
```

### 2.2 在 main() 函数开始处添加开关（第 630 行左右）

```python
def main():
    # ===== 【重要】步骤开关设置 =====
    ENABLE_STEP_1_COMPARISON = True   # ✅ 模型对比（推荐开启）
    ENABLE_STEP_2_PSO = False         # ❌ PSO 优化（耗时，初期可关闭）
    ENABLE_STEP_3_SHAP = True         # ✅ SHAP 分析（推荐开启）
    
    # ... 原有的初始化代码 ...
```

### 2.3 在 main() 函数末尾添加三步骤代码

在当前 `plot_training_loss()` 调用之后，`print("全部实验结束...")` 之前，添加：

```python
    # ============================================================
    # 【步骤1】模型构建和比较
    # ============================================================
    if ENABLE_STEP_1_COMPARISON:
        print("\\n" + "="*60)
        print("【步骤1】模型构建和比较")
        print("="*60)
        
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
        print("\\n" + "="*60)
        print("【步骤2】PSO 超参数优化")
        print("="*60)
        
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
        print("\\n" + "="*60)
        print("【步骤3】SHAP 特征重要性分析")
        print("="*60)
        
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
    
    print("\\n" + "="*60)
    print("✅ 全部实验结束。结果保存在 results/ 目录。")
    print("="*60)
```

## 第三步：运行程序（1分钟）

```bash
# 运行主程序
python main

# 或指定数据文件
python main git2.csv
```

## 预期输出

### 控制台输出示例

```
============================================================
【步骤1】模型构建和比较
============================================================

    模型    目标变量     MAE   RMSE    R²   MAPE(%)
0    RF  一次网供水  0.2342  0.4563  0.8921   2.34
1    XGB 一次网供水  0.1987  0.4124  0.9052   2.01
2    DL  一次网供水  0.1564  0.3890  0.9210   1.78

[图表] 模型对比图已保存: results/01_model_comparison_chart.png

============================================================
【步骤3】SHAP 特征��要性分析
============================================================

[SHAP] 计算特征重要性 (方法: gradient)...
[SHAP] 特征重要性排名:
     特征名称          平均绝对SHAP值  SHAP值标准差  重要性排名
0   气温2m(℃)           0.2456        0.0123         1
1   一次网供水_lag_24    0.1987        0.0089         2
...

============================================================
✅ 全部实验结束。结果保存在 results/ 目录。
============================================================
```

### 生成的文件

```
results/
├── 01_model_comparison.csv              ← 对比表格
├── 01_model_comparison_chart.png        ← 对比图
├── 01_model_ranking.csv                 ← 排名表
├── 01_model_comparison_summary.csv      ← 总结表
├── 03_shap_feature_importance.csv       ← 特征重要性
├── 03_shap_feature_importance.png       ← 特征重要性图
└── 03_shap_value_distribution.png       ← 分布图
```

---

## 🎯 四种常用场景

### 场景A：只做模型对比（快速）

```python
ENABLE_STEP_1_COMPARISON = True
ENABLE_STEP_2_PSO = False
ENABLE_STEP_3_SHAP = False
# 运行时间：< 5 分钟
```

### 场景B：模型对比 + 特征分析（推荐）

```python
ENABLE_STEP_1_COMPARISON = True
ENABLE_STEP_2_PSO = False
ENABLE_STEP_3_SHAP = True
# 运行时间：5-10 分钟
```

### 场景C：完整三步骤（全面）

```python
ENABLE_STEP_1_COMPARISON = True
ENABLE_STEP_2_PSO = True        # 耗时！
ENABLE_STEP_3_SHAP = True
# 运行时间：60-120 分钟（取决于数据量）
```

### 场景D：仅优化超参数（高级）

```python
ENABLE_STEP_1_COMPARISON = False
ENABLE_STEP_2_PSO = True
ENABLE_STEP_3_SHAP = False
# 运行时间：30-60 分钟
```

---

## 🔧 常见调整

### 如果内存不足

```python
# 在 PSO 优化中减少样本数
pso_opt = PSO_DNN_LSTM_Optimizer(
    X_train_dl[:5000],        # 只用前 5000 个样本
    y_train_scaled[:5000],
    ...
)

# 或减少交叉验证折数
n_splits=5      # 改为 5 折
```

### 如果想要更精细的 SHAP 分析

```python
# 增加计算样本数
shap_analyzer = quick_shap_analysis(
    model_dl,
    X_test_dl,
    feature_cols,
    top_n=30,           # 显示前 30 个特征
    save_dir='results'
)
```

### 如果想加快 PSO 优化

```python
pso_opt = PSO_DNN_LSTM_Optimizer(
    ...,
    n_particles=10,     # 减少粒子数
    n_iterations=6,     # 减少迭代次数
    n_splits=5          # 减少折数
)
```

---

## ✅ 检查清单

集成前检查：
- [ ] 复制了 4 个新模块文件
- [ ] 修改了 main 文件的导入
- [ ] 添加了步骤开关
- [ ] 添加了三步骤的核心代码
- [ ] results/ 目录存在

运行后检查：
- [ ] 控制台有三步骤的输出信息
- [ ] results/ 中生成了相应的文件
- [ ] 对比表显示了三个模型的指标
- [ ] 特征重要性排名已生成

---

## 🆘 故障排除

### 问题：ImportError: No module named 'model_comparison'

**解决方案**：
```bash
# 确保模块文件在同一目录
ls model_comparison.py pso_optimization.py shap_analysis.py

# 或将模块路径添加到 Python path
import sys
sys.path.append('/path/to/modules')
```

### 问题：CUDA out of memory

**解决方案**：
```python
# 方案1：关闭 PSO 优化（最耗内存）
ENABLE_STEP_2_PSO = False

# 方案2：减少样本数
X_train_dl = X_train_dl[:5000]

# 方案3：强制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### 问题：SHAP 值全为 0

**解决方案**：
```python
# 使用排列特征重要性代替梯度法
from shap_analysis import SHAPAnalyzer
analyzer = SHAPAnalyzer(model_dl, X_test_dl, feature_cols)
importance = analyzer.compute_importance(method='permutation')
```

---

## 📞 获取帮���

1. **查看详细文档**：`IMPLEMENTATION.md`
2. **查看集成指南**：`INTEGRATION_GUIDE.py`
3. **查看模块注释**：各 `.py` 文件中的详细注释
4. **检查输出日志**：程序会输出详细的执行信息

---

## 🎓 学习路径

```
初级：学习如何使用
  ↓
【第1周】实现步骤1（模型对比）
  ↓
【第2周】实现步骤2（PSO优化）
  ↓
【第3周】实现步骤3（SHAP分析）
  ↓
进阶：优化和定制
  ↓
修改超参数、调整算法、添加新功能
```

---

**总耗时**：集成 5 分钟 + 运行 5-120 分钟（取决于配置）

**难度**：⭐ 简单（只需复制粘贴 + 修改开关）

**效果**：⭐⭐⭐⭐⭐ 显著（完整的模型分析系统）
"""

if __name__ == "__main__":
    print(QUICK_START)
    
    # 保存为文件
    with open('QUICK_START.md', 'w', encoding='utf-8') as f:
        f.write(QUICK_START)
    
    print("\n✅ 快速开始指南已生成: QUICK_START.md")
