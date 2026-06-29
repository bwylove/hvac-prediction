#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3：SHAP 特征重要性分析
计算每个特征的 SHAP 值，评估对模型预测的贡献
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class SHAPAnalyzer:
    """
    SHAP 特征重要性分析器
    
    支持 Keras/TensorFlow 和 scikit-learn 模型
    """
    
    def __init__(self, model, X_sample, feature_names, model_type='keras'):
        """
        初始化 SHAP 分析器
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据 (n_samples, ...)
            feature_names: 特征名称列表
            model_type: 模型类型 ('keras' 或 'sklearn')
        """
        self.model = model
        self.X_sample = X_sample
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_values = None
        self.shap_importance = None
        
        print(f"[SHAP] 初始化���析器")
        print(f"      样本数: {len(X_sample)}")
        print(f"      特征数: {len(feature_names)}")
        print(f"      模型类型: {model_type}")
    
    def compute_shap_values_gradient(self, X_subset, n_samples=100):
        """
        使用梯度方法计算 SHAP 值（用于深度学习模型）
        
        基于特征重要性的数值梯度近似
        
        Args:
            X_subset: 用于计算的数据子集
            n_samples: 使用的样本数
        
        Returns:
            np.ndarray: SHAP 值 (n_samples, n_features)
        """
        print("[SHAP] 计算梯度方法 SHAP 值...")
        
        X_use = X_subset[:n_samples]
        n_samples_actual = len(X_use)
        
        # Reshape 为 2D (n_samples, features)
        if X_use.ndim > 2:
            n_samples_actual = X_use.shape[0]
            X_2d = X_use.reshape(n_samples_actual, -1)
        else:
            X_2d = X_use
        
        n_features = X_2d.shape[1]
        shap_vals = np.zeros((n_samples_actual, n_features))
        
        epsilon = 1e-4
        
        for feat_idx in range(n_features):
            print(f"  特征 {feat_idx+1}/{n_features}", end='\r')
            
            X_plus = X_2d.copy()
            X_minus = X_2d.copy()
            
            X_plus[:, feat_idx] += epsilon
            X_minus[:, feat_idx] -= epsilon
            
            # 重新 reshape 为原始形状
            if X_use.ndim > 2:
                X_plus = X_plus.reshape(X_use.shape)
                X_minus = X_minus.reshape(X_use.shape)
            
            try:
                pred_plus = self.model.predict(X_plus, verbose=0)
                pred_minus = self.model.predict(X_minus, verbose=0)
                
                # 计算梯度（取平均预测值）
                if pred_plus.ndim > 1:
                    pred_plus_mean = pred_plus.mean(axis=1)
                    pred_minus_mean = pred_minus.mean(axis=1)
                else:
                    pred_plus_mean = pred_plus
                    pred_minus_mean = pred_minus
                
                shap_vals[:, feat_idx] = (pred_plus_mean - pred_minus_mean) / (2 * epsilon)
            
            except Exception as e:
                print(f"[警告] 特征 {feat_idx} 计算失败: {e}")
                shap_vals[:, feat_idx] = 0
        
        print(f"  特征 {n_features}/{n_features} [完成]")
        return shap_vals
    
    def compute_shap_values_permutation(self, X_subset, n_samples=100):
        """
        使用排列特征重要性计算近似 SHAP 值
        
        Args:
            X_subset: 用于计算的数据子集
            n_samples: 使用的样本数
        
        Returns:
            np.ndarray: SHAP 值 (n_samples, n_features)
        """
        print("[SHAP] 计算排列特征重要性方法...")
        
        X_use = X_subset[:n_samples]
        n_samples_actual = len(X_use)
        
        # Reshape 为 2D
        if X_use.ndim > 2:
            n_samples_actual = X_use.shape[0]
            X_2d = X_use.reshape(n_samples_actual, -1)
        else:
            X_2d = X_use
        
        # 基准预测
        baseline_pred = self.model.predict(X_use, verbose=0)
        if baseline_pred.ndim > 1:
            baseline_pred_mean = baseline_pred.mean(axis=1)
        else:
            baseline_pred_mean = baseline_pred
        
        n_features = X_2d.shape[1]
        shap_vals = np.zeros((n_samples_actual, n_features))
        
        for feat_idx in range(n_features):
            print(f"  特征 {feat_idx+1}/{n_features}", end='\r')
            
            X_permuted = X_2d.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            
            # Reshape 回原始形状
            if X_use.ndim > 2:
                X_permuted = X_permuted.reshape(X_use.shape)
            
            try:
                pred_permuted = self.model.predict(X_permuted, verbose=0)
                
                if pred_permuted.ndim > 1:
                    pred_permuted_mean = pred_permuted.mean(axis=1)
                else:
                    pred_permuted_mean = pred_permuted
                
                shap_vals[:, feat_idx] = baseline_pred_mean - pred_permuted_mean
            
            except Exception as e:
                print(f"[警告] 特征 {feat_idx} 计算失败: {e}")
                shap_vals[:, feat_idx] = 0
        
        print(f"  特征 {n_features}/{n_features} [完成]")
        return shap_vals
    
    def compute_importance(self, n_samples=100, method='gradient'):
        """
        计算特征的平均绝对 SHAP 值作为重要性指标
        
        Args:
            n_samples: 用于计算的样本数
            method: 计算方法 ('gradient' 或 'permutation')
        
        Returns:
            pd.DataFrame: 特征重要性排名表
        """
        print(f"\n[SHAP] 计算特征重要性 (方法: {method})...")
        
        if method == 'gradient':
            self.shap_values = self.compute_shap_values_gradient(self.X_sample, n_samples)
        else:  # permutation
            self.shap_values = self.compute_shap_values_permutation(self.X_sample, n_samples)
        
        # 计算平均绝对 SHAP 值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # 构建 DataFrame
        self.shap_importance = pd.DataFrame({
            '特征': self.feature_names,
            '平均绝对SHAP值': mean_abs_shap,
            'SHAP值标准差': np.abs(self.shap_values).std(axis=0)
        }).sort_values('平均绝对SHAP值', ascending=False).reset_index(drop=True)
        
        self.shap_importance['重要性排名'] = range(1, len(self.shap_importance) + 1)
        
        print(f"\n[SHAP] 特征重要性排名:")
        print(self.shap_importance.to_string(index=False))
        
        return self.shap_importance
    
    def plot_importance(self, save_path='results/03_shap_feature_importance.png', top_n=20):
        """
        绘制特征重要性条形图
        
        Args:
            save_path: 保存路径
            top_n: 显示前 N 个特征
        """
        if self.shap_importance is None:
            print("[警告] 未计算特征重要性，请先调用 compute_importance()")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 取前 N 个特征
        top_features = self.shap_importance.head(top_n)
        
        plt.figure(figsize=(10, max(6, len(top_features) * 0.25)))
        plt.barh(range(len(top_features)), top_features['平均绝对SHAP值'], 
                color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['特征'])
        plt.xlabel('平均绝对 SHAP 值')
        plt.title(f'SHAP 特征重要性排名 (Top {top_n})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[图表] 特征重要性图已保存: {save_path}")
    
    def plot_shap_distribution(self, save_path='results/03_shap_value_distribution.png', top_n=10):
        """
        绘制 SHAP 值分布图（前 N 个特征）
        
        Args:
            save_path: 保存路径
            top_n: 显示前 N 个特征
        """
        if self.shap_values is None:
            print("[警告] 未计算 SHAP 值，请先调用 compute_importance()")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 获取前 N 个特征的索引
        top_indices = self.shap_importance.head(top_n).index.tolist()
        top_names = self.shap_importance.head(top_n)['特征'].tolist()
        
        fig, axes = plt.subplots(top_n, 1, figsize=(10, 3 * top_n))
        if top_n == 1:
            axes = [axes]
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            shap_vals_feat = self.shap_values[:, feat_idx]
            
            ax.hist(shap_vals_feat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{top_names[idx]} 的 SHAP 值分布')
            ax.set_xlabel('SHAP 值')
            ax.set_ylabel('频数')
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[图表] SHAP 值分布图已保存: {save_path}")
    
    def save_results(self, save_dir='results'):
        """
        保存分析结果
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.shap_importance is not None:
            self.shap_importance.to_csv(
                f'{save_dir}/03_shap_feature_importance.csv', 
                index=False
            )
            print(f"[表格] 特征重要性已保存: {save_dir}/03_shap_feature_importance.csv")
        
        # 保存 SHAP 值（仅前 100 个样本以节省空间）
        if self.shap_values is not None:
            n_save = min(100, self.shap_values.shape[0])
            shap_df = pd.DataFrame(
                self.shap_values[:n_save],
                columns=self.feature_names
            )
            shap_df.to_csv(f'{save_dir}/03_shap_values_sample.csv', index=False)
            print(f"[表格] SHAP 值样本已保存: {save_dir}/03_shap_values_sample.csv")


def quick_shap_analysis(model, X_test, feature_names, top_n=15, save_dir='results'):
    """
    快速执行 SHAP 分析
    
    Args:
        model: 训练好的模型
        X_test: 测试数据
        feature_names: 特征名称列表
        top_n: 显示前 N 个特征
        save_dir: 保存目录
    
    Returns:
        SHAPAnalyzer: 分析器对象
    """
    print("\n" + "="*60)
    print("【步骤3】SHAP 特征重要性分析")
    print("="*60)
    
    analyzer = SHAPAnalyzer(model, X_test, feature_names, model_type='keras')
    
    # 计算特征重要性
    importance_df = analyzer.compute_importance(n_samples=min(100, len(X_test)), method='gradient')
    
    # 绘制图表
    analyzer.plot_importance(f'{save_dir}/03_shap_feature_importance.png', top_n=top_n)
    analyzer.plot_shap_distribution(f'{save_dir}/03_shap_value_distribution.png', top_n=min(10, top_n))
    
    # 保存结果
    analyzer.save_results(save_dir)
    
    return analyzer


if __name__ == "__main__":
    print("SHAP 特征重要性分析模块已加载，可在主程序中调用")
