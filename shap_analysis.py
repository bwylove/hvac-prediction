import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class SHAPAnalyzer:
    """
    SHAP 特征重要性分析器 - 时间序列专用版

    对 3D/4D 输入 (samples, time_steps, features, [channels])
    沿时间维度取平均后计算 SHAP 值
    """

    def __init__(self, model, X_sample, feature_names, model_type='keras'):
        self.model = model
        self.X_sample = X_sample
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_values = None
        self.shap_importance = None

        print(f"[SHAP] 初始化分析器")
        print(f"      原始样本形状: {X_sample.shape}")
        print(f"      特征数: {len(feature_names)}")
        print(f"      模型类型: {model_type}")

    def _flatten_time_dimension(self, X):
        """将时间维度压缩为特征维度，用于 SHAP 计算"""
        if X.ndim == 4:  # (samples, time_steps, features, 1)
            # 对时间维度取平均 -> (samples, features)
            return X.mean(axis=1).reshape(X.shape[0], -1)
        elif X.ndim == 3:  # (samples, time_steps, features)
            return X.mean(axis=1)
        else:
            return X

    def _restore_shape(self, X_flat, original_shape):
        """将展平后的数据恢复为原始形状"""
        if len(original_shape) == 4:
            # (samples, features) -> (samples, time_steps, features, 1)
            n_samples = X_flat.shape[0]
            time_steps = original_shape[1]
            n_features = original_shape[2]
            X_restored = np.zeros((n_samples, time_steps, n_features, 1))
            for t in range(time_steps):
                X_restored[:, t, :, 0] = X_flat
            return X_restored
        elif len(original_shape) == 3:
            n_samples = X_flat.shape[0]
            time_steps = original_shape[1]
            n_features = original_shape[2]
            X_restored = np.zeros((n_samples, time_steps, n_features))
            for t in range(time_steps):
                X_restored[:, t, :] = X_flat
            return X_restored
        else:
            return X_flat

    def compute_shap_values_gradient(self, X_subset, n_samples=100):
        """
        使用梯度方法计算 SHAP 值（时间序列专用）

        对时间维度取平均后计算每个特征的 SHAP 值
        """
        print("[SHAP] 计算梯度方法 SHAP 值...")

        X_use = X_subset[:n_samples]
        n_samples_actual = len(X_use)
        original_shape = X_use.shape

        # 压缩时间维度
        X_2d = self._flatten_time_dimension(X_use)
        n_features = X_2d.shape[1]

        # 检查特征数是否匹配
        if n_features != len(self.feature_names):
            print(f"[警告] 特征数不匹配: 数据有 {n_features} 列, feature_names 有 {len(self.feature_names)} 个")
            # 取最小值
            n_features = min(n_features, len(self.feature_names))
            X_2d = X_2d[:, :n_features]

        shap_vals = np.zeros((n_samples_actual, n_features))
        epsilon = 1e-4

        for feat_idx in range(n_features):
            if feat_idx % 10 == 0 or feat_idx == n_features - 1:
                print(f"  特征 {feat_idx+1}/{n_features}", end='\r')

            X_plus = X_2d.copy()
            X_minus = X_2d.copy()
            X_plus[:, feat_idx] += epsilon
            X_minus[:, feat_idx] -= epsilon

            # 恢复原始形状用于模型预测
            X_plus_orig = self._restore_shape(X_plus, original_shape)
            X_minus_orig = self._restore_shape(X_minus, original_shape)

            try:
                pred_plus = self.model.predict(X_plus_orig, verbose=0)
                pred_minus = self.model.predict(X_minus_orig, verbose=0)

                # 对输出取平均（多目标回归）
                if pred_plus.ndim > 1:
                    pred_plus_mean = pred_plus.mean(axis=1)
                    pred_minus_mean = pred_minus.mean(axis=1)
                else:
                    pred_plus_mean = pred_plus
                    pred_minus_mean = pred_minus

                shap_vals[:, feat_idx] = (pred_plus_mean - pred_minus_mean) / (2 * epsilon)

            except Exception as e:
                print(f"\n[警告] 特征 {feat_idx} 计算失败: {e}")
                shap_vals[:, feat_idx] = 0

        print(f"\n  特征 {n_features}/{n_features} [完成]")
        return shap_vals

    def compute_shap_values_permutation(self, X_subset, n_samples=100):
        """
        使用排列特征重要性计算近似 SHAP 值
        """
        print("[SHAP] 计算排列特征重要性方法...")

        X_use = X_subset[:n_samples]
        n_samples_actual = len(X_use)
        original_shape = X_use.shape

        X_2d = self._flatten_time_dimension(X_use)
        n_features = min(X_2d.shape[1], len(self.feature_names))
        X_2d = X_2d[:, :n_features]

        # 基准预测
        baseline_pred = self.model.predict(X_use, verbose=0)
        if baseline_pred.ndim > 1:
            baseline_pred_mean = baseline_pred.mean(axis=1)
        else:
            baseline_pred_mean = baseline_pred

        shap_vals = np.zeros((n_samples_actual, n_features))

        for feat_idx in range(n_features):
            if feat_idx % 10 == 0 or feat_idx == n_features - 1:
                print(f"  特征 {feat_idx+1}/{n_features}", end='\r')

            X_permuted = X_2d.copy()
            np.random.shuffle(X_permuted[:, feat_idx])

            X_permuted_orig = self._restore_shape(X_permuted, original_shape)

            try:
                pred_permuted = self.model.predict(X_permuted_orig, verbose=0)
                if pred_permuted.ndim > 1:
                    pred_permuted_mean = pred_permuted.mean(axis=1)
                else:
                    pred_permuted_mean = pred_permuted

                shap_vals[:, feat_idx] = baseline_pred_mean - pred_permuted_mean
            except Exception as e:
                print(f"\n[警告] 特征 {feat_idx} 计算失败: {e}")
                shap_vals[:, feat_idx] = 0

        print(f"\n  特征 {n_features}/{n_features} [完成]")
        return shap_vals

    def compute_importance(self, n_samples=100, method='gradient'):
        """
        计算特征的平均绝对 SHAP 值作为重要性指标
        """
        print(f"\n[SHAP] 计算特征重要性 (方法: {method})...")

        if method == 'gradient':
            self.shap_values = self.compute_shap_values_gradient(self.X_sample, n_samples)
        else:
            self.shap_values = self.compute_shap_values_permutation(self.X_sample, n_samples)

        # 计算平均绝对 SHAP 值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        std_abs_shap = np.abs(self.shap_values).std(axis=0)

        # 构建 DataFrame
        n_features_actual = len(mean_abs_shap)
        feature_names_use = self.feature_names[:n_features_actual]

        self.shap_importance = pd.DataFrame({
            '特征': feature_names_use,
            '平均绝对SHAP值': mean_abs_shap,
            'SHAP值标准差': std_abs_shap
        }).sort_values('平均绝对SHAP值', ascending=False).reset_index(drop=True)

        self.shap_importance['重要性排名'] = range(1, len(self.shap_importance) + 1)

        print(f"\n[SHAP] 特征重要性排名 (Top 20):")
        print(self.shap_importance.head(20).to_string(index=False))

        return self.shap_importance

    def plot_summary(self, save_path='results/03_shap_summary.png', top_n=20):
        """
        绘制 SHAP summary plot（蜂群图）- 类似论文图12a
        """
        if self.shap_values is None or self.shap_importance is None:
            print("[警告] 未计算 SHAP 值，请先调用 compute_importance()")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 取前 N 个特征
        top_n = min(top_n, len(self.shap_importance))
        top_features = self.shap_importance.head(top_n)
        top_indices = [self.feature_names.index(f) for f in top_features['特征']]

        # 获取对应的 SHAP 值和特征值
        shap_vals_top = self.shap_values[:, top_indices]

        # 获取特征值（用于着色）
        X_2d = self._flatten_time_dimension(self.X_sample[:len(self.shap_values)])
        feature_vals_top = X_2d[:, top_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        # 绘制蜂群图
        y_pos = np.arange(top_n)
        for i in range(top_n):
            feat_vals = feature_vals_top[:, i]
            shap_vals_feat = shap_vals_top[:, i]

            # 按特征值排序用于着色
            sort_idx = np.argsort(feat_vals)
            feat_vals_sorted = feat_vals[sort_idx]
            shap_vals_sorted = shap_vals_feat[sort_idx]

            # 归一化特征值用于颜色
            norm = plt.Normalize(feat_vals_sorted.min(), feat_vals_sorted.max())
            colors = plt.cm.RdYlBu_r(norm(feat_vals_sorted))

            # 添加抖动避免重叠
            jitter = np.random.normal(0, 0.02, len(shap_vals_sorted))
            ax.scatter(shap_vals_sorted, np.full(len(shap_vals_sorted), i) + jitter,
                      c=colors, s=20, alpha=0.6, edgecolors='none')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['特征'])
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
        ax.set_title('SHAP Feature Importance (Summary Plot)', fontsize=14)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(
            feature_vals_top.min(), feature_vals_top.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Feature value', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表] SHAP summary plot 已保存: {save_path}")

    def plot_importance_bar(self, save_path='results/03_shap_feature_importance.png', top_n=20):
        """
        绘制 SHAP 特征重要性条形图 - 类似论文图12b
        """
        if self.shap_importance is None:
            print("[警告] 未计算特征重要性，请先调用 compute_importance()")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        top_features = self.shap_importance.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))

        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['平均绝对SHAP值'], color='#1E90FF', height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['特征'])
        ax.invert_yaxis()  # 重要性最高的在顶部
        ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=11)
        ax.set_title('SHAP Feature Importance', fontsize=14)
        ax.grid(axis='x', alpha=0.3)

        # 在条形右侧添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_features['平均绝对SHAP值'])):
            ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表] SHAP 特征重要性条形图已保存: {save_path}")

    def save_results(self, save_dir='results'):
        """保存分析结果"""
        os.makedirs(save_dir, exist_ok=True)

        if self.shap_importance is not None:
            self.shap_importance.to_csv(
                f'{save_dir}/03_shap_feature_importance.csv',
                index=False
            )
            print(f"[表格] 特征重要性已保存: {save_dir}/03_shap_feature_importance.csv")

        # 保存 SHAP 值样本
        if self.shap_values is not None:
            n_save = min(100, self.shap_values.shape[0])
            n_features = self.shap_values.shape[1]
            feature_names_use = self.feature_names[:n_features]
            shap_df = pd.DataFrame(
                self.shap_values[:n_save],
                columns=feature_names_use
            )
            shap_df.to_csv(f'{save_dir}/03_shap_values_sample.csv', index=False)
            print(f"[表格] SHAP 值样本已保存: {save_dir}/03_shap_values_sample.csv")


def quick_shap_analysis(model, X_test, feature_names, top_n=15, save_dir='results', n_samples=100):
    """
    快速执行 SHAP 分析（修复版）

    Args:
        model: 训练好的 Keras 模型
        X_test: 测试数据 (n_samples, time_steps, n_features, 1) 或 (n_samples, time_steps, n_features)
        feature_names: 特征名称列表
        top_n: 显示前 N 个特征
        save_dir: 保存目录
        n_samples: 用于计算的样本数

    Returns:
        SHAPAnalyzer: 分析器对象
    """
    print("\n" + "=" * 60)
    print("【步骤3】SHAP 特征重要性分析")
    print("=" * 60)

    analyzer = SHAPAnalyzer(model, X_test, feature_names, model_type='keras')

    # 计算特征重要性
    importance_df = analyzer.compute_importance(n_samples=min(n_samples, len(X_test)), method='gradient')

    # 绘制图表（类似论文图12）
    analyzer.plot_summary(f'{save_dir}/03_shap_summary.png', top_n=top_n)
    analyzer.plot_importance_bar(f'{save_dir}/03_shap_feature_importance.png', top_n=top_n)

    # 保存结果
    analyzer.save_results(save_dir)

    return analyzer


if __name__ == "__main__":
    print("SHAP 特征重要性分析模块已加载，可在主程序中调用")
