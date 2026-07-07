import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold


class PSO_DNN_LSTM_Optimizer:
    """
    PSO 超参数优化器 - 修复版

    优化参数：
    - lstm_units: LSTM 单元数 (64-256)
    - dense_units: Dense 单元数 (32-128)
    - dropout_rate: Dropout 比率 (0.1-0.5)
    - learning_rate: 学习率 (1e-5 到 1e-3)
    """

    # 超参数范围表（用于论文输出）
    PARAM_RANGES = {
        'lstm_units': {'min': 64, 'max': 256, 'default': 128},
        'dense_units': {'min': 32, 'max': 128, 'default': 64},
        'dropout_rate': {'min': 0.1, 'max': 0.5, 'default': 0.2},
        'learning_rate': {'min': 1e-5, 'max': 1e-3, 'default': 1e-4},
    }

    def __init__(self, X_train, y_train, X_scaler, y_scaler,
                 n_splits=10, n_particles=20, n_iterations=12,
                 random_state=42, time_steps=96, out_dim=2,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
        """
        初始化 PSO 优化器

        Args:
            w_max: 最大惯性权重 (默认 0.9)
            w_min: 最小惯性权重 (默认 0.4)
            c1: 认知权重 (默认 2.0)
            c2: 社会权重 (默认 2.0)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.n_splits = n_splits
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.time_steps = time_steps
        self.out_dim = out_dim
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.best_params = None
        self.best_score = float('inf')
        self.optimization_history = []

        print(f"[PSO] 初始化完成")
        print(f"      训练样本数: {len(X_train)}")
        print(f"      交叉验证折数: {n_splits}")
        print(f"      PSO 参数: particles={n_particles}, iterations={n_iterations}")
        print(f"      惯性权重范围: [{w_min}, {w_max}], c1={c1}, c2={c2}")

    def build_model(self, lstm_units, dense_units, dropout_rate, learning_rate):
        """
        构建 cuDNN 兼容的 DNN+LSTM 模型
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Input(shape=(self.time_steps, self.X_train.shape[2], self.X_train.shape[3])),
            tf.keras.layers.Reshape((self.time_steps, -1)),

            # 第一层 LSTM - cuDNN 兼容
            LSTM(lstm_units,
                 return_sequences=True,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 recurrent_dropout=0.0,
                 unroll=False,
                 use_bias=True,
                 implementation=2),
            Dropout(dropout_rate),

            # 第二层 LSTM
            LSTM(lstm_units // 2,
                 return_sequences=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 recurrent_dropout=0.0,
                 unroll=False,
                 use_bias=True,
                 implementation=2),
            Dropout(dropout_rate),

            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),

            Dense(dense_units // 2, activation='relu'),
            Dense(self.out_dim)
        ])

        optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def evaluate_on_fold(self, X_fold_train, X_fold_val, y_fold_train, y_fold_val,
                        lstm_units, dense_units, dropout_rate, learning_rate):
        """在单个折上评估模型"""
        try:
            model = self.build_model(lstm_units, dense_units, dropout_rate, learning_rate)

            model.fit(
                X_fold_train, y_fold_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_fold_val, y_fold_val),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                ]
            )

            y_pred_val = model.predict(X_fold_val, verbose=0)
            # 反归一化后计算 MAPE
            y_pred_inv = self.y_scaler.inverse_transform(y_pred_val)
            y_true_inv = self.y_scaler.inverse_transform(y_fold_val)
            mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-8))) * 100

            tf.keras.backend.clear_session()
            return mape

        except Exception as e:
            print(f"[警告] 折评估出错: {e}")
            return float('inf')

    def objective_function(self, params):
        """
        PSO 目标函数
        params: [lstm_units_norm, dense_units_norm, dropout_norm, lr_norm] in [0,1]
        """
        lstm_units = int(params[0] * 192 + 64)  # 64-256
        dense_units = int(params[1] * 96 + 32)   # 32-128
        dropout_rate = params[2] * 0.4 + 0.1     # 0.1-0.5
        learning_rate = 10 ** (-params[3] * 2 - 3)  # 1e-5 to 1e-3

        cv_mape_scores = []
        for train_idx, val_idx in self.kfold.split(self.X_train):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]

            mape = self.evaluate_on_fold(
                X_fold_train, X_fold_val, y_fold_train, y_fold_val,
                lstm_units, dense_units, dropout_rate, learning_rate
            )
            cv_mape_scores.append(mape)

        mean_mape = np.mean(cv_mape_scores)
        return mean_mape

    def simple_pso(self):
        """
        PSO 算法 - 动态惯性权重版本
        """
        np.random.seed(self.random_state)

        # 初始化粒子 (4维)
        positions = np.random.rand(self.n_particles, 4)
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, 4))

        best_positions = positions.copy()
        best_scores = np.array([float('inf')] * self.n_particles)

        global_best_position = positions[0].copy()
        global_best_score = float('inf')

        print(f"\n[PSO] 开始优化...")
        print(f"      动态惯性权重: w_max={self.w_max}, w_min={self.w_min}")
        print(f"      认知权重 c1={self.c1}, 社会权重 c2={self.c2}")

        for iteration in range(self.n_iterations):
            # 动态惯性权重：线性递减
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations

            print(f"\n[PSO] 迭代 {iteration+1}/{self.n_iterations} (w={w:.3f})")

            for particle_idx in range(self.n_particles):
                score = self.objective_function(positions[particle_idx])
                print(f"      粒子 {particle_idx+1}/{self.n_particles}: MAPE={score:.4f}%")

                if score < best_scores[particle_idx]:
                    best_scores[particle_idx] = score
                    best_positions[particle_idx] = positions[particle_idx].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[particle_idx].copy()

            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_score': global_best_score,
                'avg_score': np.mean(best_scores),
                'w': w
            })

            print(f"      全局最佳 MAPE: {global_best_score:.4f}%")

            # 更新速度和位置
            for particle_idx in range(self.n_particles):
                r1 = np.random.rand(4)
                r2 = np.random.rand(4)

                velocities[particle_idx] = (
                    w * velocities[particle_idx] +
                    self.c1 * r1 * (best_positions[particle_idx] - positions[particle_idx]) +
                    self.c2 * r2 * (global_best_position - positions[particle_idx])
                )

                positions[particle_idx] = positions[particle_idx] + velocities[particle_idx]
                positions[particle_idx] = np.clip(positions[particle_idx], 0, 1)

        self.best_score = global_best_score

        # 反归一化最优参数
        self.best_params = {
            'lstm_units': int(global_best_position[0] * 192 + 64),
            'dense_units': int(global_best_position[1] * 96 + 32),
            'dropout_rate': round(global_best_position[2] * 0.4 + 0.1, 4),
            'learning_rate': float(10 ** (-global_best_position[3] * 2 - 3))
        }

        return self.best_score, self.best_params

    def optimize(self):
        """执行 PSO 优化"""
        score, params = self.simple_pso()

        print(f"\n[PSO] 优化完成!")
        print(f"      最佳 MAPE: {score:.4f}%")
        print(f"      最优参数:")
        for k, v in params.items():
            print(f"        {k}: {v}")

        return params, score, self.optimization_history

    def save_results(self, save_dir='results'):
        """保存优化结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存最优参数
        with open(f'{save_dir}/02_pso_best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)

        # 保存优化历史
        history_df = pd.DataFrame(self.optimization_history)
        history_df.to_csv(f'{save_dir}/02_pso_optimization_history.csv', index=False)

        # 生成优化过程图（论文图10风格）
        self.plot_optimization_history(f'{save_dir}/02_pso_optimization_curve.png')

        # 生成超参数表（论文表4风格）
        self.plot_hyperparameter_table(f'{save_dir}/02_pso_hyperparameter_table.png')

        print(f"[PSO] 结果已保存到 {save_dir}/")

    def plot_optimization_history(self, save_path):
        """绘制优化过程曲线 - 论文图10风格"""
        if not self.optimization_history:
            return

        history_df = pd.DataFrame(self.optimization_history)

        plt.figure(figsize=(10, 6))
        plt.plot(history_df['iteration'], history_df['best_score'],
                marker='*', markersize=12, linewidth=1.5,
                linestyle='--', color='purple', label='Best Fitness')

        plt.xlabel('iter', fontsize=12)
        plt.ylabel('fitness', fontsize=12)
        plt.title('PSO Optimization Curve', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[图表] 优化曲线已保存: {save_path}")

    def plot_hyperparameter_table(self, save_path):
        """绘制超参数表 - 论文表4风格"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')

        # 构建表格数据
        table_data = []
        for param_name, ranges in self.PARAM_RANGES.items():
            row = [
                param_name,
                f"{ranges['min']}-{ranges['max']}",
                str(ranges['default']),
                str(self.best_params.get(param_name, 'N/A'))
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data,
            colLabels=['Parameter', 'Range', 'Default Value', 'Optimized Parameter Values'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Table 4. Hyperparameter selection.', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[图表] 超参数表已保存: {save_path}")


if __name__ == "__main__":
    print("PSO 超参数优化模块已加载，可在主程序中调用")
