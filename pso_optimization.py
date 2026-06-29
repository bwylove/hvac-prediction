#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2：PSO 超参数优化
使用粒子群算法 (PSO) 优化 DNN+LSTM 的超参数
采用 10 折交叉验证，以 MAPE 为优化目标
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import json
import os


class PSO_DNN_LSTM_Optimizer:
    """
    PSO 超参数优化器
    
    优化参数：
    - lstm_units: LSTM 单元数 (64-256)
    - dense_units: Dense 单元数 (32-128)
    - dropout_rate: Dropout 比率 (0.1-0.5)
    - learning_rate: 学习率 (1e-5 到 1e-4)
    """
    
    def __init__(self, X_train, y_train, X_scaler, y_scaler, 
                 n_splits=10, n_particles=20, n_iterations=12, 
                 random_state=42, time_steps=96, out_dim=2):
        """
        初始化 PSO 优化器
        
        Args:
            X_train: 训练特征 (n_samples, time_steps, n_features, 1)
            y_train: 训练标签 (n_samples, out_dim)
            X_scaler, y_scaler: 数据缩放器
            n_splits: 交叉验证折数 (默认 10)
            n_particles: PSO 粒子数 (默认 20)
            n_iterations: PSO 迭代次数 (默认 12)
            random_state: 随机种子
            time_steps: 时间步长
            out_dim: 输出维度
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
        
        # 初始化 KFold
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # 最优参数和分数
        self.best_params = None
        self.best_score = float('inf')
        self.optimization_history = []
        
        print(f"[PSO] 初始化完成")
        print(f"      训练样本数: {len(X_train)}")
        print(f"      交叉验证折数: {n_splits}")
        print(f"      PSO 参数: particles={n_particles}, iterations={n_iterations}")
    
    def build_model(self, lstm_units, dense_units, dropout_rate, learning_rate):
        """
        构建 DNN+LSTM 模型
        
        Args:
            lstm_units: LSTM 单元数
            dense_units: Dense 层单元数
            dropout_rate: Dropout 比率
            learning_rate: 学习率
        
        Returns:
            编译后的 Keras 模型
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import Huber
        
        model = Sequential([
            Input(shape=(self.time_steps, self.X_train.shape[2], self.X_train.shape[3])),
            
            # Reshape 为 (batch, time_steps, features)
            tf.keras.layers.Reshape((self.time_steps, -1)),
            
            # 第一层 LSTM
            LSTM(lstm_units, return_sequences=True, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # 第二层 LSTM
            LSTM(lstm_units // 2, return_sequences=False, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense 层
            Dense(dense_units, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(dense_units // 2, activation='relu'),
            
            # 输出层
            Dense(self.out_dim)
        ])
        
        optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)
        model.compile(optimizer=optimizer, loss=Huber(delta=1.5), metrics=['mae'])
        
        return model
    
    def evaluate_on_fold(self, X_fold_train, X_fold_val, y_fold_train, y_fold_val,
                        lstm_units, dense_units, dropout_rate, learning_rate, verbose=0):
        """
        在单个折上评估模型
        
        Args:
            X_fold_train, y_fold_train: 折的训练数据
            X_fold_val, y_fold_val: 折的验证数据
            其他：超参数
            verbose: 是否打印训练过程
        
        Returns:
            float: 验证集上的 MAPE
        """
        try:
            model = self.build_model(lstm_units, dense_units, dropout_rate, learning_rate)
            
            # 训练（少轮次加速）
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
            
            # 评估
            y_pred_val = model.predict(X_fold_val, verbose=0)
            mape = mean_absolute_percentage_error(y_fold_val, y_pred_val)
            
            tf.keras.backend.clear_session()
            return mape
        
        except Exception as e:
            print(f"[警告] 折评估出错: {e}")
            return float('inf')
    
    def objective_function(self, params):
        """
        PSO 目标函数：计算交叉验证 MAPE
        
        Args:
            params: [lstm_units_norm, dense_units_norm, dropout_norm, lr_norm]
                   (0-1 范围内的归一化参数)
        
        Returns:
            float: 平均 MAPE（越小越好）
        """
        # 反归一化参数
        lstm_units = int(params[0] * 192 + 64)  # 64-256
        dense_units = int(params[1] * 96 + 32)  # 32-128
        dropout_rate = params[2] * 0.4 + 0.1    # 0.1-0.5
        learning_rate = 10 ** (-params[3] * 1 - 4)  # 1e-5 to 1e-4
        
        cv_mape_scores = []
        fold_idx = 0
        
        # K 折交叉验证
        for train_idx, val_idx in self.kfold.split(self.X_train):
            fold_idx += 1
            
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
        简单实现的 PSO 算法（不依赖外部库）
        
        Returns:
            tuple: (best_score, best_params)
        """
        np.random.seed(self.random_state)
        
        # 初始化粒子位置和速度 (4 维：lstm, dense, dropout, lr)
        positions = np.random.rand(self.n_particles, 4)
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, 4))
        
        # 记录每个粒子的最佳位置
        best_positions = positions.copy()
        best_scores = np.array([float('inf')] * self.n_particles)
        
        # 全局最佳
        global_best_position = positions[0].copy()
        global_best_score = float('inf')
        
        # PSO 参数
        w = 0.9  # 惯性权重
        c1 = 0.5  # 认知参数
        c2 = 0.3  # 社会参数
        
        print(f"\n[PSO] 开始优化...")
        print(f"      惯性权重 w={w}, c1={c1}, c2={c2}")
        
        # 迭代
        for iteration in range(self.n_iterations):
            print(f"\n[PSO] 迭代 {iteration+1}/{self.n_iterations}")
            
            for particle_idx in range(self.n_particles):
                # 评估当前位置
                score = self.objective_function(positions[particle_idx])
                print(f"      粒子 {particle_idx+1}/{self.n_particles}: MAPE={score:.4f}")
                
                # 更新粒子的最佳分数
                if score < best_scores[particle_idx]:
                    best_scores[particle_idx] = score
                    best_positions[particle_idx] = positions[particle_idx].copy()
                
                # 更新全局最佳
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[particle_idx].copy()
            
            # 记录历史
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_score': global_best_score,
                'avg_score': np.mean(best_scores)
            })
            
            print(f"      全局最佳 MAPE: {global_best_score:.4f}")
            
            # 更新速度和位置
            for particle_idx in range(self.n_particles):
                r1 = np.random.rand(4)
                r2 = np.random.rand(4)
                
                velocities[particle_idx] = (
                    w * velocities[particle_idx] +
                    c1 * r1 * (best_positions[particle_idx] - positions[particle_idx]) +
                    c2 * r2 * (global_best_position - positions[particle_idx])
                )
                
                positions[particle_idx] = positions[particle_idx] + velocities[particle_idx]
                
                # 限制位置在 [0, 1]
                positions[particle_idx] = np.clip(positions[particle_idx], 0, 1)
        
        self.best_score = global_best_score
        
        # 反归一化最优参数
        self.best_params = {
            'lstm_units': int(global_best_position[0] * 192 + 64),
            'dense_units': int(global_best_position[1] * 96 + 32),
            'dropout_rate': round(global_best_position[2] * 0.4 + 0.1, 4),
            'learning_rate': float(10 ** (-global_best_position[3] * 1 - 4))
        }
        
        return self.best_score, self.best_params
    
    def optimize(self):
        """
        执行 PSO 优化
        
        Returns:
            tuple: (best_params, best_score, optimization_history)
        """
        score, params = self.simple_pso()
        
        print(f"\n[PSO] 优化完成!")
        print(f"      最佳 MAPE: {score:.4f}")
        print(f"      最优参数:")
        for k, v in params.items():
            print(f"        {k}: {v}")
        
        return params, score, self.optimization_history
    
    def save_results(self, save_dir='results'):
        """
        保存优化结果
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存最优参数
        with open(f'{save_dir}/02_pso_best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # 保存优化历史
        history_df = pd.DataFrame(self.optimization_history)
        history_df.to_csv(f'{save_dir}/02_pso_optimization_history.csv', index=False)
        
        # 生成优化过程图
        self.plot_optimization_history(f'{save_dir}/02_pso_optimization_curve.png')
        
        print(f"[PSO] 结果已保存到 {save_dir}/")
    
    def plot_optimization_history(self, save_path):
        """
        绘制优化过程曲线
        
        Args:
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        if not self.optimization_history:
            return
        
        history_df = pd.DataFrame(self.optimization_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['iteration'], history_df['best_score'], 
                marker='o', linewidth=2, label='全局最佳 MAPE')
        plt.plot(history_df['iteration'], history_df['avg_score'], 
                marker='s', linewidth=2, linestyle='--', label='平均 MAPE')
        
        plt.xlabel('迭代次数')
        plt.ylabel('MAPE (%)')
        plt.title('PSO 超参数优化过程')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"[图表] 优化曲线已保存: {save_path}")


if __name__ == "__main__":
    print("PSO 超参数优化模块已加载，可在主程序中调用")
