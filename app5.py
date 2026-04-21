import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import streamlit as st
import io
from datetime import datetime

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class VibrationLocatorFromPhase:
    """从解调好的相位信号进行振动定位的系统"""
    
    def __init__(self, fs=None, c=3e8, n=1.5):
        """
        初始化参数
        
        参数:
            fs: 采样率 (Hz)
            c: 光速 (m/s)，默认3e8
            n: 光纤折射率，默认1.5
        """
        self.fs = fs
        self.c = c
        self.n = n
        
    def load_phase_data(self, file_path, file_name="相位数据"):
        """
        从Excel文件加载相位数据（时间+相位）
        
        参数:
            file_path: 文件路径
            file_name: 文件名（用于调试信息）
            
        返回:
            t: 时间数组
            phase: 相位数组
        """
        try:
            # 调试：显示文件信息
            st.info(f"正在加载{file_name}...")
            
            # 首先尝试用header=0读取（有表头）
            df = pd.read_excel(file_path, header=0)
            
            # 调试：显示读取的数据信息
            st.write(f"**{file_name}信息:**")
            st.write(f"- 数据形状: {df.shape} (行×列)")
            st.write(f"- 列名: {list(df.columns)}")
            
            # 检查数据形状
            if df.shape[0] < 2:
                st.error(f"{file_name}行数不足，只有{df.shape[0]}行")
                return None, None
            
            if df.shape[1] < 2:
                st.error(f"{file_name}需要至少2列，但只有{df.shape[1]}列")
                st.write("请确保Excel文件包含至少2列：时间和相位")
                return None, None
            
            # 尝试识别时间列和相位列
            time_col = None
            phase_col = None
            
            # 方法1：根据列名识别
            for i, col in enumerate(df.columns):
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['时间', 'time', 't', 'timestamp']):
                    time_col = i
                elif any(keyword in col_lower for keyword in ['相位', 'phase', 'phi', '信号', 'signal', 'data']):
                    phase_col = i
            
            # 方法2：如果没有识别到，使用前两列
            if time_col is None or phase_col is None:
                st.warning(f"无法自动识别列名，将使用前两列作为时间和相位")
                time_col = 0
                phase_col = 1
            
            st.write(f"- 使用列{time_col}作为时间: {df.columns[time_col]}")
            st.write(f"- 使用列{phase_col}作为相位: {df.columns[phase_col]}")
            
            # 提取数据
            t = df.iloc[:, time_col].values
            phase = df.iloc[:, phase_col].values
            
            # 转换为浮点数
            try:
                t = t.astype(float)
                phase = phase.astype(float)
            except Exception as e:
                st.error(f"数据转换错误: {e}")
                st.write("请确保数据为数值类型，不包含非数值字符")
                return None, None
            
            # 检查数据是否有NaN
            if np.any(np.isnan(t)) or np.any(np.isnan(phase)):
                st.warning(f"{file_name}包含NaN值，将进行清理")
                valid_mask = ~(np.isnan(t) | np.isnan(phase))
                t = t[valid_mask]
                phase = phase[valid_mask]
            
            # 确保相位是实数
            phase = np.real(phase)
            
            # 调试：显示数据统计信息
            st.write(f"- 时间范围: {t[0]:.6f} 到 {t[-1]:.6f} s")
            st.write(f"- 时间间隔: {np.mean(np.diff(t)):.6f} s (平均)")
            st.write(f"- 相位范围: {np.min(phase):.6f} 到 {np.max(phase):.6f} rad")
            st.write(f"- 相位均值: {np.mean(phase):.6f} rad")
            st.write(f"- 相位标准差: {np.std(phase):.6f} rad")
            st.write(f"- 数据点数: {len(t)}")
            
            # 检查数据是否为常数
            if np.std(phase) < 1e-10:
                st.warning(f"警告: {file_name}的相位数据标准差极小 ({np.std(phase):.6f})，可能是一条直线！")
                
                # 显示前10个数据点
                st.write("**前10个数据点:**")
                preview_df = pd.DataFrame({
                    '时间(s)': t[:10],
                    '相位(rad)': phase[:10]
                })
                st.dataframe(preview_df)
            
            return t, phase
            
        except Exception as e:
            st.error(f"加载{file_name}时出错: {str(e)}")
            st.write("尝试使用备用方法读取...")
            
            # 备用方法：尝试无表头读取
            try:
                df_raw = pd.read_excel(file_path, header=None)
                st.write(f"备用方法读取形状: {df_raw.shape}")
                st.write("前5行数据:")
                st.dataframe(df_raw.head())
                
                if df_raw.shape[1] >= 2:
                    t = df_raw.iloc[:, 0].values.astype(float)
                    phase = df_raw.iloc[:, 1].values.astype(float)
                    phase = np.real(phase)
                    return t, phase
            except Exception as e2:
                st.error(f"备用方法也失败: {str(e2)}")
            
            return None, None
    
    def preprocess_phase(self, phase, lowpass_cutoff=None, highpass_cutoff=None):
        """预处理相位信号"""
        if phase is None or len(phase) == 0:
            return phase
        
        # 确保相位信号是实数
        processed_phase = np.real(phase.copy())
        
        # 去除直流偏移
        if len(processed_phase) > 0:
            processed_phase = processed_phase - np.mean(processed_phase)
        
        # 低通滤波（如果需要）
        if lowpass_cutoff is not None and self.fs is not None and lowpass_cutoff < self.fs/2:
            nyquist = self.fs / 2
            low = lowpass_cutoff / nyquist
            if 0 < low < 1.0 and len(processed_phase) > 10:
                b, a = butter(4, low, btype='low')
                processed_phase = filtfilt(b, a, processed_phase)
        
        # 高通滤波（如果需要，去除环境相位）
        if highpass_cutoff is not None and self.fs is not None and highpass_cutoff < self.fs/2:
            nyquist = self.fs / 2
            high = highpass_cutoff / nyquist
            if 0 < high < 1.0 and len(processed_phase) > 10:
                b, a = butter(4, high, btype='high')
                processed_phase = filtfilt(b, a, processed_phase)
        
        return processed_phase
    
    def compute_cross_correlation(self, phi1, phi2, max_lag=None):
        """
        计算两个信号的互相关函数
        
        参数:
            phi1, phi2: 两个相位信号
            max_lag: 最大时延（采样点数），如果为None则使用信号长度
            
        返回:
            lags: 时延数组（采样点数）
            correlation: 互相关函数值
        """
        if phi1 is None or phi2 is None or len(phi1) == 0 or len(phi2) == 0:
            return np.array([]), np.array([])
        
        n = len(phi1)
        if max_lag is None:
            max_lag = n // 2
        
        # 使用scipy的互相关函数
        correlation = correlate(phi1, phi2, mode='full', method='auto')
        
        # 确保互相关结果是实数
        correlation = np.real(correlation)
        
        # 计算时延
        lags = np.arange(-(n-1), n)
        
        # 只保留max_lag范围内的结果
        valid_indices = np.where(np.abs(lags) <= max_lag)[0]
        lags = lags[valid_indices]
        correlation = correlation[valid_indices]
        
        return lags, correlation
    
    def estimate_time_delay(self, phi1, phi2, method='cross_correlation'):
        """
        估计两个信号之间的时延
        
        参数:
            phi1, phi2: 两个信号
            method: 估计方法，'cross_correlation' 或 'peak_detection'
            
        返回:
            delay_samples: 时延（采样点数）
            delay_seconds: 时延（秒）
        """
        if phi1 is None or phi2 is None or len(phi1) == 0 or len(phi2) == 0:
            return 0, 0, np.array([]), np.array([])
        
        if method == 'cross_correlation':
            # 计算互相关
            lags, correlation = self.compute_cross_correlation(phi1, phi2)
            
            if len(correlation) == 0:
                return 0, 0, np.array([]), np.array([])
            
            # 找到互相关函数的峰值
            peak_index = np.argmax(np.abs(correlation))
            delay_samples = lags[peak_index]
            
            # 如果需要更精确的峰值定位，可以使用抛物线插值
            if 0 < peak_index < len(correlation) - 1:
                # 抛物线插值
                a = correlation[peak_index - 1]
                b = correlation[peak_index]
                c = correlation[peak_index + 1]
                
                # 确保a, b, c是实数
                a = np.real(a)
                b = np.real(b)
                c = np.real(c)
                
                # 避免除零错误
                denominator = a - 2*b + c
                if denominator != 0:
                    delta = 0.5 * (a - c) / denominator
                    # 确保delta是实数
                    delta = np.real(delta)
                    delay_samples += delta
            
            # 确保delay_samples是实数
            delay_samples = np.real(delay_samples)
            
        elif method == 'peak_detection':
            # 简单的峰值检测方法
            # 这里可以添加更复杂的峰值检测算法
            pass
        
        # 将采样点数转换为时间
        delay_seconds = delay_samples / self.fs if self.fs is not None else delay_samples
        
        return delay_samples, delay_seconds, lags, correlation
    
    def calculate_vibration_location(self, delta_tau, c=None, n=None):
        """
        根据时延计算振动位置
        
        参数:
            delta_tau: 时延（秒）
            c: 光速（m/s），如果为None则使用实例的c
            n: 光纤折射率，如果为None则使用实例的n
            
        返回:
            Lx: 振动位置（米）
        """
        if c is None:
            c = self.c
        if n is None:
            n = self.n
        
        # 根据公式(2-12): Lx = c * Δτ / (2n)
        Lx = c * delta_tau / (2 * n)
        
        return Lx

def main():
    """Streamlit主应用程序"""
    st.set_page_config(
        page_title="振动定位系统 (从解调相位开始)",
        page_icon="📍",
        layout="wide"
    )
    
    st.title("📍 振动定位系统 - 从解调相位开始")
    st.markdown("基于解调好的MI和SI相位信号进行互相关分析和振动定位")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # 采样率设置 - 手动输入，默认4,000,000 Hz
    st.sidebar.subheader("采样率设置")
    fs = st.sidebar.number_input("采样率 (Hz)", 
                               min_value=1000.0, 
                               max_value=10000000.0, 
                               value=4000000.0,
                               step=1000.0,
                               help="手动输入采样率，默认值为4,000,000 Hz (4 MHz)")
    
    # 文件上传
    st.sidebar.subheader("相位文件上传")
    mi_phase_file = st.sidebar.file_uploader("上传MI相位文件", type=['xlsx'], key='mi_phase')
    si_phase_file = st.sidebar.file_uploader("上传SI相位文件", type=['xlsx'], key='si_phase')
    
    if mi_phase_file is not None and si_phase_file is not None:
        # 物理参数设置
        st.sidebar.subheader("物理参数")
        c = st.sidebar.number_input("光速 c (m/s)", 
                                  min_value=2.0e8, 
                                  max_value=3.0e8, 
                                  value=3.0e8,
                                  step=1.0e7)
        
        n = st.sidebar.number_input("光纤折射率 n", 
                                  min_value=1.0, 
                                  max_value=2.0, 
                                  value=1.5,
                                  step=0.01)
        
        # 信号处理参数
        st.sidebar.subheader("信号处理参数")
        lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                               min_value=1.0, 
                                               max_value=fs/2 if fs else 10000.0, 
                                               value=1000.0,
                                               step=10.0)
        
        highpass_cutoff = st.sidebar.number_input("高通滤波器截止频率 (Hz, 去除环境相位)", 
                                                min_value=0.1, 
                                                max_value=100.0, 
                                                value=1.0,
                                                step=0.1)
        
        # 显示时间跨度设置
        st.sidebar.subheader("显示设置")
        time_span = st.sidebar.number_input("显示时间跨度 (秒)", 
                                          min_value=0.01, 
                                          max_value=1.0, 
                                          value=0.1,
                                          step=0.01)
        
        # 调试选项
        st.sidebar.subheader("调试选项")
        show_raw_data = st.sidebar.checkbox("显示原始数据详情", value=True)
        
        # 创建定位器实例
        locator = VibrationLocatorFromPhase(fs=fs, c=c, n=n)
        
        # 加载数据
        with st.spinner("正在加载相位数据..."):
            t_mi, phase_mi_raw = locator.load_phase_data(io.BytesIO(mi_phase_file.read()), "MI相位")
            t_si, phase_si_raw = locator.load_phase_data(io.BytesIO(si_phase_file.read()), "SI相位")
        
        if t_mi is not None and phase_mi_raw is not None and \
           t_si is not None and phase_si_raw is not None:
            
            # 检查时间序列是否一致
            if len(t_mi) != len(t_si):
                st.warning(f"MI和SI信号长度不一致: MI={len(t_mi)}, SI={len(t_si)}")
                # 取较短的长度
                min_len = min(len(t_mi), len(t_si))
                t_mi = t_mi[:min_len]
                phase_mi_raw = phase_mi_raw[:min_len]
                t_si = t_si[:min_len]
                phase_si_raw = phase_si_raw[:min_len]
                st.info(f"已统一长度为{min_len}点")
            
            # 显示数据对比
            st.subheader("📊 数据对比")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MI数据点数", len(t_mi))
                st.metric("MI相位范围", f"{np.ptp(phase_mi_raw):.4f} rad")
            with col2:
                st.metric("SI数据点数", len(t_si))
                st.metric("SI相位范围", f"{np.ptp(phase_si_raw):.4f} rad")
            
            # 预处理相位信号
            with st.spinner("正在预处理相位信号..."):
                phase_mi = locator.preprocess_phase(phase_mi_raw, lowpass_cutoff, highpass_cutoff)
                phase_si = locator.preprocess_phase(phase_si_raw, lowpass_cutoff, highpass_cutoff)
                
                # 根据公式(2-10)计算φ1和φ2
                # φ1(t) = Δφ_MI(t) + Δφ_SI(t) = 2φ(t-2τ_x)
                # φ2(t) = Δφ_MI(t) - Δφ_SI(t) = 2φ(t)
                phi1 = phase_mi + phase_si
                phi2 = phase_mi - phase_si
            
            # 计算要显示的数据点数
            n_display = int(time_span * fs) if fs > 0 else 1000
            n_display = min(n_display, len(t_mi))
            
            # 绘制原始相位信号
            st.subheader(f"📈 相位信号 (显示 {time_span} 秒, 共 {n_display} 点)")
            
            if n_display > 0:
                fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # MI相位
                axes[0, 0].plot(t_mi[:n_display], phase_mi_raw[:n_display], 'b-', alpha=0.7, linewidth=0.5, label='原始')
                axes[0, 0].plot(t_mi[:n_display], phase_mi[:n_display], 'r-', alpha=0.7, linewidth=0.5, label='预处理后')
                axes[0, 0].set_xlabel('时间 (s)')
                axes[0, 0].set_ylabel('相位 (rad)')
                axes[0, 0].set_title('MI相位信号')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                if len(t_mi) > 0:
                    axes[0, 0].set_xlim([t_mi[0], t_mi[0] + time_span])
                
                # SI相位
                axes[0, 1].plot(t_si[:n_display], phase_si_raw[:n_display], 'b-', alpha=0.7, linewidth=0.5, label='原始')
                axes[0, 1].plot(t_si[:n_display], phase_si[:n_display], 'r-', alpha=0.7, linewidth=0.5, label='预处理后')
                axes[0, 1].set_xlabel('时间 (s)')
                axes[0, 1].set_ylabel('相位 (rad)')
                axes[0, 1].set_title('SI相位信号')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                if len(t_si) > 0:
                    axes[0, 1].set_xlim([t_si[0], t_si[0] + time_span])
                
                # φ1信号
                axes[1, 0].plot(t_mi[:n_display], phi1[:n_display], 'g-', alpha=0.7, linewidth=0.5)
                axes[1, 0].set_xlabel('时间 (s)')
                axes[1, 0].set_ylabel('幅度')
                axes[1, 0].set_title('φ1(t) = Δφ_MI(t) + Δφ_SI(t)')
                axes[1, 0].grid(True, alpha=0.3)
                if len(t_mi) > 0:
                    axes[1, 0].set_xlim([t_mi[0], t_mi[0] + time_span])
                
                # φ2信号
                axes[1, 1].plot(t_mi[:n_display], phi2[:n_display], 'm-', alpha=0.7, linewidth=0.5)
                axes[1, 1].set_xlabel('时间 (s)')
                axes[1, 1].set_ylabel('幅度')
                axes[1, 1].set_title('φ2(t) = Δφ_MI(t) - Δφ_SI(t)')
                axes[1, 1].grid(True, alpha=0.3)
                if len(t_mi) > 0:
                    axes[1, 1].set_xlim([t_mi[0], t_mi[0] + time_span])
                
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.error("没有足够的数据点进行显示！")
            
            # 计算互相关函数
            st.subheader("🔄 互相关分析")
            
            with st.spinner("正在计算互相关函数..."):
                delay_samples, delay_seconds, lags, correlation = locator.estimate_time_delay(phi1, phi2)
            
            if len(correlation) > 0:
                # 绘制互相关函数
                fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # 互相关函数全视图
                axes[0].plot(lags/fs if fs else lags, correlation, 'k-', linewidth=1)
                axes[0].axvline(x=delay_seconds, color='r', linestyle='--', linewidth=1, 
                              label=f'时延峰值: {delay_seconds*1e6:.2f} μs')
                axes[0].set_xlabel('时延 (s)')
                axes[0].set_ylabel('互相关值')
                axes[0].set_title('φ1(t) 和 φ2(t) 的互相关函数 (全视图)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 互相关函数细节视图（放大峰值区域）
                if len(lags) > 0 and len(correlation) > 0 and fs is not None:
                    window_seconds = 0.0001  # 100微秒窗口
                    window_samples = int(window_seconds * fs)
                    peak_lag_samples = int(np.real(delay_samples))
                    zoom_min = max(lags[0], peak_lag_samples - window_samples)
                    zoom_max = min(lags[-1], peak_lag_samples + window_samples)
                    
                    # 找到对应的索引
                    idx = (lags >= zoom_min) & (lags <= zoom_max)
                    if np.any(idx):
                        lags_zoom = lags[idx] / fs
                        correlation_zoom = correlation[idx]
                        
                        axes[1].plot(lags_zoom, correlation_zoom, 'k-', linewidth=2)
                        axes[1].axvline(x=delay_seconds, color='r', linestyle='--', linewidth=2, 
                                      label=f'时延峰值: {delay_seconds*1e6:.2f} μs')
                        axes[1].set_xlabel('时延 (s)')
                        axes[1].set_ylabel('互相关值')
                        axes[1].set_title('互相关函数峰值区域 (放大)')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                
                # 计算振动位置
                st.subheader("📍 振动位置计算")
                
                # 计算振动位置
                Lx = locator.calculate_vibration_location(delay_seconds, c, n)
                
                # 显示结果
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("时延 Δτ", f"{delay_seconds*1e6:.4f} μs")
                with col2:
                    st.metric("时延 Δτ", f"{delay_seconds*1e3:.4f} ms")
                with col3:
                    st.metric("时延采样点数", f"{int(np.real(delay_samples))}")
                with col4:
                    st.metric("振动位置 Lx", f"{Lx/1000:.2f} km")
            else:
                st.error("无法计算互相关函数，数据可能有问题！")
    
    else:
        st.info("👈 请从左侧上传MI和SI相位文件开始分析")

if __name__ == "__main__":
    main()