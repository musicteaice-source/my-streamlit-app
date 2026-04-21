import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import streamlit as st
import io
from datetime import datetime

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class FiberInterferometerLocator:
    """光纤干涉仪振动定位系统"""
    
    def __init__(self, fs=None, c=3e8, n=1.468):
        """
        初始化参数
        
        参数:
            fs: 采样率 (Hz)
            c: 光速 (m/s)，默认3e8
            n: 光纤折射率，默认1.468
        """
        self.fs = fs
        self.c = c
        self.n = n
        
    def load_two_channel_data(self, file_path):
        """从Excel文件加载两通道数据（时间+两路信号）"""
        try:
            df = pd.read_excel(file_path, header=None)
            
            if df.shape[0] < 2:
                st.error("数据文件行数不足")
                return None, None, None
            
            if df.shape[1] < 4:
                st.error(f"数据文件需要至少4列，但只有{df.shape[1]}列")
                return None, None, None
            
            # 从第二行开始提取数据
            t = df.iloc[1:, 0].values.astype(float)
            sig1 = df.iloc[1:, 1].values.astype(float)
            sig2 = df.iloc[1:, 3].values.astype(float)
            
            # 检查时间列是否一致
            t_col2 = df.iloc[1:, 2].values.astype(float)
            if not np.allclose(t, t_col2, rtol=1e-5):
                st.warning("警告：第一列和第三列时间数据不完全一致，使用第一列时间数据")
            
            # 自动计算采样率
            if self.fs is None and len(t) > 1:
                self.fs = 1.0 / np.mean(np.diff(t))
            
            return t, sig1, sig2
            
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")
            return None, None, None
    
    def calculate_phase_dcm(self, sig1, sig2):
        """原有的微分交叉相乘(DCM)算法"""
        n = len(sig1)
        
        # 去直流
        sig1_ac = sig1 - np.mean(sig1)
        sig2_ac = sig2 - np.mean(sig2)
        
        # 微分
        sig1_diff = np.diff(sig1_ac, prepend=sig1_ac[0])
        sig2_diff = np.diff(sig2_ac, prepend=sig2_ac[0])
        
        # DCM算法
        numerator = sig1_ac * sig2_diff - sig2_ac * sig1_diff
        denominator = sig1_ac**2 + sig2_ac**2 + sig1_ac * sig2_ac
        
        epsilon = 1e-10
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        
        phase_derivative = 2.0 / np.sqrt(3.0) * numerator / denominator
        # 关键修复：减去均值去除直流偏移
        phase_derivative = phase_derivative - np.mean(phase_derivative)
        
        # 积分
        phase = np.cumsum(phase_derivative) / self.fs
        
        return phase
    
    def calculate_phase_document_method(self, sig1, sig2, psi=2*np.pi/3):
        """
        根据文档第19-20页描述的算法进行相位解调。
        算法基于公式(42)到(48)。
        """
        n = len(sig1)
        
        # 步骤1: 计算信号的最大值和最小值
        P1_max = np.max(sig1)
        P1_min = np.min(sig1)
        P2_max = np.max(sig2)
        P2_min = np.min(sig2)
        
        # 防止除零
        if (P1_max - P1_min) == 0 or (P2_max - P2_min) == 0:
            st.warning("信号动态范围不足，可能未达到最大最小值。尝试使用DCM算法。")
            return self.calculate_phase_dcm(sig1, sig2)
        
        # 步骤2: 信号归一化 (式44-46)
        P1_prime = 2 * ((sig1 - P1_min) / (P1_max - P1_min)) - 1
        P2_prime = 2 * ((sig2 - P2_min) / (P2_max - P2_min)) - 1
        
        # 步骤3: 计算组合信号 (式47)
        P_plus = P1_prime + P2_prime      # 对应 -cos(Δφ)
        P_minus = P1_prime - P2_prime     # 对应 -√3 * sin(Δφ)
        
        # 步骤4: 计算相位 (式48)
        epsilon = 1e-10
        P_plus_safe = np.where(np.abs(P_plus) < epsilon, epsilon * np.sign(P_plus), P_plus)
        phase_wrapped = np.arctan2(P_minus, np.sqrt(3) * P_plus_safe)
        
        # 步骤5: 正确的相位解卷绕（修复不连续问题）
        # 方法1：使用numpy的unwrap函数（推荐）
        phase_unwrapped = np.unwrap(phase_wrapped, discont=np.pi)  # 默认阈值就是π
        
        # 或者方法2：手动解卷绕
        # phase_unwrapped = self._manual_unwrap(phase_wrapped)  
              
        return phase_unwrapped

    def _manual_unwrap(self, phase):
        """手动相位解卷绕"""
        unwrapped = phase.copy()
        offset = 0.0
        
        for i in range(1, len(phase)):
            diff = phase[i] - phase[i-1]
            
            # 检测2π的跳变（实际上检测π的跳变）
            if diff > np.pi:
                offset -= 2 * np.pi
            elif diff < -np.pi:
                offset += 2 * np.pi
            
            unwrapped[i] = phase[i] + offset
    
        return unwrapped

    
    def demodulate_phase(self, t, sig1, sig2, method='document', lowpass_cutoff=None):
        """
        完整的相位解调流程
        
        参数:
            method: 'document' 或 'dcm'
        """
        # 选择解调方法
        if method == 'document':
            phase = self.calculate_phase_document_method(sig1, sig2)
        else:  # 'dcm'
            phase = self.calculate_phase_dcm(sig1, sig2)
        
        # 相位解卷绕 (使用numpy的unwrap作为备用)
        try:
            phase_unwrapped = np.unwrap(phase)
        except:
            phase_unwrapped = phase.copy()
        
        # 去除直流偏移
        phase_unwrapped = phase_unwrapped - np.mean(phase_unwrapped)
        
        # 低通滤波 (如果需要)
        phase_filtered = phase_unwrapped.copy()
        if lowpass_cutoff is not None and self.fs is not None and lowpass_cutoff < self.fs/2:
            nyquist = self.fs / 2
            low = lowpass_cutoff / nyquist
            if 0 < low < 1.0:
                b, a = butter(4, low, btype='low')
                phase_filtered = filtfilt(b, a, phase_unwrapped)
        
        return phase_filtered
    
    def compute_cross_correlation(self, phi1, phi2, max_lag=None):
        """
        计算两个信号的互相关函数
        
        参数:
            phi1, phi2: 两个信号
            max_lag: 最大时延（采样点数），如果为None则使用信号长度
            
        返回:
            lags: 时延数组（采样点数）
            correlation: 互相关函数值
        """
        n = len(phi1)
        if max_lag is None:
            max_lag = n // 2
        
        # 使用scipy的互相关函数
        correlation = correlate(phi1, phi2, mode='full', method='auto')
        
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
        if method == 'cross_correlation':
            # 计算互相关
            lags, correlation = self.compute_cross_correlation(phi1, phi2)
            
            # 找到互相关函数的峰值
            peak_index = np.argmax(np.abs(correlation))
            delay_samples = lags[peak_index]
            
            # 如果需要更精确的峰值定位，可以使用抛物线插值
            if 0 < peak_index < len(correlation) - 1:
                # 抛物线插值
                a = correlation[peak_index - 1]
                b = correlation[peak_index]
                c = correlation[peak_index + 1]
                delta = 0.5 * (a - c) / (a - 2*b + c)
                delay_samples += delta
            
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
    
    def filter_signal(self, signal, cutoff_freq, filter_type='low'):
        """滤波器函数"""
        if self.fs is None or cutoff_freq is None:
            return signal
        
        nyquist = self.fs / 2
        if filter_type == 'low':
            btype = 'low'
        elif filter_type == 'high':
            btype = 'high'
        elif filter_type == 'band':
            btype = 'band'
        else:
            return signal
        
        # 设计滤波器
        if btype == 'band':
            if len(cutoff_freq) != 2:
                return signal
            low = cutoff_freq[0] / nyquist
            high = cutoff_freq[1] / nyquist
            if 0 < low < 1.0 and 0 < high < 1.0 and low < high:
                b, a = butter(4, [low, high], btype='band')
            else:
                return signal
        else:
            cutoff = cutoff_freq / nyquist
            if 0 < cutoff < 1.0:
                b, a = butter(4, cutoff, btype=btype)
            else:
                return signal
        
        # 应用滤波器
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal

def main():
    """Streamlit主应用程序"""
    st.set_page_config(
        page_title="光纤MI-SI系统振动定位系统",
        page_icon="📍",
        layout="wide"
    )
    
    st.title("📍 光纤MI-SI系统振动定位系统")
    st.markdown("基于互相关函数的时延估计定位算法")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # 文件上传
    st.sidebar.subheader("数据文件上传")
    mi_file = st.sidebar.file_uploader("上传MI信号文件", type=['xlsx'], key='mi')
    si_file = st.sidebar.file_uploader("上传SI信号文件", type=['xlsx'], key='si')
    
    if mi_file is not None and si_file is not None:
        # 物理参数设置
        st.sidebar.subheader("物理参数")
        c = st.sidebar.number_input("光速 c (m/s)", 
                                  min_value=2.0e8, 
                                  max_value=3.0e8, 
                                  value=3.0e8,
                                  step=1.0e7)
        
        n = st.sidebar.number_input("光纤折射率 n", 
                                  min_value=1.00, 
                                  max_value=2.00, 
                                  value=1.468,
                                  step=0.001)
        
        # 信号处理参数
        st.sidebar.subheader("信号处理参数")
        lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                               min_value=1.0, 
                                               max_value=10000.0, 
                                               value=1000.0,
                                               step=10.0)
        
        # 创建定位器实例
        locator = FiberInterferometerLocator(c=c, n=n)
        
        # 加载数据
        with st.spinner("正在加载数据..."):
            t_mi, mi_sig1, mi_sig2 = locator.load_two_channel_data(io.BytesIO(mi_file.read()))
            t_si, si_sig1, si_sig2 = locator.load_two_channel_data(io.BytesIO(si_file.read()))
        
        if t_mi is not None and mi_sig1 is not None and mi_sig2 is not None and \
           t_si is not None and si_sig1 is not None and si_sig2 is not None:
            
            # 检查时间序列是否一致
            if len(t_mi) != len(t_si):
                st.warning(f"MI和SI信号长度不一致: MI={len(t_mi)}, SI={len(t_si)}")
                # 取较短的长度
                min_len = min(len(t_mi), len(t_si))
                t_mi = t_mi[:min_len]
                mi_sig1 = mi_sig1[:min_len]
                mi_sig2 = mi_sig2[:min_len]
                t_si = t_si[:min_len]
                si_sig1 = si_sig1[:min_len]
                si_sig2 = si_sig2[:min_len]
            
            # 自动设置采样率
            if locator.fs is None and len(t_mi) > 1:
                locator.fs = 1.0 / np.mean(np.diff(t_mi))
                st.sidebar.info(f"自动计算采样率: {locator.fs:.2f} Hz")
            
            # 显示数据信息
            st.subheader("📊 数据信息")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MI信号点数", len(t_mi))
            with col2:
                st.metric("SI信号点数", len(t_si))
            with col3:
                st.metric("采样率", f"{locator.fs:.2f} Hz" if locator.fs else "未知")
            with col4:
                st.metric("时长", f"{t_mi[-1] - t_mi[0]:.4f} s")
            
            # 绘制原始信号
            st.subheader("📈 原始信号")
            
            # 选择显示的数据点数
            display_points = st.slider("显示数据点数", 100, len(t_mi), len(t_mi), 100)
            
            fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
            
            # MI信号
            axes[0, 0].plot(t_mi[:display_points], mi_sig1[:display_points], 'b-', alpha=0.7, linewidth=1)
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('幅度')
            axes[0, 0].set_title('MI信号 - 通道1')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(t_mi[:display_points], mi_sig2[:display_points], 'r-', alpha=0.7, linewidth=1)
            axes[0, 1].set_xlabel('时间 (s)')
            axes[0, 1].set_ylabel('幅度')
            axes[0, 1].set_title('MI信号 - 通道2')
            axes[0, 1].grid(True, alpha=0.3)
            
            # SI信号
            axes[1, 0].plot(t_si[:display_points], si_sig1[:display_points], 'g-', alpha=0.7, linewidth=1)
            axes[1, 0].set_xlabel('时间 (s)')
            axes[1, 0].set_ylabel('幅度')
            axes[1, 0].set_title('SI信号 - 通道1')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(t_si[:display_points], si_sig2[:display_points], 'm-', alpha=0.7, linewidth=1)
            axes[1, 1].set_xlabel('时间 (s)')
            axes[1, 1].set_ylabel('幅度')
            axes[1, 1].set_title('SI信号 - 通道2')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
            # 相位解调
            st.subheader("🔄 相位解调结果")
            
            with st.spinner("正在进行相位解调..."):
                # 解调MI相位
                phase_mi = locator.demodulate_phase(mi_sig1, mi_sig2, lowpass_cutoff)
                
                # 解调SI相位
                phase_si = locator.demodulate_phase(si_sig1, si_sig2, lowpass_cutoff)
                
                # 根据公式(2-10)计算φ1和φ2
                # φ1(t) = Δφ_MI(t) + Δφ_SI(t) = 2φ(t-2τ_x)
                # φ2(t) = Δφ_MI(t) - Δφ_SI(t) = 2φ(t)
                phi1 = phase_mi + phase_si
                phi2 = phase_mi - phase_si
            
            # 绘制相位解调结果
            fig2, axes = plt.subplots(3, 2, figsize=(14, 12))
            
            # MI相位
            axes[0, 0].plot(t_mi[:display_points], phase_mi[:display_points], 'b-', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('相位 (rad)')
            axes[0, 0].set_title('MI系统相位 Δφ_MI(t)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # SI相位
            axes[0, 1].plot(t_si[:display_points], phase_si[:display_points], 'r-', alpha=0.7, linewidth=0.5)
            axes[0, 1].set_xlabel('时间 (s)')
            axes[0, 1].set_ylabel('相位 (rad)')
            axes[0, 1].set_title('SI系统相位 Δφ_SI(t)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # φ1信号
            axes[1, 0].plot(t_mi[:display_points], phi1[:display_points], 'g-', alpha=0.7, linewidth=0.5)
            axes[1, 0].set_xlabel('时间 (s)')
            axes[1, 0].set_ylabel('幅度')
            axes[1, 0].set_title('φ1(t) = Δφ_MI(t) + Δφ_SI(t)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # φ2信号
            axes[1, 1].plot(t_mi[:display_points], phi2[:display_points], 'm-', alpha=0.7, linewidth=0.5)
            axes[1, 1].set_xlabel('时间 (s)')
            axes[1, 1].set_ylabel('幅度')
            axes[1, 1].set_title('φ2(t) = Δφ_MI(t) - Δφ_SI(t)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # φ1和φ2叠加显示
            axes[2, 0].plot(t_mi[:display_points], phi1[:display_points], 'g-', alpha=0.7, linewidth=0.5, label='φ1(t)')
            axes[2, 0].plot(t_mi[:display_points], phi2[:display_points], 'm-', alpha=0.7, linewidth=0.5, label='φ2(t)')
            axes[2, 0].set_xlabel('时间 (s)')
            axes[2, 0].set_ylabel('幅度')
            axes[2, 0].set_title('φ1(t) 和 φ2(t) 对比')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 计算互相关函数
            with st.spinner("正在计算互相关函数..."):
                delay_samples, delay_seconds, lags, correlation = locator.estimate_time_delay(phi1, phi2)
            
            # 互相关函数
            axes[2, 1].plot(lags/locator.fs if locator.fs else lags, correlation, 'k-', linewidth=1)
            axes[2, 1].axvline(x=delay_seconds, color='r', linestyle='--', linewidth=1, 
                             label=f'时延峰值: {delay_seconds*1e6:.2f} μs')
            axes[2, 1].set_xlabel('时延 (s)')
            axes[2, 1].set_ylabel('互相关值')
            axes[2, 1].set_title('φ1(t) 和 φ2(t) 的互相关函数')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
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
                st.metric("时延采样点数", f"{int(delay_samples)}")
            with col4:
                st.metric("振动位置 Lx", f"{Lx/1000:.2f} km")
            
            # 显示详细计算过程
            with st.expander("查看详细计算过程"):
                st.markdown(f"""
                ### 计算公式
                根据讲义中的公式(2-12):
                
                $$
                L_x = \\frac{{c \\cdot \\Delta\\tau}}{{2n}}
                $$
                
                ### 参数值
                - 光速 c = {c:.1e} m/s
                - 光纤折射率 n = {n:.3f}
                - 时延 Δτ = {delay_seconds:.6f} s = {delay_seconds*1e6:.2f} μs
                
                ### 计算结果
                $$
                L_x = \\frac{{{c:.1e} \\times {delay_seconds:.6f}}}{{2 \\times {n:.3f}}}
                = {Lx:.2f} \\text{{ m}} = {Lx/1000:.2f} \\text{{ km}}
                $$
                
                ### 互相关函数信息
                - 最大互相关值: {np.max(correlation):.4f}
                - 时延对应采样点: {int(delay_samples)}
                - 互相关函数长度: {len(correlation)} 点
                """)
            
            # 保存结果选项
            st.subheader("💾 保存结果")
            
            # 创建结果数据框
            result_data = pd.DataFrame({
                '时间_s': t_mi,
                'MI_通道1': mi_sig1,
                'MI_通道2': mi_sig2,
                'SI_通道1': si_sig1,
                'SI_通道2': si_sig2,
                '相位_MI': phase_mi,
                '相位_SI': phase_si,
                'phi1': phi1,
                'phi2': phi2
            })
            
            # 将数据转换为Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_data.to_excel(writer, index=False, sheet_name='解调结果')
                
                # 添加摘要信息
                summary_df = pd.DataFrame({
                    '参数': ['采样率', '光速', '折射率', '时延(秒)', '时延(微秒)', '振动位置(米)', '振动位置(公里)'],
                    '值': [f"{locator.fs:.2f} Hz" if locator.fs else "未知", 
                          f"{c:.1e} m/s", 
                          f"{n:.3f}", 
                          f"{delay_seconds:.6f}", 
                          f"{delay_seconds*1e6:.2f}", 
                          f"{Lx:.2f}", 
                          f"{Lx/1000:.2f}"]
                })
                summary_df.to_excel(writer, index=False, sheet_name='定位结果')
            
            output.seek(0)
            
            # 下载按钮
            st.download_button(
                label="下载解调与定位结果 (Excel)",
                data=output,
                file_name=f"vibration_location_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # 显示算法原理
            st.subheader("📖 算法原理")
            
            st.markdown("""
            ### 基于互相关函数的时延估计定位算法原理
            
            根据讲义中的推导，该定位方法基于以下原理：
            
            1. **MI-SI干涉系统相位差**:
               - MI系统相位差: $\\Delta\\varphi_{MI}(t) = \\phi(t-2\\tau_x) + \\phi(t)$
               - SI系统相位差: $\\Delta\\varphi_{SI}(t) = \\phi(t-2\\tau_x) - \\phi(t)$
            
            2. **构造延迟信号**:
               - $\\varphi_1(t) = \\Delta\\varphi_{MI}(t) + \\Delta\\varphi_{SI}(t) = 2\\phi(t-2\\tau_x)$
               - $\\varphi_2(t) = \\Delta\\varphi_{MI}(t) - \\Delta\\varphi_{SI}(t) = 2\\phi(t)$
            
            3. **时延关系**:
               - $\\varphi_1(t)$ 和 $\\varphi_2(t)$ 之间存在固定时延: $\\Delta\\tau = 2\\tau_x = \\frac{2nL_x}{c}$
            
            4. **互相关计算时延**:
               - 计算 $\\varphi_1(t)$ 和 $\\varphi_2(t)$ 的互相关函数
               - 互相关函数峰值对应的自变量即为时延 $\\Delta\\tau$
            
            5. **计算振动位置**:
               - $L_x = \\frac{c\\Delta\\tau}{2n}$
            
            **其中**:
            - $\\phi(t)$: 振动施加的相位调制
            - $\\tau_x$: 由长度 $L_x$ 引起的时间延迟
            - $n$: 光纤折射率
            - $c$: 光速
            """)
    
    else:
        st.info("👈 请从左侧上传MI和SI数据文件开始分析")
        
        st.markdown("""
        ## 光纤MI-SI系统振动定位系统使用说明
        
        本程序基于《光纤分布式振动定位》讲义中的时延估计定位算法，实现振动位置的精确计算。
        
        ### 使用方法
        
        1. **准备数据文件**：
           - 准备两个Excel文件，分别包含MI系统和SI系统的两路干涉信号
           - 每个文件应包含4列：时间1，信号1，时间2，信号2
           - 第一行是列名，从第二行开始是数据
        
        2. **上传文件**：
           - 在左侧分别上传MI信号文件和SI信号文件
        
        3. **配置参数**：
           - 设置光速和光纤折射率（默认值通常适用）
           - 设置低通滤波器截止频率
        
        4. **查看结果**：
           - 原始信号波形
           - 相位解调结果
           - 互相关函数
           - 计算出的振动位置
        
        5. **下载结果**：
           - 可以下载包含所有解调结果和定位结果的Excel文件
        
        ### 算法特点
        
        1. **高精度定位**：基于互相关函数的时延估计，定位精度高
        2. **抗干扰能力强**：通过3x3耦合器相位解调，消除光源波动影响
        3. **实时处理**：算法计算效率高，适合实时振动监测
        4. **可视化分析**：提供完整的信号处理流程可视化
        
        ### 适用场景
        
        - 光纤分布式振动传感系统
        - 管道泄漏监测
        - 周界安防系统
        - 地震监测
        - 结构健康监测
        """)

if __name__ == "__main__":
    main()