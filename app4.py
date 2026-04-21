import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import streamlit as st
import io
from datetime import datetime

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class Interferometer3x3Demodulator:
    """3x3耦合器干涉信号相位解调器（复用自app2.py，仅作小幅修改）"""
    
    def __init__(self, fs=None):
        self.fs = fs
        
    def load_data(self, file_path):
        """从Excel文件加载数据，跳过第一行（列名）"""
        try:
            df = pd.read_excel(file_path, header=None)
            
            if df.shape[0] < 2:
                st.error("数据文件行数不足，需要至少2行数据（1行列名+1行数据）")
                return None, None, None
            
            if df.shape[1] < 4:
                st.error(f"数据文件需要至少4列，但只有{df.shape[1]}列")
                return None, None, None
            
            # 从第二行开始提取数据
            t = df.iloc[1:, 0].values.astype(float)
            sig1 = df.iloc[1:, 1].values.astype(float)
            sig2 = df.iloc[1:, 3].values.astype(float)
            
            # 自动计算采样率
            if self.fs is None and len(t) > 1:
                self.fs = 1.0 / np.mean(np.diff(t))
                st.sidebar.info(f"自动计算采样率: {self.fs:.2f} Hz")
            
            return t, sig1, sig2
            
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")
            return None, None, None
    
    def calculate_phase_document_method(self, sig1, sig2, psi=2*np.pi/3):
        """
        根据文档描述的算法进行相位解调。
        这是app2.py中的核心算法，这里直接复用。
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
        
        # 步骤2: 信号归一化
        P1_prime = 2 * ((sig1 - P1_min) / (P1_max - P1_min)) - 1
        P2_prime = 2 * ((sig2 - P2_min) / (P2_max - P2_min)) - 1
        
        # 步骤3: 计算组合信号
        P_plus = P1_prime + P2_prime      # 对应 -cos(Δφ)
        P_minus = P1_prime - P2_prime     # 对应 -√3 * sin(Δφ)
        
        # 步骤4: 计算相位
        epsilon = 1e-10
        P_plus_safe = np.where(np.abs(P_plus) < epsilon, epsilon * np.sign(P_plus), P_plus)
        phase_wrapped = np.arctan2(P_minus, np.sqrt(3) * P_plus_safe)
        
        # 步骤5: 相位解卷绕
        phase_unwrapped = np.unwrap(phase_wrapped, discont=np.pi)
              
        return phase_unwrapped

    def demodulate_phase(self, t, sig1, sig2, method='document', lowpass_cutoff=None):
        """
        完整的相位解调流程
        """
        # 选择解调方法
        phase = self.calculate_phase_document_method(sig1, sig2)
        
        # 去除直流偏移
        phase = phase - np.mean(phase)
        
        # 低通滤波 (如果需要)
        phase_filtered = phase.copy()
        if lowpass_cutoff is not None and self.fs is not None and lowpass_cutoff < self.fs/2:
            nyquist = self.fs / 2
            low = lowpass_cutoff / nyquist
            if 0 < low < 1.0:
                b, a = butter(4, low, btype='low')
                phase_filtered = filtfilt(b, a, phase)
        
        return phase_filtered

class MI_SI_Localization:
    """基于MI-SI系统的振动定位器"""
    
    def __init__(self, n=1.468, c=3e8):
        """
        初始化定位器
        n: 光纤折射率，默认为1.468
        c: 光速，默认为3e8 m/s
        """
        self.n = n
        self.c = c
        self.demodulator = Interferometer3x3Demodulator()
        
    def load_mi_si_data(self, mi_file_path, si_file_path, fs=None):
        """
        加载MI和SI数据文件
        返回: t, phi_MI, phi_SI
        """
        self.demodulator.fs = fs
        
        # 加载MI数据
        t_mi, sig1_mi, sig2_mi = self.demodulator.load_data(mi_file_path)
        if t_mi is None:
            return None, None, None
            
        # 加载SI数据
        t_si, sig1_si, sig2_si = self.demodulator.load_data(si_file_path)
        if t_si is None:
            return None, None, None
        
        # 检查时间轴是否一致
        if len(t_mi) != len(t_si) or not np.allclose(t_mi[:10], t_si[:10], rtol=1e-3):
            st.warning("MI和SI数据的时间轴不完全一致，但将继续处理")
        
        # 使用MI的时间作为统一时间轴
        t = t_mi
        self.fs = 1.0 / np.mean(np.diff(t)) if fs is None else fs
        
        return t, (sig1_mi, sig2_mi), (sig1_si, sig2_si)
    
    def demodulate_phases(self, t, mi_signals, si_signals, lowpass_cutoff=None):
        """
        对MI和SI信号进行相位解调
        """
        sig1_mi, sig2_mi = mi_signals
        sig1_si, sig2_si = si_signals
        
        # 解调MI相位
        phi_MI = self.demodulator.demodulate_phase(t, sig1_mi, sig2_mi, 
                                                  method='document', 
                                                  lowpass_cutoff=lowpass_cutoff)
        
        # 解调SI相位
        phi_SI = self.demodulator.demodulate_phase(t, sig1_si, sig2_si, 
                                                  method='document', 
                                                  lowpass_cutoff=lowpass_cutoff)
        
        return phi_MI, phi_SI
    
    def calculate_time_delay_signals(self, phi_MI, phi_SI):
        """
        根据文档公式(2-10)计算具有固定时延的两个信号
        φ1(t) = Δφ_MI(t) + Δφ_SI(t) = 2φ(t-2τ_x)
        φ2(t) = Δφ_MI(t) - Δφ_SI(t) = 2φ(t)
        """
        phi1 = phi_MI + phi_SI
        phi2 = phi_MI - phi_SI
        
        return phi1, phi2
    
    def compute_cross_correlation(self, phi1, phi2, max_lag_samples=None):
        """
        计算两个信号的互相关函数
        返回: 时延样本数组, 互相关值数组, 峰值位置索引
        """
        # 如果未指定最大滞后，默认使用信号长度的一半
        if max_lag_samples is None:
            max_lag_samples = len(phi1) // 2
        
        # 归一化信号（减去均值）
        phi1_norm = phi1 - np.mean(phi1)
        phi2_norm = phi2 - np.mean(phi2)
        
        # 计算互相关
        correlation = correlate(phi1_norm, phi2_norm, mode='full')
        
        # 创建时延数组（以样本数为单位）
        lags = np.arange(-len(phi2_norm) + 1, len(phi1_norm))
        
        # 转换为时间延迟
        time_delays = lags / self.fs
        
        # 找到峰值位置
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_delay = time_delays[peak_idx]
        
        return time_delays, correlation, peak_idx, peak_lag, peak_delay
    
    def calculate_vibration_position(self, time_delay):
        """
        根据文档公式(2-12)计算振动位置
        L_x = (c * Δτ) / (2n)
        其中: Δτ = 2τ_x
        """
        L_x = (self.c * time_delay) / (2 * self.n)
        return L_x
    
    def compute_fft(self, signal, window='hann'):
        """计算信号的傅里叶变换"""
        n = len(signal)
        if n < 2:
            return np.array([]), np.array([])
        
        if window == 'hann':
            window_func = np.hanning(n)
        elif window == 'hamming':
            window_func = np.hamming(n)
        elif window == 'blackman':
            window_func = np.blackman(n)
        else:
            window_func = np.ones(n)
        
        signal_windowed = signal * window_func
        fft_result = np.fft.fft(signal_windowed)
        fft_mag = np.abs(fft_result)[:n//2] * 2 / n
        freqs = np.fft.fftfreq(n, 1/self.fs)[:n//2]
        
        return freqs, fft_mag

def main():
    """Streamlit主应用程序"""
    st.set_page_config(
        page_title="光纤MI-SI振动定位系统 (时延估计法)",
        page_icon="📍",
        layout="wide"
    )
    
    st.title("📍 光纤MI-SI振动定位系统 - 时延估计法")
    st.markdown("基于《拓展实验 光纤分布式振动定位》文档中的互相关时延估计定位算法")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # 文件上传
    st.sidebar.subheader("数据文件上传")
    mi_file = st.sidebar.file_uploader("上传MI数据文件 (MI_data.xlsx)", type=['xlsx'])
    si_file = st.sidebar.file_uploader("上传SI数据文件 (SI_data.xlsx)", type=['xlsx'])
    
    if mi_file is not None and si_file is not None:
        # 采样率设置
        fs = st.sidebar.number_input("采样率 (Hz)", 
                                    min_value=1.0, 
                                    max_value=1e9, 
                                    value=1000000.0,
                                    step=1000.0)
        
        # 物理参数
        st.sidebar.subheader("物理参数")
        n = st.sidebar.number_input("光纤折射率 (n)", 
                                   min_value=1.0, 
                                   max_value=3.0, 
                                   value=1.468,
                                   step=0.001)
        
        c = st.sidebar.number_input("光速 (m/s)", 
                                   min_value=2.9e8, 
                                   max_value=3.1e8, 
                                   value=3.0e8,
                                   step=1e6)
        
        # 滤波器设置
        st.sidebar.subheader("滤波器设置")
        lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                               min_value=1.0, 
                                               max_value=fs/2 if fs else 1e6, 
                                               value=1000.0,
                                               step=10.0)
        
        # 互相关参数
        st.sidebar.subheader("互相关参数")
        max_lag_ms = st.sidebar.slider("最大滞后时间 (ms)", 
                                      min_value=0.1, 
                                      max_value=100.0, 
                                      value=10.0,
                                      step=0.1)
        
        # 创建定位器
        localizer = MI_SI_Localization(n=n, c=c)
        
        # 加载数据
        with st.spinner("正在加载数据..."):
            t, mi_signals, si_signals = localizer.load_mi_si_data(
                io.BytesIO(mi_file.read()), 
                io.BytesIO(si_file.read()),
                fs=fs
            )
        
        if t is not None and mi_signals is not None and si_signals is not None:
            # 显示数据信息
            st.subheader("📊 数据信息")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据点数", len(t))
            with col2:
                st.metric("采样率", f"{localizer.fs:.2f} Hz")
            with col3:
                st.metric("时长", f"{t[-1] - t[0]:.4f} s")
            with col4:
                st.metric("光纤折射率", f"{n:.4f}")
            
            # 原始信号显示
            st.subheader("📈 原始信号")
            sig1_mi, sig2_mi = mi_signals
            sig1_si, sig2_si = si_signals
            
            fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
            
            display_points = len(t)
            ax1.plot(t[:display_points], sig1_mi[:display_points], 'b-', alpha=0.7, linewidth=0.5)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('幅度')
            ax1.set_title('MI - 通道1信号')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t[:display_points], sig2_mi[:display_points], 'g-', alpha=0.7, linewidth=0.5)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('幅度')
            ax2.set_title('MI - 通道2信号')
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(t[:display_points], sig1_si[:display_points], 'r-', alpha=0.7, linewidth=0.5)
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('幅度')
            ax3.set_title('SI - 通道1信号')
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(t[:display_points], sig2_si[:display_points], 'm-', alpha=0.7, linewidth=0.5)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('幅度')
            ax4.set_title('SI - 通道2信号')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
            # 相位解调
            st.subheader("🔄 相位解调结果")
            
            with st.spinner("正在解调相位..."):
                phi_MI, phi_SI = localizer.demodulate_phases(
                    t, mi_signals, si_signals, lowpass_cutoff=lowpass_cutoff
                )
            
            fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 8))
            
            display_phase_points = min(2000, len(t))
            ax5.plot(t[:display_phase_points], phi_MI[:display_phase_points], 'b-', alpha=0.7, linewidth=1)
            ax5.set_xlabel('时间 (s)')
            ax5.set_ylabel('相位 (rad)')
            ax5.set_title('MI系统解调相位 Δφ_MI(t)')
            ax5.grid(True, alpha=0.3)
            
            ax6.plot(t[:display_phase_points], phi_SI[:display_phase_points], 'r-', alpha=0.7, linewidth=1)
            ax6.set_xlabel('时间 (s)')
            ax6.set_ylabel('相位 (rad)')
            ax6.set_title('SI系统解调相位 Δφ_SI(t)')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
            # 计算时延信号
            st.subheader("⏱️ 时延信号计算")
            st.markdown("""
            **根据文档公式(2-10):**
            - φ₁(t) = Δφ_MI(t) + Δφ_SI(t) = 2φ(t-2τ_x)
            - φ₂(t) = Δφ_MI(t) - Δφ_SI(t) = 2φ(t)
            """)
            
            phi1, phi2 = localizer.calculate_time_delay_signals(phi_MI, phi_SI)
            
            fig3, (ax7, ax8) = plt.subplots(2, 1, figsize=(12, 8))
            
            display_points_delay = min(2000, len(t))
            ax7.plot(t[:display_points_delay], phi1[:display_points_delay], 'g-', alpha=0.7, linewidth=1)
            ax7.set_xlabel('时间 (s)')
            ax7.set_ylabel('幅度')
            ax7.set_title('φ₁(t) = Δφ_MI(t) + Δφ_SI(t)')
            ax7.grid(True, alpha=0.3)
            
            ax8.plot(t[:display_points_delay], phi2[:display_points_delay], 'm-', alpha=0.7, linewidth=1)
            ax8.set_xlabel('时间 (s)')
            ax8.set_ylabel('幅度')
            ax8.set_title('φ₂(t) = Δφ_MI(t) - Δφ_SI(t)')
            ax8.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            # 互相关计算
            st.subheader("🔗 互相关函数计算")
            
            max_lag_samples = int(max_lag_ms * 0.001 * localizer.fs)
            time_delays, correlation, peak_idx, peak_lag, peak_delay = localizer.compute_cross_correlation(
                phi1, phi2, max_lag_samples
            )
            
            fig4, ax9 = plt.subplots(1, 1, figsize=(12, 6))
            
            # 只显示最大滞后范围内的互相关
            center_idx = len(correlation) // 2
            display_range = slice(center_idx - max_lag_samples, center_idx + max_lag_samples + 1)
            
            ax9.plot(time_delays[display_range] * 1e6, correlation[display_range], 'b-', linewidth=1.5, label='互相关函数')
            ax9.axvline(x=peak_delay * 1e6, color='r', linestyle='--', linewidth=1.5, 
                       label=f'峰值时延: {peak_delay*1e6:.2f} μs')
            ax9.set_xlabel('时延 (μs)')
            ax9.set_ylabel('互相关值')
            ax9.set_title('互相关函数 R(τ)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
            
            # 定位结果
            st.subheader("📍 振动定位结果")
            
            # 计算振动位置
            vibration_distance = localizer.calculate_vibration_position(peak_delay)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("互相关峰值时延", f"{peak_delay*1e6:.4f} μs")
                st.metric("对应的样本滞后", f"{peak_lag} 个样本")
                
            with col2:
                st.metric("计算出的距离 Lₓ", f"{vibration_distance/1000:.2f} km")
                st.metric("计算出的距离 Lₓ", f"{vibration_distance:.2f} m")
                
            with col3:
                st.metric("采样率", f"{localizer.fs:.2f} Hz")
                st.metric("时间分辨率", f"{1/localizer.fs*1e6:.4f} μs")
            
            # 显示公式
            st.markdown(f"""
            **计算依据公式:**

            根据文档公式(2-12):

            $$
            L_x = \\frac{{c \\cdot \\Delta\\tau}}{{2n}}
            $$

            其中：
            - $\\Delta\\tau$ = 峰值时延 = {peak_delay*1e6:.4f} μs
            - $c$ = 光速 = {c:.0e} m/s
            - $n$ = 光纤折射率 = {n:.4f}

            代入计算：
            $$
            L_x = \\frac{{({c:.0e}) \\times ({peak_delay:.4f} \\times 10^{{-6}})}}{{2 \\times {n:.4f}}} = {vibration_distance:.2f} \\text{{ m}} = {vibration_distance/1000:.2f} \\text{{ km}}
            $$
            """)
            
            # 频谱分析
            st.subheader("📊 频谱分析")
            
            fig5, (ax10, ax11) = plt.subplots(2, 1, figsize=(12, 8))
            
            # MI相位频谱
            freqs_mi, fft_mi = localizer.compute_fft(phi_MI)
            if len(freqs_mi) > 0:
                max_freq = 5000
                idx_max = np.where(freqs_mi <= max_freq)[0][-1] if np.any(freqs_mi <= max_freq) else len(freqs_mi)-1
                
                ax10.plot(freqs_mi[:idx_max], fft_mi[:idx_max], 'b-', alpha=0.7)
                ax10.set_xlabel('频率 (Hz)')
                ax10.set_ylabel('幅度')
                ax10.set_title('MI相位频谱')
                ax10.grid(True, alpha=0.3)
            
            # SI相位频谱
            freqs_si, fft_si = localizer.compute_fft(phi_SI)
            if len(freqs_si) > 0:
                idx_max = np.where(freqs_si <= max_freq)[0][-1] if np.any(freqs_si <= max_freq) else len(freqs_si)-1
                
                ax11.plot(freqs_si[:idx_max], fft_si[:idx_max], 'r-', alpha=0.7)
                ax11.set_xlabel('频率 (Hz)')
                ax11.set_ylabel('幅度')
                ax11.set_title('SI相位频谱')
                ax11.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
            
            # 误差分析
            st.subheader("📈 误差分析")
            
            # 计算距离分辨率
            time_resolution = 1 / localizer.fs
            distance_resolution = (c * time_resolution) / (2 * n)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("时间分辨率", f"{time_resolution*1e6:.4f} μs")
            with col2:
                st.metric("距离分辨率", f"{distance_resolution:.2f} m")
            with col3:
                st.metric("信噪比估计", f"{np.max(np.abs(correlation))/np.std(correlation):.2f}")
            
            st.info("""
            **注意:**
            1. 定位精度受采样率限制，距离分辨率约为 {:.2f} 米
            2. 实际定位精度还受光纤折射率测量误差、时间延迟估计误差等因素影响
            3. 如果定位结果与预期不符，请检查：
               - 数据文件是否正确（MI和SI对应同一振动事件）
               - 采样率设置是否正确
               - 光纤折射率是否准确
            """.format(distance_resolution))
    
    else:
        st.info("👈 请从左侧上传MI和SI数据文件开始分析")
        
        st.markdown("""
        ## 基于互相关时延估计的振动定位原理
        
        本程序实现了《拓展实验 光纤分布式振动定位》文档中描述的时延估计定位算法，具体步骤如下：
        
        ### 1. 相位解调
        使用文档中描述的3×3耦合器相位解调算法，分别从MI和SI的干涉信号中解调出相位差：
        - Δφ_MI(t)：迈克尔逊干涉仪相位差
        - Δφ_SI(t)：萨格纳克干涉仪相位差
        
        ### 2. 时延信号生成
        根据文档公式(2-10)：
        - φ₁(t) = Δφ_MI(t) + Δφ_SI(t) = 2φ(t-2τ_x)
        - φ₂(t) = Δφ_MI(t) - Δφ_SI(t) = 2φ(t)
        
        其中φ(t)是原始振动引起的相位调制，τ_x是振动点到参考点的时间延迟。
        
        ### 3. 互相关计算
        计算φ₁(t)和φ₂(t)的互相关函数：
        $$ R_{φ_1φ_2}(τ) = \\int φ_1(t) · φ_2(t+τ) dt $$
        
        互相关函数的峰值位置对应两信号之间的时延Δτ = 2τ_x。
        
        ### 4. 振动位置计算
        根据文档公式(2-12)：
        $$ L_x = \\frac{c · Δτ}{2n} $$
        
        其中：
        - L_x：振动点距离参考点的距离
        - c：光速 (3×10⁸ m/s)
        - n：光纤折射率 (约1.468)
        - Δτ：从互相关计算得到的时延
        
        ### 5. 输入文件要求
        需要两个Excel文件，格式与app2.py相同：
        1. **MI数据文件**：包含两路迈克尔逊干涉仪信号
        2. **SI数据文件**：包含两路萨格纳克干涉仪信号
        
        文件格式应为4列：时间、信号1、信号2、信号3（实际使用第1、2、4列）
        """)

if __name__ == "__main__":
    main()