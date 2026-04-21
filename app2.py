import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import streamlit as st
import io
from datetime import datetime

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class Interferometer3x3Demodulator:
    """3x3耦合器干涉信号相位解调器"""
    
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
        
        return phase, phase_unwrapped, phase_filtered
    
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
        page_title="光纤干涉相位解调系统 (文档算法)",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 光纤干涉相位解调系统 - 文档算法实现")
    st.markdown("基于《光纤实验讲义》第19-20页的相位解调算法")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    uploaded_file = st.sidebar.file_uploader("上传数据文件 (data.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        # 采样率设置
        fs = st.sidebar.number_input("采样率 (Hz)", 
                                    min_value=1.0, 
                                    max_value=1e9, 
                                    value=1000000.0,
                                    step=1000.0)
        
        # 解调方法选择
        st.sidebar.subheader("解调算法")
        method = st.sidebar.selectbox("选择算法", 
                                     ["文档算法", "DCM算法"],
                                     index=0)
        
        # 滤波器设置
        st.sidebar.subheader("滤波器设置")
        lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                               min_value=1.0, 
                                               max_value=fs/2 if fs else 1e6, 
                                               value=1000.0,
                                               step=10.0)
        
        # 创建解调器并加载数据
        demodulator = Interferometer3x3Demodulator(fs=fs)
        
        with st.spinner("正在加载数据..."):
            t, sig1, sig2 = demodulator.load_data(io.BytesIO(uploaded_file.read()))
        
        if t is not None and sig1 is not None and sig2 is not None:
            # 显示数据信息
            st.subheader("📊 数据信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据点数", len(t))
            with col2:
                st.metric("采样率", f"{fs:.2f} Hz")
            with col3:
                st.metric("时长", f"{t[-1] - t[0]:.4f} s")
            
            # 绘制原始信号
            st.subheader("📈 原始信号")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            display_points = len(t)
            ax1.plot(t[:display_points], sig1[:display_points], 'b-', alpha=0.7, linewidth=1)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('幅度')
            ax1.set_title('通道1信号 (P1)')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t[:display_points], sig2[:display_points], 'r-', alpha=0.7, linewidth=1)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('幅度')
            ax2.set_title('通道2信号 (P2)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
            # 相位解调
            st.subheader("🔄 相位解调结果")
            
            method_key = 'document' if method == '文档算法' else 'dcm'
            with st.spinner(f"正在使用{method}进行相位解调..."):
                phase_demod, phase_unwrapped, phase_filtered = demodulator.demodulate_phase(
                    t, sig1, sig2, method=method_key, lowpass_cutoff=lowpass_cutoff
                )
            
            # 绘制解调结果
            fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 10))
            
            ax3.plot(t, phase_demod, 'g-', alpha=0.7, linewidth=0.5)
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('相位 (rad)')
            ax3.set_title(f'{method} - 解调相位（原始）')
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(t, phase_unwrapped, 'b-', alpha=0.7, linewidth=0.5)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('相位 (rad)')
            ax4.set_title(f'{method} - 解调相位（解卷绕后）')
            ax4.grid(True, alpha=0.3)
            
            ax5.plot(t, phase_filtered, 'r-', alpha=0.7, linewidth=0.5)
            ax5.set_xlabel('时间 (s)')
            ax5.set_ylabel('相位 (rad)')
            ax5.set_title(f'{method} - 解调相位（低通滤波后）')
            ax5.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
            # 傅里叶变换分析
            st.subheader("📊 傅里叶变换分析")
            
            freqs_phase, fft_phase = demodulator.compute_fft(phase_filtered)
            
            fig3, ax6 = plt.subplots(1, 1, figsize=(12, 4))
            if len(freqs_phase) > 0:
                max_freq = min(5000, fs/2)
                idx_max = np.where(freqs_phase <= max_freq)[0][-1] if np.any(freqs_phase <= max_freq) else len(freqs_phase)-1
                
                ax6.plot(freqs_phase[:idx_max], fft_phase[:idx_max], 'g-', alpha=0.7)
                ax6.set_xlabel('频率 (Hz)')
                ax6.set_ylabel('幅度')
                ax6.set_title('解调相位傅里叶变换')
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            # 结果显示
            st.subheader("📊 解调结果统计")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("相位均值", f"{np.mean(phase_filtered):.6f} rad")
                st.metric("相位标准差", f"{np.std(phase_filtered):.6f} rad")
            
            with col2:
                phase_range = np.ptp(phase_filtered)
                st.metric("相位范围", f"{phase_range:.6f} rad")
            
            with col3:
                if len(fft_phase) > 0:
                    main_freq_idx = np.argmax(fft_phase[:len(fft_phase)//2])
                    main_freq = freqs_phase[main_freq_idx]
                    st.metric("主频率", f"{main_freq:.2f} Hz")
            
            # 算法对比 (可选)
            if st.checkbox("显示算法对比"):
                st.subheader("🔄 算法对比")
                
                with st.spinner("正在使用两种算法解调..."):
                    # 文档算法
                    phase_doc, _, phase_doc_filt = demodulator.demodulate_phase(
                        t, sig1, sig2, method='document', lowpass_cutoff=lowpass_cutoff
                    )
                    # DCM算法
                    phase_dcm, _, phase_dcm_filt = demodulator.demodulate_phase(
                        t, sig1, sig2, method='dcm', lowpass_cutoff=lowpass_cutoff
                    )
                
                fig4, (ax7, ax8) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax7.plot(t, phase_doc_filt, 'b-', alpha=0.7, linewidth=0.5, label='文档算法')
                ax7.plot(t, phase_dcm_filt, 'r--', alpha=0.7, linewidth=0.5, label='DCM算法')
                ax7.set_xlabel('时间 (s)')
                ax7.set_ylabel('相位 (rad)')
                ax7.set_title('算法对比 - 滤波后相位')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
                
                # 计算差值
                phase_diff = phase_doc_filt - phase_dcm_filt
                ax8.plot(t, phase_diff, 'k-', alpha=0.7, linewidth=0.5)
                ax8.set_xlabel('时间 (s)')
                ax8.set_ylabel('相位差 (rad)')
                ax8.set_title('算法差值 (文档算法 - DCM算法)')
                ax8.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
                
                st.info(f"算法差值统计: 均值={np.mean(phase_diff):.6f} rad, 标准差={np.std(phase_diff):.6f} rad")
    
    else:
        st.info("👈 请从左侧上传数据文件开始分析")
        
        st.markdown("""
        ## 基于文档算法的相位解调原理
        
        本程序实现了《光纤实验讲义》第19-20页描述的相位解调算法，主要步骤如下：
        
        1. **信号归一化**：对两路干涉信号P1(t)和P2(t)进行归一化处理
           - P1'(t) = 2 × [(P1(t) - P1_min) / (P1_max - P1_min)] - 1
           - P2'(t) = 2 × [(P2(t) - P2_min) / (P2_max - P2_min)] - 1
        
        2. **组合信号计算**：
           - P+(t) = P1'(t) + P2'(t) = -cos[Δφ(t)]
           - P-(t) = P1'(t) - P2'(t) = -√3 × sin[Δφ(t)]
        
        3. **相位计算**：
           - Δφ(t) = arctan[ P-(t) / (√3 × P+(t)) ]
        
        4. **相位解卷绕**：对不连续点进行加减π处理，保证相位连续性
        
        **注意事项**：
        - 算法要求调制信号足够大，使干涉信号能达到其最大最小值
        - 适用于3×3耦合器输出相位差为120°的两路信号
        - 相比DCM算法，避免了积分漂移问题
        """)

if __name__ == "__main__":
    main()