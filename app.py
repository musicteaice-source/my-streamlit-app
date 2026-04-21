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
    
    def __init__(self, fs=None, mod_depth=2.5, noise_level=0.05):
        """
        初始化参数
        
        参数:
            fs: 采样率 (Hz)，如果为None则从时间数据计算
            mod_depth: 调制深度 (弧度)，用于模拟时使用
            noise_level: 噪声水平，用于模拟时使用
        """
        self.fs = fs
        self.mod_depth = mod_depth
        self.noise_level = noise_level
        
    def load_data(self, file_path):
        """
        从Excel文件加载数据，跳过第一行（列名）
        
        参数:
            file_path: 文件路径
            
        返回:
            t: 时间数组
            sig1: 第一路信号
            sig2: 第二路信号
        """
        try:
            # 使用header=None读取原始数据，然后跳过第一行
            df = pd.read_excel(file_path, header=None)
            
            # 检查行数是否足够
            if df.shape[0] < 2:
                st.error("数据文件行数不足，需要至少2行数据（1行列名+1行数据）")
                return None, None, None
            
            # 检查列数
            if df.shape[1] < 4:
                st.error(f"数据文件需要至少4列，但只有{df.shape[1]}列")
                return None, None, None
            
            # 获取列名（第一行）
            column_names = df.iloc[0].tolist()
            
            # 从第二行开始提取数据
            t_col1 = df.iloc[1:, 0].values.astype(float)
            sig1 = df.iloc[1:, 1].values.astype(float)
            t_col3 = df.iloc[1:, 2].values.astype(float)
            sig2 = df.iloc[1:, 3].values.astype(float)
            
            # 检查数据长度是否一致
            if len(t_col1) != len(sig1) or len(t_col1) != len(t_col3) or len(t_col1) != len(sig2):
                st.error(f"数据长度不一致: 时间1={len(t_col1)}, 信号1={len(sig1)}, 时间2={len(t_col3)}, 信号2={len(sig2)}")
                return None, None, None
            
            # 检查时间列是否一致
            if not np.allclose(t_col1, t_col3, rtol=1e-5, atol=1e-8):
                st.warning("警告：第一列和第三列时间数据不完全一致，使用第一列时间数据")
            
            # 如果未指定采样率，从时间数据计算
            if self.fs is None and len(t_col1) > 1:
                dt = np.mean(np.diff(t_col1))
                if dt > 0:
                    self.fs = 1.0 / dt
                    st.info(f"自动计算采样率: {self.fs:.2f} Hz")
                else:
                    st.error("无法从时间数据计算采样率，时间间隔为零或负值")
                    return None, None, None
            
            # 显示列名信息
            st.sidebar.info(f"列名: {column_names[:4]}")
            
            return t_col1, sig1, sig2
            
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")
            return None, None, None
    
    def remove_dc(self, signal, cutoff_freq=10):
        """
        去除直流分量（使用高通滤波器）
        """
        if self.fs is None:
            return signal - np.mean(signal)
        
        nyquist = self.fs / 2
        high = cutoff_freq / nyquist
        
        if high >= 1.0:  # 如果截止频率过高，使用简单去均值
            return signal - np.mean(signal)
        
        b, a = butter(4, high, btype='high')
        signal_ac = filtfilt(b, a, signal)
        
        return signal_ac
    
    def calculate_phase_3x3(self, sig1, sig2):
        """
        使用3x3耦合器解调算法计算相位
        基于微分交叉相乘(DCM)算法
        """
        n = len(sig1)
        
        if n < 10:
            st.error("数据点数太少，无法进行相位解调")
            return np.zeros(n, dtype=np.float64)
        
        # 方法1: 微分交叉相乘算法（针对3x3耦合器）
        # 去除直流分量
        sig1_ac = self.remove_dc(sig1)
        sig2_ac = self.remove_dc(sig2)
        
        # 计算微分
        sig1_diff = np.diff(sig1_ac, prepend=sig1_ac[0])
        sig2_diff = np.diff(sig2_ac, prepend=sig2_ac[0])
        
        # 微分交叉相乘算法
        numerator = sig1_ac * sig2_diff - sig2_ac * sig1_diff
        denominator = sig1_ac**2 + sig2_ac**2 + sig1_ac * sig2_ac
        
        # 避免除零
        epsilon = 1e-10
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        
        # 计算相位导数
        phase_derivative = 2.0 / np.sqrt(3.0) * numerator / denominator
        
        # 积分得到相位
        phase = np.cumsum(phase_derivative) / self.fs
        
        return phase.astype(np.float64)
    
    def demodulate_phase(self, t, sig1, sig2, lowpass_cutoff=None):
        """
        完整的相位解调流程
        
        参数:
            t: 时间数组
            sig1, sig2: 输入信号
            lowpass_cutoff: 低通滤波器截止频率(Hz)，如果为None则自动计算
            
        返回:
            phase_demod: 解调出的原始相位
            phase_unwrapped: 解卷绕后的相位
            phase_filtered: 滤波后的相位
        """
        # 1. 使用3x3算法解调
        phase_demod = self.calculate_phase_3x3(sig1, sig2)
        
        # 2. 相位解卷绕
        try:
            phase_unwrapped = np.unwrap(phase_demod)
        except Exception as e:
            st.warning(f"相位解卷绕时出错: {e}，使用原始相位")
            phase_unwrapped = phase_demod.copy()
        
        # 3. 去除直流偏移
        phase_unwrapped = phase_unwrapped - np.mean(phase_unwrapped)
        
        # 4. 低通滤波提取信号
        if self.fs is not None and lowpass_cutoff is not None and lowpass_cutoff < self.fs/2:
            nyquist = self.fs / 2
            low = lowpass_cutoff / nyquist
            
            if 0 < low < 1.0:
                b, a = butter(4, low, btype='low')
                phase_filtered = filtfilt(b, a, phase_unwrapped)
            else:
                phase_filtered = phase_unwrapped.copy()
        else:
            # 如果没有指定截止频率，尝试自动估计
            if self.fs is not None:
                # 默认使用采样率的1/20
                default_cutoff = self.fs / 20
                nyquist = self.fs / 2
                low = default_cutoff / nyquist
                
                if 0 < low < 1.0:
                    b, a = butter(4, low, btype='low')
                    phase_filtered = filtfilt(b, a, phase_unwrapped)
                else:
                    phase_filtered = phase_unwrapped.copy()
            else:
                phase_filtered = phase_unwrapped.copy()
        
        return phase_demod, phase_unwrapped, phase_filtered
    
    def compute_fft(self, signal, window='hann'):
        """
        计算信号的傅里叶变换
        
        参数:
            signal: 输入信号
            window: 窗函数类型
            
        返回:
            freqs: 频率数组
            fft_mag: 傅里叶变换幅度
        """
        n = len(signal)
        
        if n < 2:
            return np.array([]), np.array([])
        
        # 应用窗函数
        if window == 'hann':
            window_func = np.hanning(n)
        elif window == 'hamming':
            window_func = np.hamming(n)
        elif window == 'blackman':
            window_func = np.blackman(n)
        else:
            window_func = np.ones(n)
        
        signal_windowed = signal * window_func
        
        # 计算FFT
        fft_result = np.fft.fft(signal_windowed)
        fft_mag = np.abs(fft_result)[:n//2] * 2 / n
        freqs = np.fft.fftfreq(n, 1/self.fs)[:n//2]
        
        return freqs, fft_mag

def main():
    """Streamlit主应用程序"""
    st.set_page_config(
        page_title="光纤干涉信号相位解调系统",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 光纤Sagnac干涉信号相位解调系统")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # 文件上传
    uploaded_file = st.sidebar.file_uploader("上传数据文件 (data.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        # 从文件加载数据
        data_bytes = uploaded_file.read()
        
        # 尝试预览文件前几行
        try:
            df_preview = pd.read_excel(io.BytesIO(data_bytes), header=None, nrows=5)
            st.sidebar.info("文件前5行预览:")
            st.sidebar.dataframe(df_preview)
        except:
            pass
        
        # 采样率设置
        fs = st.sidebar.number_input("采样率 (Hz)", 
                                    min_value=1.0, 
                                    max_value=1e9, 
                                    value=1000000.0,
                                    step=1000.0,
                                    help="如果设置为0，将尝试从时间数据自动计算")
        
        if fs == 0:
            fs = None
        
        # 滤波器设置
        st.sidebar.subheader("滤波器设置")
        lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                               min_value=1.0, 
                                               max_value=50000.0,
                                               value=1000.0,
                                               step=10.0)
        
        # 数据处理设置
        st.sidebar.subheader("数据处理设置")
        remove_dc = st.sidebar.checkbox("去除直流分量", value=True)
        apply_window = st.sidebar.checkbox("应用窗函数", value=True)
        window_type = st.sidebar.selectbox("窗函数类型", 
                                          ["hann", "hamming", "blackman", "none"],
                                          index=0)
        
        # 创建解调器
        demodulator = Interferometer3x3Demodulator(fs=fs)
        
        # 加载数据
        with st.spinner("正在加载数据..."):
            t, sig1, sig2 = demodulator.load_data(io.BytesIO(data_bytes))
        
        if t is not None and sig1 is not None and sig2 is not None:
            # 显示数据信息
            st.subheader("📊 数据信息")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据点数", len(t))
            with col2:
                st.metric("采样率", f"{demodulator.fs:.2f} Hz")
            with col3:
                st.metric("时长", f"{t[-1] - t[0]:.4f} s")
            with col4:
                st.metric("通道1均值", f"{np.mean(sig1):.4f}")
            
            # 绘制原始信号
            st.subheader("📈 原始信号")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 显示全部数据
            display_points = len(t)
            
            ax1.plot(t[:display_points], sig1[:display_points], 'b-', linewidth=1, alpha=0.7)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('幅度')
            ax1.set_title('通道1信号')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t[:display_points], sig2[:display_points], 'r-', linewidth=1, alpha=0.7)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('幅度')
            ax2.set_title('通道2信号')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
            # 相位解调
            st.subheader("🔄 相位解调处理")
            
            with st.spinner("正在进行相位解调..."):
                phase_demod, phase_unwrapped, phase_filtered = demodulator.demodulate_phase(
                    t, sig1, sig2, lowpass_cutoff
                )
            
            # 绘制解调结果
            st.subheader("📈 相位解调结果")
            
            # 创建三个子图
            fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 10))
            
            # 原始解调相位
            ax3.plot(t, phase_demod, 'g-', linewidth=0.5, alpha=0.7)
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('相位 (rad)')
            ax3.set_title('解调相位（原始）')
            ax3.grid(True, alpha=0.3)
            
            # 解卷绕相位
            ax4.plot(t, phase_unwrapped, 'b-', linewidth=0.5, alpha=0.7)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('相位 (rad)')
            ax4.set_title('解调相位（解卷绕后）')
            ax4.grid(True, alpha=0.3)
            
            # 滤波后相位
            ax5.plot(t, phase_filtered, 'r-', linewidth=0.5, alpha=0.7)
            ax5.set_xlabel('时间 (s)')
            ax5.set_ylabel('相位 (rad)')
            ax5.set_title('解调相位（低通滤波后）')
            ax5.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
            # 傅里叶变换分析
            st.subheader("📊 傅里叶变换分析")
            
            # 计算FFT
            with st.spinner("正在进行傅里叶变换..."):
                window = window_type if window_type != "none" and apply_window else None
                
                # 计算原始信号的FFT
                freqs_sig1, fft_sig1 = demodulator.compute_fft(
                    sig1 - np.mean(sig1) if remove_dc else sig1, 
                    window
                )
                freqs_sig2, fft_sig2 = demodulator.compute_fft(
                    sig2 - np.mean(sig2) if remove_dc else sig2, 
                    window
                )
                
                # 计算解调相位的FFT
                freqs_phase, fft_phase = demodulator.compute_fft(phase_filtered, window)
            
            # 创建FFT图形
            fig3, (ax6, ax7) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 原始信号的FFT
            max_freq = min(10000, demodulator.fs/2)  # 限制显示频率范围
            if len(freqs_sig1) > 0:
                idx_max = min(len(freqs_sig1) - 1, np.where(freqs_sig1 <= max_freq)[0][-1] if np.any(freqs_sig1 <= max_freq) else len(freqs_sig1) - 1)
                
                ax6.plot(freqs_sig1[:idx_max], fft_sig1[:idx_max], 'b-', linewidth=0.5, alpha=0.7, label='通道1')
                ax6.plot(freqs_sig2[:idx_max], fft_sig2[:idx_max], 'r-', linewidth=0.5, alpha=0.7, label='通道2')
                ax6.set_xlabel('频率 (Hz)')
                ax6.set_ylabel('幅度')
                ax6.set_title('原始信号傅里叶变换')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
                ax6.set_xlim([0, max_freq])
            
            # 解调相位的FFT
            max_freq_phase = min(5000, demodulator.fs/2)  # 相位信号频率较低
            if len(freqs_phase) > 0:
                idx_max_phase = min(len(freqs_phase) - 1, np.where(freqs_phase <= max_freq_phase)[0][-1] if np.any(freqs_phase <= max_freq_phase) else len(freqs_phase) - 1)
                
                ax7.plot(freqs_phase[:idx_max_phase], fft_phase[:idx_max_phase], 'g-', linewidth=0.5, alpha=0.7)
                ax7.set_xlabel('频率 (Hz)')
                ax7.set_ylabel('幅度')
                ax7.set_title('解调相位傅里叶变换')
                ax7.grid(True, alpha=0.3)
                ax7.set_xlim([0, max_freq_phase])
            
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
                
                # 找到主要频率成分
                if len(fft_phase) > 0:
                    main_freq_idx = np.argmax(fft_phase[:len(fft_phase)//2])
                    main_freq = freqs_phase[main_freq_idx]
                    st.metric("主频率", f"{main_freq:.2f} Hz")
            
            with col3:
                # 计算两路信号的相关性
                correlation = np.corrcoef(sig1, sig2)[0, 1]
                st.metric("信号相关性", f"{correlation:.4f}")
                
                # 估计信噪比
                if len(phase_filtered) > 100:
                    signal_power = np.var(phase_filtered)
                    noise_power = np.var(phase_filtered - np.convolve(
                        phase_filtered, np.ones(100)/100, mode='same'
                    ))
                    if noise_power > 0:
                        snr_est = 10 * np.log10(signal_power / noise_power)
                        st.metric("信噪比估计", f"{snr_est:.2f} dB")
            
            # 数据导出
            st.subheader("💾 数据导出")
            
            # 创建结果数据框
            result_data = pd.DataFrame({
                '时间_s': t,
                '通道1_原始': sig1,
                '通道2_原始': sig2,
                '相位_原始': phase_demod,
                '相位_解卷绕': phase_unwrapped,
                '相位_滤波后': phase_filtered
            })
            
            # 将数据转换为Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_data.to_excel(writer, index=False, sheet_name='解调结果')
            
            output.seek(0)
            
            # 下载按钮
            st.download_button(
                label="下载解调结果 (Excel)",
                data=output,
                file_name=f"phase_demod_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # 显示数据预览
            with st.expander("查看解调数据预览"):
                st.dataframe(result_data.head(10))
    
    else:
        # 如果没有上传文件，显示使用说明
        st.info("👈 请从左侧上传数据文件开始分析")
        
        st.markdown("""
        ## 使用说明
        
        1. **准备数据文件**：确保数据文件为Excel格式，包含至少4列：
           - 第1列：时间数据
           - 第2列：第一路信号幅值
           - 第3列：时间数据（与第1列相同）
           - 第4列：第二路信号幅值
        
        2. **文件格式**：第一行是列名，从第二行开始是数据
        
        3. **上传文件**：点击左侧"上传数据文件"按钮上传文件
        
        4. **配置参数**：
           - 采样率：可以手动输入，或设置为0来自动计算
           - 滤波器设置：设置低通滤波器截止频率
           - 数据处理选项：选择是否去除直流分量、应用窗函数等
        
        5. **查看结果**：
           - 原始信号波形
           - 相位解调结果
           - 傅里叶变换频谱
           - 解调结果统计
        
        6. **下载结果**：可以下载包含解调结果的Excel文件
        """)

if __name__ == "__main__":
    main()