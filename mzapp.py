import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import streamlit as st
import io
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class MZI_Demodulator:
    """马赫-曾德尔干涉仪相位解调器"""
    
    def __init__(self, fs=None):
        self.fs = fs
        self.I_signal = None
        self.Q_signal = None
        
    def load_mzi_data(self, file_path, p1_col=1, p2_col=2):
        """加载MZI数据"""
        try:
            df = pd.read_excel(file_path, header=None)
            
            if df.shape[0] < 2:
                st.error("数据文件行数不足")
                return None, None, None
            
            t = df.iloc[1:, 0].values.astype(float)
            p1 = df.iloc[1:, p1_col].values.astype(float)
            p2 = df.iloc[1:, p2_col].values.astype(float)
            
            if self.fs is None and len(t) > 1:
                self.fs = 1.0 / np.mean(np.diff(t))
                st.sidebar.info(f"自动计算采样率: {self.fs:.2f} Hz")
            
            return t, p1, p2
            
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")
            return None, None, None
    
    def create_quadrature_signals(self, p1, p2, method='hilbert'):
        """从MZI的反相输出生成正交信号"""
        p1_ac = p1 - np.mean(p1)
        p2_ac = p2 - np.mean(p2)
        
        if method == 'hilbert':
            I = p1_ac
            analytic_signal = hilbert(I)
            Q = np.imag(analytic_signal)
            
        elif method == 'differential':
            I = p1_ac
            dt = 1/self.fs
            Q = -np.gradient(I) / dt
            
            I_amp = np.std(I)
            Q_amp = np.std(Q) if np.std(Q) > 0 else 1
            Q = Q * (I_amp / Q_amp)
            
        elif method == 'combine':  # 修复：这里应该是'combine'
            I = p1_ac - p2_ac
            analytic_signal = hilbert(I)
            Q = np.imag(analytic_signal)
            
        elif method == 'phase_shift':
            I = p1_ac
            n_taps = 101
            t = np.arange(n_taps) - (n_taps-1)//2
            h = np.zeros_like(t, dtype=float)
            
            for i, ti in enumerate(t):
                if ti == 0:
                    h[i] = 0
                else:
                    h[i] = (1 - np.cos(np.pi * ti)) / (np.pi * ti)
            
            Q = np.convolve(I, h, mode='same')
            
        else:
            raise ValueError(f"未知的正交信号生成方法: {method}")
        
        I_norm = I / (np.std(I) + 1e-10)
        Q_norm = Q / (np.std(Q) + 1e-10)
        
        self.I_signal = I_norm
        self.Q_signal = Q_norm
        
        return I_norm, Q_norm
    
    def calibrate_quadrature(self, I, Q):
        """校正I-Q信号的正交性和幅度平衡"""
        data = np.column_stack([I, Q])
        data_centered = data - np.mean(data, axis=0)
        
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        scale = 1.0 / np.sqrt(s)
        data_corrected = data_centered @ Vt.T * scale
        
        I_corr = data_corrected[:, 0]
        Q_corr = data_corrected[:, 1]
        
        return I_corr, Q_corr
    
    def demodulate_phase_mzi(self, t, p1, p2, method='combine'):  # 修复：这里应该是'combine'
        """MZI相位解调主函数"""
        if method == 'hilbert':
            I, Q = self.create_quadrature_signals(p1, p2, method='hilbert')
            phase = np.arctan2(Q, I)
            
        elif method == 'differential':
            I, Q = self.create_quadrature_signals(p1, p2, method='differential')
            I_corr, Q_corr = self.calibrate_quadrature(I, Q)
            phase = np.arctan2(Q_corr, I_corr)
            
        elif method == 'combine':  # 修复：这里应该是'combine'
            I, Q = self.create_quadrature_signals(p1, p2, method='combine')
            I_corr, Q_corr = self.calibrate_quadrature(I, Q)
            phase = np.arctan2(Q_corr, I_corr)
            
        elif method == 'fringe_counting':
            p1_norm = (p1 - np.mean(p1)) / np.std(p1)
            zero_crossings = np.where(np.diff(np.sign(p1_norm)))[0]
            phase = np.zeros_like(t)
            fringe_count = 0
            
            for i in range(1, len(t)):
                if i-1 in zero_crossings:
                    slope = p1_norm[i] - p1_norm[i-1]
                    if slope > 0:
                        fringe_count += 0.5
                    else:
                        fringe_count -= 0.5
                phase[i] = fringe_count * np.pi
            return phase
            
        elif method == 'direct_atan':
            p1_norm = (p1 - np.mean(p1)) / np.std(p1)
            p2_norm = (p2 - np.mean(p2)) / np.std(p2)
            phase = np.arctan2(p2_norm, p1_norm)
            
        else:
            raise ValueError(f"未知的解调方法: {method}")
        
        phase_unwrapped = np.unwrap(phase)
        phase_unwrapped = phase_unwrapped - np.mean(phase_unwrapped)
        
        return phase_unwrapped
    
    def evaluate_signal_quality(self, p1, p2):
        """评估MZI信号质量"""
        metrics = {}
        metrics['p1_mean'] = np.mean(p1)
        metrics['p2_mean'] = np.mean(p2)
        metrics['p1_std'] = np.std(p1)
        metrics['p2_std'] = np.std(p2)
        
        correlation = np.corrcoef(p1, p2)[0, 1]
        metrics['correlation'] = correlation
        metrics['amplitude_ratio'] = metrics['p1_std'] / (metrics['p2_std'] + 1e-10)
        
        p1_ac = p1 - metrics['p1_mean']
        p2_ac = p2 - metrics['p2_mean']
        analytic1 = hilbert(p1_ac)
        analytic2 = hilbert(p2_ac)
        phase1 = np.unwrap(np.angle(analytic1))
        phase2 = np.unwrap(np.angle(analytic2))
        phase_diff = np.mean(np.mod(np.abs(phase1 - phase2), 2*np.pi))
        metrics['estimated_phase_diff'] = phase_diff
        
        return metrics

def main():
    """Streamlit主应用程序"""
    st.set_page_config(
        page_title="MZI相位解调系统",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 马赫-曾德尔干涉仪相位解调系统")
    st.markdown("适用于2x2耦合器MZI结构的相位解调")
    
    st.sidebar.header("MZI配置参数")
    uploaded_file = st.sidebar.file_uploader("上传MZI数据文件 (Excel格式)", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        fs = st.sidebar.number_input("采样率 (Hz)", 
                                    min_value=1.0, 
                                    max_value=1e9, 
                                    value=1000000.0,
                                    step=1000.0)
        
        st.sidebar.subheader("信号列选择")
        col_p1 = st.sidebar.number_input("P1信号列索引（从0开始）", 
                                        min_value=1, 
                                        max_value=10, 
                                        value=1,
                                        step=1)
        col_p2 = st.sidebar.number_input("P2信号列索引（从0开始）", 
                                        min_value=1, 
                                        max_value=10, 
                                        value=2,
                                        step=1)
        
        st.sidebar.subheader("解调方法选择")
        method = st.sidebar.selectbox(
            "选择解调方法",
            ["combine", "hilbert", "differential", "fringe_counting", "direct_atan"],  # 修复：这里应该是"combine"
            format_func=lambda x: {
                "combine": "结合P1和P2（推荐）",  # 修复：这里应该是"combine"
                "hilbert": "Hilbert变换法",
                "differential": "微分法",
                "fringe_counting": "条纹计数法",
                "direct_atan": "直接反正切法（不推荐）"
            }[x]
        )
        
        st.sidebar.subheader("滤波器设置")
        enable_filter = st.sidebar.checkbox("启用低通滤波", value=True)
        if enable_filter:
            lowpass_cutoff = st.sidebar.number_input("低通滤波器截止频率 (Hz)", 
                                                   min_value=1.0, 
                                                   max_value=fs/2 if fs else 1e6, 
                                                   value=1000.0,
                                                   step=10.0)
        else:
            lowpass_cutoff = None
        
        demodulator = MZI_Demodulator(fs=fs)
        
        with st.spinner("正在加载MZI数据..."):
            try:
                df = pd.read_excel(io.BytesIO(uploaded_file.read()), header=None)
                
                if df.shape[0] < 2:
                    st.error("数据文件行数不足，需要至少2行数据")
                    st.stop()
                
                t = df.iloc[1:, 0].values.astype(float)
                p1 = df.iloc[1:, col_p1].values.astype(float)
                p2 = df.iloc[1:, col_p2].values.astype(float)
                
                if demodulator.fs is None and len(t) > 1:
                    demodulator.fs = 1.0 / np.mean(np.diff(t))
                    st.sidebar.info(f"自动计算采样率: {demodulator.fs:.2f} Hz")
                    
            except Exception as e:
                st.error(f"加载数据时出错: {str(e)}")
                st.stop()
        
        if t is not None and p1 is not None and p2 is not None:
            st.subheader("📊 数据信息与信号质量")
            metrics = demodulator.evaluate_signal_quality(p1, p2)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据点数", len(t))
                st.metric("P1-P2相关性", f"{metrics['correlation']:.3f}")
            with col2:
                st.metric("采样率", f"{demodulator.fs:.2f} Hz")
                st.metric("幅度比(P1/P2)", f"{metrics['amplitude_ratio']:.3f}")
            with col3:
                st.metric("时长", f"{t[-1] - t[0]:.4f} s")
                st.metric("估计相位差", f"{metrics['estimated_phase_diff']*180/np.pi:.1f}°")
            with col4:
                st.metric("P1标准差", f"{metrics['p1_std']:.3f}")
                st.metric("P2标准差", f"{metrics['p2_std']:.3f}")
            
            if abs(metrics['correlation']) > 0.9:
                st.warning("⚠️ P1和P2高度相关，可能是反相或同相，而非正交")
                st.info("💡 对于MZI结构，这是正常现象。P1和P2通常是反相的。")
            
            st.subheader("📈 MZI原始信号")
            fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            display_points = min(5000, len(t))
            
            ax1.plot(t[:display_points], p1[:display_points], 'b-', alpha=0.7, linewidth=1)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('幅度')
            ax1.set_title('探测器P1信号')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t[:display_points], p2[:display_points], 'r-', alpha=0.7, linewidth=1)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('幅度')
            ax2.set_title('探测器P2信号')
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(t[:display_points], p1[:display_points], 'b-', alpha=0.5, linewidth=1, label='P1')
            ax3.plot(t[:display_points], p2[:display_points], 'r-', alpha=0.5, linewidth=1, label='P2')
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('幅度')
            ax3.set_title('P1和P2信号对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            
            st.subheader("🔄 生成的正交信号")
            methods = ['hilbert', 'differential', 'combine']  # 修复：这里应该是'combine'
            method_names = ['Hilbert变换', '微分法', '结合P1/P2']
            
            fig2, axes = plt.subplots(3, 2, figsize=(14, 10))
            
            for idx, (method_name, display_name) in enumerate(zip(methods, method_names)):
                try:
                    I, Q = demodulator.create_quadrature_signals(p1, p2, method=method_name)
                    
                    ax_sig = axes[idx, 0]
                    ax_sig.plot(t[:display_points], I[:display_points], 'b-', alpha=0.7, linewidth=0.5, label='I信号')
                    ax_sig.plot(t[:display_points], Q[:display_points], 'r-', alpha=0.7, linewidth=0.5, label='Q信号')
                    ax_sig.set_xlabel('时间 (s)')
                    ax_sig.set_ylabel('幅度')
                    ax_sig.set_title(f'{display_name} - 生成的I/Q信号')
                    ax_sig.legend(fontsize=8)
                    ax_sig.grid(True, alpha=0.3)
                    
                    ax_liss = axes[idx, 1]
                    ax_liss.plot(I[:display_points], Q[:display_points], 'g-', alpha=0.5, linewidth=0.5)
                    ax_liss.set_xlabel('I信号')
                    ax_liss.set_ylabel('Q信号')
                    ax_liss.set_title(f'{display_name} - 李萨如图')
                    ax_liss.grid(True, alpha=0.3)
                    ax_liss.axis('equal')
                    
                except Exception as e:
                    st.warning(f"方法 {display_name} 生成正交信号时出错: {str(e)}")
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            
            st.subheader("🎯 相位解调结果")
            
            with st.spinner(f"正在使用{method}方法进行相位解调..."):
                try:
                    phase = demodulator.demodulate_phase_mzi(t, p1, p2, method=method)
                    
                    if enable_filter and lowpass_cutoff is not None and demodulator.fs is not None:
                        nyquist = demodulator.fs / 2
                        low = lowpass_cutoff / nyquist
                        if 0 < low < 1.0:
                            b, a = butter(4, low, btype='low')
                            phase_filtered = filtfilt(b, a, phase)
                        else:
                            phase_filtered = phase
                    else:
                        phase_filtered = phase
                    
                except Exception as e:
                    st.error(f"相位解调出错: {str(e)}")
                    st.stop()
            
            fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
            ax3.plot(t[:display_points], phase[:display_points], 'b-', alpha=0.7, linewidth=0.5, label='原始相位')
            ax3.plot(t[:display_points], phase_filtered[:display_points], 'r-', alpha=0.7, linewidth=1, label='滤波后相位')
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('相位 (rad)')
            ax3.set_title('解调相位结果')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            if len(t) > 1:
                dt = np.mean(np.diff(t))
                phase_diff = np.diff(phase_filtered) / dt
                phase_diff[phase_diff > np.pi/dt] = 0
                phase_diff[phase_diff < -np.pi/dt] = 0
                
                ax4.plot(t[1:display_points], phase_diff[:display_points-1], 'g-', alpha=0.7, linewidth=0.5)
                ax4.set_xlabel('时间 (s)')
                ax4.set_ylabel('频率 (rad/s)')
                ax4.set_title('相位变化率（瞬时频率）')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            st.subheader("📊 频谱分析")
            n = len(phase_filtered)
            if n > 1:
                window = np.hanning(n)
                phase_windowed = phase_filtered * window
                fft_result = np.fft.fft(phase_windowed)
                fft_mag = np.abs(fft_result)[:n//2] * 2 / n
                freqs = np.fft.fftfreq(n, 1/demodulator.fs)[:n//2]
                
                fig4, ax5 = plt.subplots(1, 1, figsize=(12, 4))
                max_freq = min(5000, demodulator.fs/2)
                idx_max = np.where(freqs <= max_freq)[0][-1] if np.any(freqs <= max_freq) else len(freqs)-1
                
                ax5.plot(freqs[:idx_max], fft_mag[:idx_max], 'b-', alpha=0.7)
                ax5.set_xlabel('频率 (Hz)')
                ax5.set_ylabel('幅度')
                ax5.set_title('解调相位频谱')
                ax5.grid(True, alpha=0.3)
                
                if len(fft_mag) > 0:
                    main_freq_idx = np.argmax(fft_mag[:len(fft_mag)//2])
                    main_freq = freqs[main_freq_idx]
                    main_amp = fft_mag[main_freq_idx]
                    ax5.axvline(x=main_freq, color='r', linestyle='--', alpha=0.5)
                    ax5.text(main_freq*1.1, main_amp*0.9, f'{main_freq:.2f} Hz', color='r')
                
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            
            st.subheader("📈 解调结果统计")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                phase_mean = np.mean(phase_filtered)
                st.metric("相位均值", f"{phase_mean:.6f} rad")
            with col2:
                phase_std = np.std(phase_filtered)
                st.metric("相位标准差", f"{phase_std:.6f} rad")
            with col3:
                phase_range = np.ptp(phase_filtered)
                st.metric("相位范围", f"{phase_range:.6f} rad")
                st.metric("相位范围(条纹数)", f"{phase_range/(2*np.pi):.3f}")
            with col4:
                if 'main_freq' in locals():
                    st.metric("主频率", f"{main_freq:.2f} Hz")
                st.metric("信噪比(估算)", f"{20*np.log10(phase_range/(phase_std+1e-10)):.1f} dB")
            
            st.subheader("📊 相位分布")
            fig5, ax6 = plt.subplots(1, 1, figsize=(10, 4))
            ax6.hist(phase_filtered, bins=100, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('相位 (rad)')
            ax6.set_ylabel('频数')
            ax6.set_title('相位分布直方图')
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
            
    else:
        st.info("👈 请从左侧上传MZI数据文件开始分析")
        
        st.markdown("""
        ## 马赫-曾德尔干涉仪(MZI)相位解调说明
        
        本程序专门针对2x2耦合器MZI结构的相位解调，考虑了MZI输出信号的特殊性：
        
        ### MZI信号特性
        在标准的MZI结构中，两个探测器的输出为：
        - P1 = I₀[1 + V cos(Δφ)]
        - P2 = I₀[1 - V cos(Δφ)]
        
        其中：
        - I₀是光强
        - V是可见度
        - Δφ是待测相位
        
        ### 关键点
        1. **P1和P2是反相的**，不是正交的
        2. 不能直接对P1和P2使用反正切法
        3. 需要先构造正交信号
        
        ### 支持的解调方法
        1. **结合P1和P2法（推荐）**：使用P1-P2作为I信号，再通过Hilbert变换生成Q信号
        2. **Hilbert变换法**：对P1进行Hilbert变换生成正交信号
        3. **微分法**：对P1微分生成正交信号
        4. **条纹计数法**：通过过零点计数，适用于大信号
        
        ### 数据文件格式
        Excel文件应包含：
        - 第1列：时间
        - 第2列：探测器P1信号
        - 第3列：探测器P2信号
        
        ### 使用建议
        1. 首先检查"P1-P2相关性"，理想情况下应接近-1（完全反相）
        2. 尝试不同的解调方法，观察李萨如图是否接近圆形
        3. 如果解调结果不理想，尝试不同的正交信号生成方法
        """)

if __name__ == "__main__":
    main()