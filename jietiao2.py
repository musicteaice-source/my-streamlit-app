import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class Interferometer3x3Demodulator:
    """
    3x3耦合器干涉信号解调器
    模拟两路相位差120度的干涉信号，并解调相位
    """
    
    def __init__(self, fs, f_mod, mod_depth, noise_level=0.01):
        """
        初始化参数
        
        参数:
            fs: 采样率 (Hz)
            f_mod: 调制频率 (Hz)
            mod_depth: 调制深度 (弧度)
            noise_level: 噪声水平
        """
        self.fs = fs
        self.f_mod = f_mod
        self.mod_depth = mod_depth
        self.noise_level = noise_level
        
    def generate_signals(self, t, signal_phase):
        """
        生成3x3耦合器输出的两路干涉信号（相位差120度）
        
        参数:
            t: 时间数组
            signal_phase: 待测相位信号（慢变）
            
        返回:
            sig1, sig2: 两路相位差120度的干涉信号
        """
        n = len(t)
        
        # 模拟干涉信号
        # 直流分量
        DC1 = 1.0 + 0.1 * np.sin(2*np.pi*0.5*t)  # 带有轻微漂移
        DC2 = 1.0 + 0.1 * np.sin(2*np.pi*0.51*t)  # 两路DC分量略有不同
        
        # 交流分量幅度
        AC_amp1 = 0.5 + 0.05 * np.sin(2*np.pi*0.3*t)  # 幅度有轻微变化
        AC_amp2 = 0.5 + 0.05 * np.sin(2*np.pi*0.31*t)
        
        # 相位调制（高频载波）
        phase_carrier = self.mod_depth * np.sin(2*np.pi*self.f_mod*t)
        
        # 总相位 = 载波相位 + 信号相位
        total_phase = phase_carrier + signal_phase
        
        # 生成两路相位差120度的信号
        sig1 = DC1 + AC_amp1 * np.cos(total_phase)  # 第一路
        sig2 = DC2 + AC_amp2 * np.cos(total_phase + 2*np.pi/3)  # 第二路，相位差120度
        
        # 添加噪声
        noise1 = self.noise_level * np.random.randn(n)
        noise2 = self.noise_level * np.random.randn(n)
        
        return sig1 + noise1, sig2 + noise2, total_phase
    
    def remove_dc(self, signal, cutoff_freq=10):
        """
        去除直流分量（使用高通滤波器）
        
        参数:
            signal: 输入信号
            cutoff_freq: 截止频率 (Hz)
            
        返回:
            signal_ac: 去除直流后的信号
        """
        # 设计高通滤波器
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
        
        参数:
            sig1, sig2: 两路相位差120度的干涉信号
            
        返回:
            phase: 解调出的相位
        """
        n = len(sig1)
        
        # 方法1: 微分交叉相乘算法（针对3x3耦合器）
        # 去除直流分量
        sig1_ac = self.remove_dc(sig1)
        sig2_ac = self.remove_dc(sig2)
        
        # 计算微分
        sig1_diff = np.diff(sig1_ac, prepend=sig1_ac[0])  # 使用prepend保持长度
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
        
        return phase.astype(np.float64)  # 确保返回浮点数
    
    def demodulate_3x3(self, t, sig1, sig2, signal_phase):
        """
        完整的3x3解调流程
        
        参数:
            t: 时间数组
            sig1, sig2: 输入信号
            signal_phase: 真实相位信号（用于对比）
            
        返回:
            phase_demod: 解调出的相位
            phase_unwrapped: 解卷绕后的相位
            phase_filtered: 滤波后的相位
        """
        # 1. 使用3x3算法解调
        phase_demod = self.calculate_phase_3x3(sig1, sig2)
        
        # 确保是浮点数类型
        phase_demod = phase_demod.astype(np.float64)
        
        # 2. 相位解卷绕
        try:
            phase_unwrapped = np.unwrap(phase_demod)
        except Exception as e:
            print(f"解卷绕错误: {e}")
            print(f"phase_demod数据类型: {phase_demod.dtype}, 形状: {phase_demod.shape}")
            print(f"phase_demod前5个值: {phase_demod[:5]}")
            # 如果解卷绕失败，使用原始相位
            phase_unwrapped = phase_demod.copy()
        
        # 3. 去除低频漂移（可选）
        phase_unwrapped = phase_unwrapped - np.mean(phase_unwrapped)
        
        # 4. 提取低频信号成分（去除高频载波）
        # 设计低通滤波器，截止频率远低于调制频率
        nyquist = self.fs / 2
        lowpass_cutoff = self.f_mod / 20  # 截止频率为调制频率的1/20
        low = lowpass_cutoff / nyquist
        
        if low < 1.0:
            b, a = butter(4, low, btype='low')
            phase_filtered = filtfilt(b, a, phase_unwrapped)
        else:
            phase_filtered = phase_unwrapped.copy()
        
        return phase_demod, phase_unwrapped, phase_filtered
    
    def plot_results(self, t, sig1, sig2, signal_phase, phase_demod, phase_unwrapped, phase_filtered):
        """
        可视化结果
        """
        fig, axes = plt.subplots(4, 2, figsize=(14, 10))
        
        # 1. 原始信号
        axes[0, 0].plot(t[:2000], sig1[:2000], 'b-', label='信号1', alpha=0.7, linewidth=1)
        axes[0, 0].plot(t[:2000], sig2[:2000], 'r-', label='信号2', alpha=0.7, linewidth=1)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].set_title('3x3耦合器输出信号（前2000点）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 频谱分析
        n = len(t)
        freqs = np.fft.fftfreq(n, 1/self.fs)[:n//2]
        sig1_fft = np.abs(np.fft.fft(sig1)[:n//2])
        sig2_fft = np.abs(np.fft.fft(sig2)[:n//2])
        
        axes[0, 1].plot(freqs[:n//20], sig1_fft[:n//20], 'b-', alpha=0.7, label='信号1频谱')
        axes[0, 1].plot(freqs[:n//20], sig2_fft[:n//20], 'r-', alpha=0.7, label='信号2频谱')
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].set_title('信号频谱')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 解调相位与真实相位对比
        axes[1, 0].plot(t, signal_phase, 'b-', label='真实相位', alpha=0.7, linewidth=1.5)
        axes[1, 0].plot(t, phase_unwrapped, 'r--', label='解调相位(解卷绕)', alpha=0.7, linewidth=1)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('相位 (rad)')
        axes[1, 0].set_title('解调相位与真实相位对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 滤波后相位
        # 去掉开头和结尾的瞬态响应
        n_transient = 100
        if len(t) > 2*n_transient:
            t_plot = t[n_transient:-n_transient]
            signal_phase_plot = signal_phase[n_transient:-n_transient]
            phase_filtered_plot = phase_filtered[n_transient:-n_transient]
        else:
            t_plot = t
            signal_phase_plot = signal_phase
            phase_filtered_plot = phase_filtered
            
        axes[1, 1].plot(t_plot, signal_phase_plot, 'b-', label='真实相位', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(t_plot, phase_filtered_plot, 'g-', label='滤波后相位', alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('相位 (rad)')
        axes[1, 1].set_title('滤波后相位与真实相位对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 相位误差
        phase_error = phase_filtered_plot - signal_phase_plot
        axes[2, 0].plot(t_plot, phase_error, 'k-', linewidth=0.5)
        axes[2, 0].set_xlabel('时间 (s)')
        axes[2, 0].set_ylabel('相位误差 (rad)')
        axes[2, 0].set_title('相位误差')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 相位误差直方图
        axes[2, 1].hist(phase_error, bins=50, edgecolor='black', alpha=0.7)
        axes[2, 1].set_xlabel('相位误差 (rad)')
        axes[2, 1].set_ylabel('频数')
        axes[2, 1].set_title('相位误差分布')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. 李萨如图形（验证120度相位差）
        # 去除直流分量
        sig1_ac = sig1 - np.mean(sig1)
        sig2_ac = sig2 - np.mean(sig2)
        
        axes[3, 0].plot(sig1_ac[:2000], sig2_ac[:2000], 'b.', markersize=1, alpha=0.5)
        axes[3, 0].set_xlabel('信号1 (去直流)')
        axes[3, 0].set_ylabel('信号2 (去直流)')
        axes[3, 0].set_title('李萨如图形（验证120°相位差）')
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 0].axis('equal')
        
        # 8. 相位差的统计分析
        # 计算相位差的统计信息
        error_mean = np.mean(phase_error)
        error_std = np.std(phase_error)
        error_pp = np.ptp(phase_error)
        
        axes[3, 1].text(0.1, 0.8, f'误差统计:', fontsize=12, fontweight='bold')
        axes[3, 1].text(0.1, 0.7, f'均值: {error_mean:.6f} rad', fontsize=10)
        axes[3, 1].text(0.1, 0.6, f'标准差: {error_std:.6f} rad', fontsize=10)
        axes[3, 1].text(0.1, 0.5, f'峰峰值: {error_pp:.6f} rad', fontsize=10)
        axes[3, 1].text(0.1, 0.4, f'采样率: {self.fs/1e3:.1f} kHz', fontsize=10)
        axes[3, 1].text(0.1, 0.3, f'调制频率: {self.f_mod/1e3:.1f} kHz', fontsize=10)
        axes[3, 1].text(0.1, 0.2, f'调制深度: {self.mod_depth:.2f} rad', fontsize=10)
        axes[3, 1].set_xlim(0, 1)
        axes[3, 1].set_ylim(0, 1)
        axes[3, 1].axis('off')
        axes[3, 1].set_title('性能统计')
        
        plt.tight_layout()
        plt.show()
        
        return error_mean, error_std, error_pp

# 主程序
if __name__ == "__main__":
    # 设置参数
    fs = 1e6  # 采样率 1 MHz
    f_mod = 100e3  # 调制频率 100 kHz
    mod_depth = 2.5  # 调制深度 2.5 rad
    duration = 0.01  # 信号时长 10 ms
    
    # 创建解调器实例
    demod = Interferometer3x3Demodulator(fs, f_mod, mod_depth, noise_level=0.05)
    
    # 生成时间序列
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # 生成待测相位信号（低频信号）
    # 包含多个频率成分
    f_signal1 = 500  # 500 Hz
    f_signal2 = 1200  # 1200 Hz
    signal_phase = 0.5 * np.sin(2*np.pi*f_signal1*t) + 0.3 * np.sin(2*np.pi*f_signal2*t)
    
    # 生成3x3耦合器输出的两路信号
    sig1, sig2, total_phase = demod.generate_signals(t, signal_phase)
    
    print("信号生成完成:")
    print(f"  采样点数: {n_samples}")
    print(f"  信号1均值: {np.mean(sig1):.4f}, 标准差: {np.std(sig1):.4f}")
    print(f"  信号2均值: {np.mean(sig2):.4f}, 标准差: {np.std(sig2):.4f}")
    print(f"  两路信号相关系数: {np.corrcoef(sig1, sig2)[0,1]:.4f}")
    
    # 解调相位
    phase_demod, phase_unwrapped, phase_filtered = demod.demodulate_3x3(t, sig1, sig2, signal_phase)
    
    # 计算性能指标
    error_mean, error_std, error_pp = demod.plot_results(t, sig1, sig2, signal_phase, 
                                                        phase_demod, phase_unwrapped, phase_filtered)
    
    print("\n解调性能:")
    print(f"  相位误差均值: {error_mean:.6f} rad")
    print(f"  相位误差标准差: {error_std:.6f} rad")
    print(f"  相位误差峰峰值: {error_pp:.6f} rad")
    print(f"  信噪比(近似): {20*np.log10(np.std(signal_phase)/error_std):.2f} dB")
    
    # 添加调试信息
    print(f"\n调试信息:")
    print(f"  phase_demod数据类型: {phase_demod.dtype}")
    print(f"  phase_unwrapped数据类型: {phase_unwrapped.dtype}")
    print(f"  phase_filtered数据类型: {phase_filtered.dtype}")