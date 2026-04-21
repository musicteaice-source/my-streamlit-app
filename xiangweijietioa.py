import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def demodulate_phase(reference_signal, detected_signal, fs, f_mod, lowpass_cutoff, filter_order=4):
    """
    使用正交解调（数字锁相放大）算法从两路信号中解调相位。
    
    参数:
        reference_signal: 参考信号数组（施加的调制信号）
        detected_signal: 探测信号数组（光电探测器输出）
        fs: 采样率 (Hz)
        f_mod: 调制频率 (Hz)
        lowpass_cutoff: 低通滤波器截止频率 (Hz)，应大于待测信号最高频率，远小于f_mod
        filter_order: 低通滤波器阶数
    
    返回:
        phase: 解调出的相位（弧度），已解卷绕
        I: 同相分量
        Q: 正交分量
    """
    n = len(reference_signal)
    t = np.arange(n) / fs  # 时间序列
    
    # 1. 预处理：去直流分量
    ref_ac = reference_signal - np.mean(reference_signal)
    det_ac = detected_signal - np.mean(detected_signal)
    
    # 2. 生成正交参考信号
    # 计算参考信号的相位（假设从t=0开始，无初始相位差）
    # 注意：如果参考信号有初始相位，需要先提取其相位信息
    phase_ref = np.unwrap(np.angle(signal.hilbert(ref_ac)))  # 通过希尔伯特变换提取相位
    # 或者直接生成：如果已知参考信号是纯净的sin(2πf_mod t)
    # 这里使用提取的相位更鲁棒
    I_ref = np.sin(phase_ref)  # 同相参考信号
    Q_ref = np.cos(phase_ref)  # 正交参考信号
    
    # 3. 正交解调：相乘
    I_mixed = det_ac * I_ref
    Q_mixed = det_ac * Q_ref
    
    # 4. 低通滤波
    # 设计低通滤波器
    nyquist = fs / 2
    low = lowpass_cutoff / nyquist
    b, a = butter(filter_order, low, btype='low')
    
    # 使用零相位滤波（filtfilt）避免相位失真
    I_filtered = filtfilt(b, a, I_mixed)
    Q_filtered = filtfilt(b, a, Q_mixed)
    
    # 5. 计算相位
    # 使用arctan2计算四象限相位，得到包裹在(-π, π]的相位
    phase_wrapped = np.arctan2(Q_filtered, I_filtered)
    
    # 6. 相位解卷绕
    phase_unwrapped = np.unwrap(phase_wrapped)
    
    return phase_unwrapped, I_filtered, Q_filtered, t

# 示例使用
if __name__ == "__main__":
    # 生成模拟数据
    fs = 1e6  # 采样率 1 MHz
    f_mod = 100e3  # 调制频率 100 kHz
    t_duration = 0.01  # 信号时长 10 ms
    n_samples = int(fs * t_duration)
    t = np.arange(n_samples) / fs
    
    # 模拟参考信号（施加在相位调制器上的信号）
    amp_ref = 1.0
    # 添加微小频偏和初始相位模拟实际情况
    f_ref = f_mod * 1.001  # 实际频率可能有微小偏差
    phase_ref_init = np.pi/6
    reference_signal = amp_ref * np.sin(2*np.pi*f_ref*t + phase_ref_init)
    
    # 模拟探测信号（光电探测器输出）
    # 包含：1) 直流偏置 2) 干涉信号 3) 噪声
    dc_offset = 2.0
    amp_det = 0.5
    # 模拟待测相位信号（低频，变化缓慢）
    phase_signal = 0.1 * np.sin(2*np.pi*500*t)  # 500 Hz的待测信号
    # 干涉信号：假设为余弦形式，相位包含调制和待测信号
    carrier = np.cos(2*np.pi*f_mod*t + phase_signal)
    noise = 0.05 * np.random.randn(n_samples)  # 高斯噪声
    detected_signal = dc_offset + amp_det * carrier + noise
    
    # 设置低通滤波器截止频率（应大于待测信号最高频率，远小于f_mod）
    f_signal_max = 1000  # 假设待测信号最高频率1 kHz
    lowpass_cutoff = 2 * f_signal_max  # 2 kHz截止
    
    # 解调
    phase, I, Q, t_demod = demodulate_phase(
        reference_signal, detected_signal, fs, f_mod, lowpass_cutoff
    )
    
    # 可视化结果
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # 1. 原始信号
    axes[0].plot(t[:2000]*1000, reference_signal[:2000], 'b-', label='参考信号', alpha=0.7)
    axes[0].plot(t[:2000]*1000, detected_signal[:2000], 'r-', label='探测信号', alpha=0.7)
    axes[0].set_xlabel('时间 (ms)')
    axes[0].set_ylabel('幅度 (V)')
    axes[0].set_title('原始信号（前2ms）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 正交分量
    axes[1].plot(t_demod*1000, I, 'g-', label='I分量', alpha=0.7)
    axes[1].plot(t_demod*1000, Q, 'm-', label='Q分量', alpha=0.7)
    axes[1].set_xlabel('时间 (ms)')
    axes[1].set_ylabel('幅度')
    axes[1].set_title('解调后的I/Q分量')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 解调出的相位
    axes[2].plot(t_demod*1000, phase, 'k-', linewidth=1.5)
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('相位 (rad)')
    axes[2].set_title('解调相位（已解卷绕）')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 与原始相位信号比较
    axes[3].plot(t_demod*1000, phase_signal, 'b-', label='原始相位信号', alpha=0.7)
    axes[3].plot(t_demod*1000, phase, 'r--', label='解调相位', alpha=0.7)
    axes[3].set_xlabel('时间 (ms)')
    axes[3].set_ylabel('相位 (rad)')
    axes[3].set_title('解调相位与原始相位对比')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 性能评估
    # 截去滤波器瞬态响应部分（通常为滤波器阶数的3倍）
    trans_len = 3 * 4  # 滤波器阶数4
    phase_eval = phase[trans_len:-trans_len]
    phase_signal_eval = phase_signal[trans_len:-trans_len]
    
    # 计算残差
    residual = phase_eval - phase_signal_eval
    print(f"相位残差统计:")
    print(f"  均值: {np.mean(residual):.6f} rad")
    print(f"  标准差: {np.std(residual):.6f} rad")
    print(f"  峰峰值: {np.ptp(residual):.6f} rad")