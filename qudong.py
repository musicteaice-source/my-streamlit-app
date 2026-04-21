from ctypes import *
import os

# 指定DLL的完整路径
dll_path = r"D:\毕业论文\自制数据采集模块驱动及应用程序\数据采集模块驱动及应用程序\USBDAQlvapi.dll"

# 直接加载
try:
    daq_dll = cdll.LoadLibrary(dll_path)
    print("✅ DLL加载成功！")
    
    # 现在可以调用函数了
    # 例如：print(f"版本: {daq_dll.USBDAQ_GetVersion()}")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")