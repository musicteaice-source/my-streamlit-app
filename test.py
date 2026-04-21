import platform
import struct

print(f"Python版本: {platform.python_version()}")
print(f"系统架构: {platform.architecture()[0]}")
print(f"Python位数: {struct.calcsize('P') * 8}位")