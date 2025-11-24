# 十一剑的CS_DN.博客 - PyQt5第一个示例程序

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

# 创建应用对象，sys.argv是命令行参数列表
app = QApplication(sys.argv)

# 创建一个窗口
window = QWidget()
window.setWindowTitle('十一剑的第一个PyQt5程序')  # 设置窗口标题
window.setGeometry(100, 100, 280, 80)  # 设置窗口位置和大小(x,y,width,height)

# 创建一个标签控件
label = QLabel('欢迎来到十一剑的CS_DN.博客学习PyQt5!', parent=window)
label.move(60, 30)  # 移动标签位置

# 显示窗口
window.show()

# 进入应用的主事件循环，等待用户操作
sys.exit(app.exec_())
