# 十一剑的CS_DN.博客 - 自定义信号示例

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys


# 创建一个包含信号的类
class Communicate(QObject):
    closeApp = pyqtSignal()  # 定义一个信号


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('十一剑的自定义信号示例')
        self.setGeometry(100, 100, 300, 200)

        # 创建通信对象
        self.c = Communicate()
        self.c.closeApp.connect(self.close)  # 连接信号到槽

        # 创建按钮
        btn = QPushButton('关闭窗口', self)
        btn.move(100, 80)
        btn.clicked.connect(self.emit_signal)  # 按钮点击触发信号

    def emit_signal(self):
        print('发射关闭信号 - 十一剑')
        self.c.closeApp.emit()  # 发射信号


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
