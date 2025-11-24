# 十一剑的CS_DN.博客 - PyQt5按钮示例

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import sys


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('十一剑的按钮示例')
        self.setGeometry(100, 100, 300, 200)

        # 创建一个垂直布局
        layout = QVBoxLayout()

        # 创建按钮1
        btn1 = QPushButton('按钮1', self)
        btn1.clicked.connect(self.on_btn1_click)  # 连接点击信号到槽函数

        # 创建按钮2
        btn2 = QPushButton('按钮2', self)
        btn2.clicked.connect(self.on_btn2_click)

        # 将按钮添加到布局中
        layout.addWidget(btn1)
        layout.addWidget(btn2)

        # 设置窗口的布局
        self.setLayout(layout)

    def on_btn1_click(self):
        print('按钮1被点击了 - 十一剑的CS_DN.博客')

    def on_btn2_click(self):
        print('按钮2被点击了 - 十一剑的CS_DN.博客')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
