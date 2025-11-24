# 十一剑的CS_DN.博客 - 文本框示例

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout)
import sys


class LineEditExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('十一剑的文本框示例')

        # 创建布局
        layout = QVBoxLayout()

        # 创建标签
        self.label = QLabel('输入内容将显示在这里')

        # 创建文本框
        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText('请输入文本...')

        # 创建按钮
        btn = QPushButton('显示文本', self)
        btn.clicked.connect(self.on_click)

        # 将控件添加到布局中
        layout.addWidget(self.label)
        layout.addWidget(self.textbox)
        layout.addWidget(btn)

        # 设置窗口布局
        self.setLayout(layout)

    def on_click(self):
        text = self.textbox.text()
        self.label.setText(f'你输入了: {text} - 十一剑')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LineEditExample()
    ex.show()
    sys.exit(app.exec_())
