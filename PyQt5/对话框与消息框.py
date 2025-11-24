# 十一剑的CS_DN.博客 - 消息框示例

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                             QVBoxLayout, QMessageBox)
import sys


class MessageBoxExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('十一剑的消息框示例')

        # 创建布局
        layout = QVBoxLayout()

        # 创建按钮
        btn_info = QPushButton('显示信息框', self)
        btn_info.clicked.connect(self.show_info)

        btn_warn = QPushButton('显示警告框', self)
        btn_warn.clicked.connect(self.show_warning)

        btn_question = QPushButton('显示问题框', self)
        btn_question.clicked.connect(self.show_question)

        # 将按钮添加到布局中
        layout.addWidget(btn_info)
        layout.addWidget(btn_warn)
        layout.addWidget(btn_question)

        # 设置窗口布局
        self.setLayout(layout)

    def show_info(self):
        QMessageBox.information(self, '信息',
                                '这是一个信息框 - 十一剑')

    def show_warning(self):
        QMessageBox.warning(self, '警告',
                            '这是一个警告框 - 十一剑')

    def show_question(self):
        reply = QMessageBox.question(self, '问题',
                                     '你确定要继续吗? - 十一剑',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            print('用户选择了"是"')
        else:
            print('用户选择了"否"')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MessageBoxExample()
    ex.show()
    sys.exit(app.exec_())
