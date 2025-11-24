# 十一剑的CS_DN.博客 - 网格布局示例

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                             QGridLayout, QLabel)
import sys


class GridLayoutExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('十一剑的网格布局示例')

        # 创建网格布局
        grid = QGridLayout()
        self.setLayout(grid)

        # 创建按钮并添加到网格中
        names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        positions = [(i, j) for i in range(3) for j in range(3)]

        for position, name in zip(positions, names):
            button = QPushButton(name)
            grid.addWidget(button, *position)

        # 添加一个跨越多列的标签
        label = QLabel('十一剑 - 3x3网格布局')
        grid.addWidget(label, 3, 0, 1, 3)  # 第4行，跨越3列


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GridLayoutExample()
    ex.show()
    sys.exit(app.exec_())
