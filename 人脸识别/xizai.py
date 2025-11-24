import os
import pip
import requests
from bs4 import BeautifulSoup


def install_dlib():
    # 目标页面：dlib 预编译包列表
    url = "https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib"
    try:
        # 获取页面内容
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找 Python 3.12 对应的 dlib 包（优先 19.24.2 版本）
        target_version = "cp312"
        dlib_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and "dlib-" in href and target_version in href and "win_amd64.whl" in href:
                dlib_links.append(href)

        # 筛选最新的 19.24.x 版本
        dlib_links.sort(reverse=True)  # 按版本号排序
        for link in dlib_links:
            if "19.24" in link:
                download_url = f"https://www.lfd.uci.edu/~gohlke/pythonlibs/{link}"
                print(f"找到适配包：{download_url}")

                # 下载并安装
                pip.main(['install', download_url])
                return

        # 如果没有 19.24 版本，用最新的 19.x 版本
        if dlib_links:
            download_url = f"https://www.lfd.uci.edu/~gohlke/pythonlibs/{dlib_links[0]}"
            print(f"使用最新兼容版本：{download_url}")
            pip.main(['install', download_url])
            return

        print("未找到适配 Python 3.12 的 dlib 预编译包")

    except Exception as e:
        print(f"自动安装失败：{str(e)}")
        print("请手动下载安装：https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib")


if __name__ == "__main__":
    install_dlib()