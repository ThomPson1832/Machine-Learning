@echo off

REM 这是一个用于一键启动实时人脸识别与语音播报系统的批处理脚本
REM 功能：检查Python环境、安装依赖、运行系统测试、启动主程序、支持重启

REM 清屏并显示启动信息
cls
echo 正在启动实时人脸识别与语音播报系统...
echo ========================================

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请先安装Python 3.6或更高版本。
    pause
    exit /b 1
)

REM 检查并安装依赖包
REM 这一步确保系统运行所需的Python库已安装

echo 检查并安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 安装依赖包时出现错误，请手动安装。
)

REM 运行系统测试
REM 测试步骤确保系统组件正常工作

echo 正在运行系统测试...
python test_system.py
if %errorlevel% neq 0 (
    echo 警告: 系统测试未全部通过，可能影响使用。
)

REM 显示系统准备就绪信息并等待用户输入
REM 提示用户可以按q键退出系统
echo. 
echo 系统准备就绪，按任意键启动人脸识别系统...
echo 注意：运行时按 'q' 键可退出系统。
pause >nul

REM 定义启动循环标签
:start_loop

REM 启动主程序
REM 这是系统的核心人脸识别功能
python main.py

REM 检查是否需要重启
REM 允许用户选择是否重新启动系统
set /p restart="是否重启系统? (y/n): "
if /i "%restart%" equ "y" goto start_loop

REM 退出系统
REM 显示退出信息并等待用户确认
echo 系统已退出。
pause