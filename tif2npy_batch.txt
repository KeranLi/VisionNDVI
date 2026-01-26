@echo off
REM 设置 Python 环境路径（如果你使用的是 Anaconda 或其他虚拟环境，确保激活它）
REM 如果你的 Python 可执行文件已经在系统 PATH 中，可以跳过此行
set PATH=C:\path\to\python;%PATH%

REM 激活虚拟环境（如果有的话）
REM call C:\path\to\your\env\Scripts\activate.bat

REM 设置输入和输出目录
set INPUT_DIR=E:/scenarioMIP_output/future_resampled_this
set OUTPUT_DIR=E:/npy

REM 进入 Python 脚本所在的目录
cd /d F:/code/VisionNDVI

REM 遍历 INPUT_DIR 中的所有 .tif 文件并运行 Python 脚本
for /r %%f in (%INPUT_DIR%\*.tif) do (
    REM 运行 Python 脚本，传递当前 .tif 文件路径给 Python 脚本
    python tif2npy_batch.py.py "%%f" "%OUTPUT_DIR%"
)

REM 脚本完成后退出
pause
