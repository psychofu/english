@echo off

echo "启动运行环境..."

timeout /T 1 /NOBREAK

rem 根据路径启动环境
call E:\miniconda\Scripts\activate.bat E:\miniconda\envs\tf1

echo "运行服务..."

timeout /T 1 /NOBREAK

rem 执行程序
python HttpServer.py