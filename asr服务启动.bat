@echo off

echo "�������л���..."

timeout /T 1 /NOBREAK

rem ����·����������
call E:\miniconda\Scripts\activate.bat E:\miniconda\envs\tf1

echo "���з���..."

timeout /T 1 /NOBREAK

rem ִ�г���
python HttpServer.py