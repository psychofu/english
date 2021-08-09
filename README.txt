语音识别服务说明：

一、运行环境
       识别环境需要运行再python环境下，为保证服务正常运行，需要使用python3.7的实验环境。另外需要安装指定版本的依赖包（包文件已下载至packages目录，可以离线安装）。

二、环境配置
       防止此python3环境干扰到服务器中已有的python环境，考虑使用anaconda来进行python环境管理，下载使用可参考anaconda官网。安装好anaconda之后，使用“conda create -n tf1 python=3.7.10”创建python3环境。尔后执行“pip install --no-index --find-links=packages -r packages/requirements.txt”安装离线依赖包

三、启动服务
       在windows服务器上使用时，执行项目目录下的“asr服务启动.bat”启动服务。