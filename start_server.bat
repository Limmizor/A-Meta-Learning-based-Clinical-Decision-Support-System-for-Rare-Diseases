@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  肺影智诊 - 启动服务
echo ============================================
echo 正在停止端口 5000 上的旧实例...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":5000" ^| findstr /c:"LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul
echo 正在启动服务，访问地址: http://127.0.0.1:5000
start "PF-Server" /min cmd /c "python app.py > server_out.log 2> server_err.log"

timeout /t 5 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo 服务已启动。停止服务请关闭名为 PF-Server 的最小化窗口。
pause
