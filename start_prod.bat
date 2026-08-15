@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  肺影智诊 - 生产模式启动（waitress）
echo ============================================
echo 正在停止端口 5000 上的旧实例...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":5000" ^| findstr /c:"LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul
echo 正在启动服务（监听 0.0.0.0:5000）...
start "PF-Server-Prod" /min cmd /c "python -m waitress --listen=0.0.0.0:5000 --threads=8 app:app > prod_out.log 2> prod_err.log"

timeout /t 5 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo 服务已启动。
echo 本机访问:   http://127.0.0.1:5000
echo 局域网访问: http://本机IP:5000 （需放行防火墙 TCP 5000）
echo 停止服务:   关闭名为 PF-Server-Prod 的最小化窗口，或运行 stop_prod.bat
pause
