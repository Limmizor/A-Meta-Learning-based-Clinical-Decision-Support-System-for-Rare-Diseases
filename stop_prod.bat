@echo off
chcp 65001 >nul
echo 正在停止端口 5000 上的服务...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":5000" ^| findstr /c:"LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo 已停止。
pause
