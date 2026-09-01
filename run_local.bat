@echo off
setlocal
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env >nul
set DATA_MODE=synthetic
set DB_PATH=data\paper_trader_v030.db
set DASHBOARD_TOKEN=demo
echo Dashboard: http://localhost:8000/?token=demo
python -m app
