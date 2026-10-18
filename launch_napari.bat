@echo off
setlocal
for %%I in ("%~dp0..") do set "WORKSPACE=%%~fI"
set "PYTHONPATH=%WORKSPACE%\napari-raman-widget;%WORKSPACE%\raman-control;%WORKSPACE%\raman-mda-engine;%WORKSPACE%\cns-control\src"
cd /d "%~dp0"
"C:\Users\spraman\anaconda3\python.exe" -m conda run --no-capture-output -n confocal_control_gui_cellpose4 python "%~dp0run_napari.py"
exit /b %errorlevel%
