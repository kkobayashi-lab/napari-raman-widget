@echo off
call C:\Users\spraman\anaconda3\Scripts\activate.bat confocal_control_gui_cellpose4
if errorlevel 1 exit /b %errorlevel%
cd /d C:\Users\spraman\confocal_control_gui\napari-raman-widget
python run_napari.py
exit /b %errorlevel%
