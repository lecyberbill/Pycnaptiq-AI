@echo off
echo [INFO] Activation de l'environnement virtuel... (Activating the virtual environment...)
call venv\Scripts\activate.bat
echo [OK] Environement virtuel active (Virtual environment activated)

echo [INFO] Lancement de l'application... (Launching the application...)
python cyberbill_SDXL.py

pause