@echo off
echo [INFO] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environement virtuel active

echo [INFO] Lancement de l'application...
python cyberbill_SDXL.py

pause