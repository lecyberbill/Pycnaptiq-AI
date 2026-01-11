@echo off
SETLOCAL
:: Script de demarrage genere par install.bat
:: Ne pas modifier manuellement, relancer install.bat si besoin.

title Pycnaptiq-AI.py Launcher

:: Definit le repertoire du script de demarrage
set "start_script_dir=%~dp0"

:: Definit le chemin absolu de l'environnement virtuel
set "venv_dir=%start_script_dir%venv"
set "main_script=%start_script_dir%Pycnaptiq-AI.py"

:: Verifie si le venv existe
if not exist "%venv_dir%\Scripts\activate.bat" (
   echo [ERREUR] Environnement virtuel introuvable a '%venv_dir%'.
   echo         Veuillez relancer install.bat.
   echo [ERROR] Virtual environment not found at '%venv_dir%'.
   echo         Please re-run install.bat.
   pause
   exit /b 1
)

:: Active l'environnement virtuel
echo Activation de l'environnement virtuel...
call "%venv_dir%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel.
    echo [ERROR] Failed to activate the virtual environment.
    pause
    exit /b 1
)
echo Environnement virtuel active.

:: Verifie si le script principal existe
if not exist "%main_script%" (
   echo [ERREUR] Script principal '%main_script%' introuvable.
   echo [ERROR] Main script '%main_script%' not found.
   pause
   exit /b 1
)

:: Lance l'application principale
echo Lancement de l'application: %main_script% ...
python "%main_script%" %*

echo L'application s'est terminee. Appuyez sur une touche pour fermer cette fenetre.
pause > nul
ENDLOCAL
