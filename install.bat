@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Aller directement au flux principal
goto :main

:: Définir les fonctions ici
:load_language
set "lang_file=%1"
echo Test: Trying to load %locales_dir%%lang_file%
if not exist "%locales_dir%%lang_file%" (
    echo [ERREUR] Fichier de langue %lang_file% manquant !
    exit /b 1
)
for /f "tokens=1,2 delims==" %%a in ('type "%locales_dir%%lang_file%"') do (
    set "%%a=%%b"
)
goto :eof


:main

:: Définir le dossier courant
set "script_dir=%~dp0"
set "locales_dir=%script_dir%locales\"

:: Choix de la langue
echo Choisissez votre langue (Choose your language):
echo 1. Francais
echo 2. English
choice /c 12 /n /m "Votre choix (Your choice) ? "
if errorlevel 2 (
    call :load_language "install_en.txt"
) else (
    call :load_language "install_fr.txt"
)


chcp 65001 > nul
set "pythonDir=%cd%\python-3.10.11-embed-amd64"
set "PYTHON_SCRIPTS=%pythonDir%\Scripts"
set "PYTHON_PATH=%pythonDir%"

setx PATH "%PYTHON_SCRIPTS%;%pythonDir%"
set PATH "%PYTHON_SCRIPTS%;%pythonDir%;%PATH%"
:: Mise à jour du PATH dans l'environnement courant
set PATH "%PYTHON_PATH%\Scripts;%PYTHON_PATH%;%PATH%"
echo !INFO_PYTHON_ADDED_PATH!


echo !INFO_VERIFY_CUDA!
nvcc --version 2>NUL | findstr /C:"release 12.6" >nul
if errorlevel 1 (
    echo !ERROR_CUDA!
	pause
    exit /b 1
) else (
    echo !OK_CUDA!
)

echo !INFO_CREATE_VENV!
"%pythonDir%\python.exe" -m virtualenv venv
if errorlevel 1 (
    echo !ERROR_CREATE_VENV!
    pause
    exit /b 1
)
echo !OK_CREATE_VENV!

echo !INFO_ACTIVATE_VENV!
call venv\Scripts\activate.bat
echo !OK_ACTIVATE_VENV!

echo !INFO_INSTALL_DEP!
python -m pip install --upgrade pip
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir -r requirements.txt

echo !INFO_INSTALL_DONE!
pause
exit