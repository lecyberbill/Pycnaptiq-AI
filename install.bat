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
echo !INFO_INSTALL_PYTHON!
set "url=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "zipfile=%cd%\python-3.10.11-embed-amd64.zip"
set "extractdir=%cd%\python-3.10.11-embed-amd64"
set "get_pip_url=https://bootstrap.pypa.io/get-pip.py"
set "get_pip_file=%cd%\get-pip.py"
set "PYTHON_SCRIPTS=%extractdir%\Scripts"
set "PYTHON_PATH=%extractdir%"

setx PATH "%PYTHON_SCRIPTS%;%extractdir%"
set PATH "%PYTHON_SCRIPTS%;%extractdir%;%PATH%"

echo !INFO_PYTHON_ADDED_PATH!
echo !INFO_DOWNLOAD_PYTHON!
bitsadmin /transfer "PythonDownloadJob" %url% "%zipfile%"

if errorlevel 1 (
  echo !ERROR_DOWNLOAD!
  pause
  exit /b 1
)

echo !INFO_DOWNLOAD_DONE!

echo !INFO_CREATE_DIR!
mkdir "%extractdir%"

echo !INFO_UNZIP!
powershell -command "Expand-Archive -Path '%zipfile%' -DestinationPath '%extractdir%' -Force"

if errorlevel 1 (
  echo !ERROR_UNZIP!
  pause
  exit /b 1
)

echo !OK_UNZIP!

:: Mise à jour du PATH dans l'environnement courant
set PATH "%PYTHON_PATH%\Scripts;%PYTHON_PATH%;%PATH%"

echo !INFO_UNCOMMENT_PTH!
powershell -Command "(Get-Content '%extractdir%\python310._pth') -replace '^#(import site)', 'import site' | Set-Content '%extractdir%\python310._pth'"
if errorlevel 1 (
    echo !ERROR_PTH!
    pause
    exit /b 1
)
echo !OK_PTH!

echo !INFO_DOWNLOAD_GETPIP!
bitsadmin /transfer "GetPipDownload" %get_pip_url% "%get_pip_file%"

if errorlevel 1 (
  echo !ERROR_GETPIP!
  pause
  exit /b 1
)

echo !OK_GETPIP!

echo !INFO_INSTALL_PIP!
"%extractdir%\python.exe" "%get_pip_file%"

if errorlevel 1 (
  echo !ERROR_INSTALL_PIP!
  pause
  exit /b 1
)

echo !OK_INSTALL_PIP!

echo !INFO_INSTALL_VENV!
"%extractdir%\python.exe" -m pip install --no-cache-dir virtualenv

if errorlevel 1 (
  echo !ERROR_INSTALL_VENV!
  pause
  exit /b 1
)

echo !INFO_INSTALL_VENV_DONE!
echo !INFO_DELETE_ZIP!
del %zipfile%
del "%get_pip_file%"

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
"%extractdir%\python.exe" -m virtualenv venv
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