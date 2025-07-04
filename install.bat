@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: --- Configuration ---
set "PYTHON_DIR_NAME=python-3.10.11-embed-amd64"
set "VENV_DIR_NAME=venv"
set "REQUIREMENTS_FILE=requirements.txt"
set "MAIN_APP_SCRIPT=Pycnaptiq-AI.py"
set "locales_dir=%~dp0locales\"

:: --- D�but du Script ---
goto :main

::-------------------------------------------------
:: Fonction: load_language
:: Charge les cha�nes de caract�res depuis un fichier de langue.
:: Param�tre %1: Nom du fichier de langue (ex: install_fr.txt)
::-------------------------------------------------
:load_language
set "lang_file=%~1"
echo Chargement du fichier de langue:/loading language file... %locales_dir%%lang_file%
if not exist "%locales_dir%%lang_file%" (
    echo [ERREUR] Fichier de langue '%lang_file%' introuvable dans '%locales_dir%' !
    echo [ERROR] Language file '%lang_file%' not found in '%locales_dir%' !
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%a in ("%locales_dir%%lang_file%") do (
    set "%%a=%%b"
)
if not defined INFO_WELCOME (
    echo [ERREUR] Le fichier de langue '%lang_file%' semble invalide ou vide.
    echo [ERROR] Language file '%lang_file%' seems invalid or empty.
    exit /b 1
)
goto :eof

::-------------------------------------------------
:: Fonction: check_error
:: V�rifie le code d'erreur pr�c�dent et affiche un message si > 0.
:: Param�tre %1: Message d'erreur � afficher (cl� de langue)
::-------------------------------------------------
:check_error
if errorlevel 1 (
    echo !%1!
    pause
    exit /b 1
)
goto :eof

::-------------------------------------------------
:: Flux Principal
::-------------------------------------------------
:main
:: D�finir le dossier courant du script d'installation
:: %~dp0 se termine par un backslash, pas besoin d'en ajouter un.
set "script_dir=%~dp0"

:: --- Choix de la langue ---
echo Choisissez votre langue / Choose your language:
echo 1. Francais
echo 2. English
choice /c 12 /n /m "Votre choix / Your choice ? "
if errorlevel 2 (
    call :load_language "install_en.txt"
) else (
    call :load_language "install_fr.txt"
)

:: Active la bonne page de code pour l'UTF-8 (pour les messages)
chcp 65001 > nul
echo !INFO_WELCOME!

:: --- D�finition des chemins absolus ---
set "python_embed_dir=%script_dir%%PYTHON_DIR_NAME%"
set "python_exe=%python_embed_dir%\python.exe"
set "venv_dir=%script_dir%%VENV_DIR_NAME%"
set "venv_python_exe=%venv_dir%\Scripts\python.exe"
set "venv_pip_exe=%venv_dir%\Scripts\pip.exe"
set "req_file_path=%script_dir%%REQUIREMENTS_FILE%"
set "start_script_path=%script_dir%start.bat"
set "main_app_path=%script_dir%%MAIN_APP_SCRIPT%"

:: --- V�rifications Pr�liminaires ---
echo !INFO_CHECK_PYTHON_EMBED!
if not exist "%python_exe%" (
    echo !ERROR_PYTHON_NOT_FOUND! "%python_exe%"
    pause
    exit /b 1
)
echo !OK_PYTHON_FOUND! "%python_exe%"

echo !INFO_CHECK_REQUIREMENTS!
if not exist "%req_file_path%" (
    echo !ERROR_REQUIREMENTS_NOT_FOUND! "%req_file_path%"
    pause
    exit /b 1
)
echo !OK_REQUIREMENTS_FOUND! "%req_file_path%"

:: --- V�rification CUDA ---
echo !INFO_VERIFY_CUDA!
nvcc --version > nul 2>&1
if errorlevel 9009 (
    echo !WARN_NVCC_NOT_FOUND!
    echo !INFO_CUDA_DOWNLOAD! https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
) else (
    nvcc --version 2>NUL | findstr /R /C:"release 12\.[0-9][0-9]*" >nul
    if errorlevel 1 (
        echo !ERROR_CUDA_VERSION!
        nvcc --version
        pause
        exit /b 1
    ) else (
        echo !OK_CUDA!
    )
)

:: --- Cr�ation de l'Environnement Virtuel ---
echo !INFO_CREATE_VENV! "%venv_dir%"
if exist "%venv_dir%" (
    echo !WARN_VENV_EXISTS!
    set /p "overwrite_venv=!PROMPT_OVERWRITE_VENV! (O/N): "
    if /i not "!overwrite_venv!"=="O" (
        echo !INFO_VENV_SKIP_CREATE!
        goto :skip_venv_creation
    )
    echo !INFO_DELETING_VENV!
    rd /s /q "%venv_dir%"
    call :check_error ERROR_DELETE_VENV
)
"%python_exe%" -m virtualenv "%venv_dir%"
call :check_error ERROR_CREATE_VENV
echo !OK_CREATE_VENV!
:skip_venv_creation

:: --- Installation des D�pendances ---
echo !INFO_INSTALL_DEP!

echo !INFO_UPGRADE_PIP!
"%venv_python_exe%" -m pip install --upgrade pip
call :check_error ERROR_UPGRADE_PIP
echo !OK_UPGRADE_PIP!

echo !INFO_INSTALL_TORCH!
"%venv_pip_exe%" install --no-cache-dir -U torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
call :check_error ERROR_INSTALL_TORCH
echo !OK_INSTALL_TORCH!

echo !INFO_INSTALL_XFORMER!
"%venv_pip_exe%" install --no-cache-dir -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
call :check_error ERROR_INSTALL_XFORMER
echo !OK_INSTALL_XFORMER!


echo !INFO_INSTALL_REQUIREMENTS! "%REQUIREMENTS_FILE%"
"%venv_pip_exe%" install --no-cache-dir -r "%req_file_path%"
call :check_error ERROR_INSTALL_REQUIREMENTS
echo !OK_INSTALL_REQUIREMENTS!

:: --- Cr�ation du script de d�marrage start.bat ---
echo !INFO_CREATE_START_SCRIPT! "%start_script_path%"
(
    echo @echo off
    echo SETLOCAL
    echo :: Script de demarrage genere par install.bat
    echo :: Ne pas modifier manuellement, relancer install.bat si besoin.
    echo.
    echo title %MAIN_APP_SCRIPT% Launcher
    echo.
    echo :: Definit le repertoire du script de demarrage
    echo set "start_script_dir=%%~dp0"
    echo.
    echo :: Definit le chemin absolu de l'environnement virtuel
    echo set "venv_dir=%%start_script_dir%%%VENV_DIR_NAME%%"
    echo set "main_script=%%start_script_dir%%%MAIN_APP_SCRIPT%%"
    echo.
    echo :: Verifie si le venv existe
    echo if not exist "%%venv_dir%%\Scripts\activate.bat" (
    echo    echo [ERREUR] Environnement virtuel introuvable a '%%venv_dir%%'.
    echo    echo         Veuillez relancer install.bat.
    echo    echo [ERROR] Virtual environment not found at '%%venv_dir%%'.
    echo    echo         Please re-run install.bat.
    echo    pause
    echo    exit /b 1
    echo ^)
    echo.
    echo :: Active l'environnement virtuel
    echo echo Activation de l'environnement virtuel...
    echo call "%%venv_dir%%\Scripts\activate.bat"
    echo if errorlevel 1 (
    echo     echo [ERREUR] Impossible d'activer l'environnement virtuel.
    echo     echo [ERROR] Failed to activate the virtual environment.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo echo Environnement virtuel active.
    echo.
    echo :: Verifie si le script principal existe
    echo if not exist "%%main_script%%" (
    echo    echo [ERREUR] Script principal '%%main_script%%' introuvable.
    echo    echo [ERROR] Main script '%%main_script%%' not found.
    echo    pause
    echo    exit /b 1
    echo ^)
    echo.
    echo :: Lance l'application principale
    echo echo Lancement de l'application: %%main_script%% ...
    echo python "%%main_script%%" %%*
    echo.
    echo echo L'application s'est terminee. Appuyez sur une touche pour fermer cette fenetre.
    echo pause ^> nul
    echo ENDLOCAL
) > "%start_script_path%"

:: Ajout d'une v�rification explicite pour �tre s�r
if not exist "%start_script_path%" (
    echo [ERREUR CRITIQUE] Le fichier start.bat n'a pas pu etre cree. Verifiez les permissions d'ecriture ou des caracteres speciaux dans le script.
    echo [CRITICAL ERROR] The start.bat file could not be created. Check write permissions or special characters in the script.
    pause
    exit /b 1
)

call :check_error ERROR_CREATE_START_SCRIPT
echo !OK_CREATE_START_SCRIPT!


echo.
echo !INFO_INSTALL_DONE!
echo !INFO_HOW_TO_START! "%start_script_path%"
echo.
pause
exit /b 0