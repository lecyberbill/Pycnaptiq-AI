@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo [INFO] Installation de python 3.10.11 Amd64 embeddable
set "url=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "zipfile=python-3.10.11-embed-amd64.zip"
set "extractdir=python-3.10.11-embed-amd64"
set "PYTHON_PATH=%extractdir%"

:: Ajouter Python et Scripts au PATH de l'utilisateur
setx PATH "%PYTHON_PATH%\Scripts;%PYTHON_PATH%;%PATH%"




echo [INFO] Python et les scripts ajoutés au PATH.

@echo off 
chcp 65001 > nul


echo [INFO] Installation de python 3.10.11 Amd64 embeddable
set "url=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "zipfile=%cd%\python-3.10.11-embed-amd64.zip"
set "extractdir=%cd%\python-3.10.11-embed-amd64"
set "get_pip_url=https://bootstrap.pypa.io/get-pip.py"
set "get_pip_file=%cd%\get-pip.py"
set "PYTHON_SCRIPTS=%extractdir%\Scripts"


setx PATH "%PYTHON_SCRIPTS%;%extractdir%"
set PATH "%PYTHON_SCRIPTS%;%extractdir%;%PATH%"


echo [INFO] Téléchargement de l'archive Python...
bitsadmin /transfer "PythonDownloadJob" %url% "%zipfile%"

if errorlevel 1 (
  echo [Erreur] Erreur lors du téléchargement.
  pause
  exit /b 1
)

echo [INFO] Téléchargement terminé.

echo [INFO] Création du répertoire d'extraction...
mkdir "%extractdir%"

echo [INFO] Décompression de l'archive...
powershell -command "Expand-Archive -Path '%zipfile%' -DestinationPath '%extractdir%' -Force"

if errorlevel 1 (
  echo [Erreur] Erreur lors de la decompression.
  pause
  exit /b 1
)

echo [OK] Décompression terminée.

:: Mise à jour du PATH dans l'environnement courant
set PATH "%PYTHON_PATH%\Scripts;%PYTHON_PATH%;%PATH%"

echo [INFO] Décommentage de "import site" dans python310._pth...
powershell -Command "(Get-Content '%extractdir%\python310._pth') -replace '^#(import site)', 'import site' | Set-Content '%extractdir%\python310._pth'"
if errorlevel 1 (
    echo [ERREUR] La modification de python310._pth a échoué.
    pause
    exit /b 1
)
echo [OK] Modification terminée.




echo [INFO] Téléchargement de get-pip.py...
bitsadmin /transfer "GetPipDownload" %get_pip_url% "%get_pip_file%"

if errorlevel 1 (
  echo Erreur lors du téléchargement de get-pip.py.
  pause
  exit /b 1
)

echo [OK] Téléchargement de get-pip.py terminé.

echo [INFO] Installation de pip...

"%extractdir%\python.exe" "%get_pip_file%"

if errorlevel 1 (
  echo Erreur lors de l'installation de pip.
  pause
  exit /b 1
)

echo [OK] Installation de pip terminée.


echo [INFO] Installation de virtualenv...

"%extractdir%\python.exe" -m pip install --no-cache-dir virtualenv

if errorlevel 1 (
  echo [Erreur] lors de l'installation de virtualenv.
  pause
  exit /b 1
)

echo [INFO] Installation de virtualenv terminée.

echo [INFO] Suppression de l'archive ZIP...
del %zipfile%
del "%get_pip_file%"



:: Vérification de CUDA...
echo [INFO] Vérification de CUDA...
nvcc --version 2>NUL | findstr /C:"release 11.8" >nul
if errorlevel 1 (
    echo [/!\ Erreur] CUDA 11.8 non detecte. Veuillez installer CUDA 11.8 et ses drivers.
    exit /b 1
) else (
    echo [OK] CUDA 11.8 detecte.
)

echo [INFO] Création de l'environnement virtuel...
"%extractdir%\python.exe" -m virtualenv venv
if errorlevel 1 (
    echo [ERREUR] Impossible de créer l'environnement virtuel.
    pause
    exit /b 1
)
echo [OK] Environnement virtuel créé.

echo [INFO] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel cree


echo [INFO] Installation des dépendances...
python -m pip install --upgrade pip
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir -r requirements.txt

echo [INFO] Installation terminée ! lancer le programme avec start.bat
pause
exit
