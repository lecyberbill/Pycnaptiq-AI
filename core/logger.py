import logging
import os
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, init

# Initialiser colorama pour Windows
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Formateur personnalisé pour ajouter des couleurs en console."""
    
    COLORS = {
        logging.DEBUG: Fore.WHITE + Style.DIM,
        logging.INFO: Fore.CYAN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        # Format pour la console : [TIME] [LEVEL] [MODULE] Message
        timestamp = self.formatTime(record, self.datefmt)
        
        level_name = f"[{record.levelname}]"
        # On peut limiter la taille du nom du module pour plus de clarté
        module_name = f"[{record.name}]"
        
        message = f"{Fore.WHITE}{timestamp} {log_color}{level_name:<10} {Fore.WHITE}{module_name:<15} {record.getMessage()}"
        
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
            
        return message

def setup_logger(name="Pycnaptiq", log_dir="logs", log_level=logging.INFO):
    """Configure et retourne un logger avec rotation et couleurs."""
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # On capture tout à la source, les handlers filtreront
    
    # Éviter d'ajouter des handlers si le logger est déjà configuré
    if logger.hasHandlers():
        return logger

    # 1. Handler pour la Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    
    # 2. Handler pour le Fichier avec Rotation
    log_file = os.path.join(log_dir, "pycnaptiq.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5 * 1024 * 1024, # 5 Mo
        backupCount=5,           # Garde 5 fichiers de backup
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG) # Le fichier contient tout pour le debug
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Instance globale pour un usage facile
logger = setup_logger()
logger.info("Système de logging initialisé (Rotation: 5MB, Backup: 5)")
