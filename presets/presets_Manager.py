# presets/presets_Manager.py

import sqlite3
import os
import json
from datetime import datetime
from io import BytesIO
from PIL import Image
import traceback
import shutil
from contextlib import contextmanager # Ajout pour le gestionnaire de contexte

# Import direct depuis Utils (sans try/except comme vous l'avez suggéré)
from Utils.utils import txt_color, translate

# Chemin vers la base de données
DB_FILE_PATH = os.path.join(os.path.dirname(__file__), 'presets.db')
MODULE_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'module_config.json')

class PresetManager:
    """Gère les opérations CRUD pour les presets dans la base de données SQLite."""

    def __init__(self, translations):
        """
        Initialise le gestionnaire de presets.

        Args:
            translations (dict): Le dictionnaire de traductions chargé.
        """
        self.translations = translations
        self.db_path = DB_FILE_PATH
        self.module_config_path = MODULE_CONFIG_FILE_PATH
        self._init_preset_db() # Initialiser la DB lors de la création de l'instance
        self.loaded_module_states = self._load_module_states() # Charger l'état des modules

    def _get_db_connection(self):
        """Établit et retourne une connexion à la base de données SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            # Utilisation unifiée
            log_message = translate("erreur_db_connexion_log", self.translations).format(error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
            return None

    @contextmanager
    def _managed_db_connection(self):
        """Fournit une connexion DB gérée avec fermeture automatique."""
        conn = self._get_db_connection()
        if conn is None:
           
            yield None # Permet de vérifier la connexion dans le bloc 'with'
            return
        try:
            yield conn
        finally:
            if conn:
                conn.close()

    def _init_preset_db(self):
        """
        Initialise la base de données et crée la table 'presets' si elle n'existe pas.
        Méthode interne appelée par __init__.
        """
        with self._managed_db_connection() as conn:
            if conn is None:
                print(txt_color("[ERREUR DB] ", "erreur"), translate("erreur_init_db_connexion", self.translations))
                return

            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS presets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        preset_type TEXT DEFAULT 'gen_image' NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        rating INTEGER DEFAULT 0,
                        model TEXT, vae TEXT, original_prompt TEXT, prompt TEXT, negative_prompt TEXT,
                        styles TEXT, custom_pipeline_id TEXT,
                        guidance_scale REAL, num_steps INTEGER, sampler_key TEXT, seed INTEGER,
                        width INTEGER, height INTEGER, loras TEXT, preview_image BLOB, notes TEXT
                    );
                """)
                conn.commit()
                print(txt_color("[INFO DB] ", "ok"), translate("db_init_success", self.translations))
                cursor.execute("PRAGMA table_info(presets)")
                columns = [column['name'] for column in cursor.fetchall()]
                if 'original_prompt' not in columns:
                    print(txt_color("[INFO DB] ", "info"), translate("db_adding_original_prompt_column", self.translations))
                    cursor.execute("ALTER TABLE presets ADD COLUMN original_prompt TEXT")
                    conn.commit()
                    print(txt_color("[INFO DB] ", "ok"), translate("db_original_prompt_column_added", self.translations))
                if 'custom_pipeline_id' not in columns:
                    # Utiliser une clé de traduction dédiée pour custom_pipeline_id
                    print(txt_color("[INFO DB] ", "info"), translate("db_adding_custom_pipeline_id_column", self.translations))
                    cursor.execute("ALTER TABLE presets ADD COLUMN custom_pipeline_id TEXT")
                    conn.commit()
                    print(txt_color("[INFO DB] ", "ok"), translate("db_custom_pipeline_id_column_added", self.translations)) # Nouvelle clé
            except sqlite3.Error as e:
                log_message = translate("erreur_init_table", self.translations).format(error=e)
                print(txt_color("[ERREUR DB] ", "erreur"), log_message)
                traceback.print_exc()

    def save_gen_image_preset(self, preset_name, gen_data, preview_image_pil):
        """
        Sauvegarde un preset 'gen_image'. Retourne (bool, message).
        Utilise self.translations.
        """
        # Utilise self.translations pour les messages
        msg_err_nom_vide = translate("erreur_nom_preset_vide", self.translations)
        msg_err_donnees = translate("erreur_pas_donnees_generation", self.translations)
        msg_err_connexion = translate("erreur_db_connexion", self.translations)
        msg_success = translate("preset_sauvegarde", self.translations).format(preset_name)
        msg_err_db = translate("erreur_db_sauvegarde", self.translations)
        msg_err_general = translate("erreur_sauvegarde_preset", self.translations)
        msg_err_seed_overflow = translate("erreur_seed_trop_grand", self.translations) # Ajouter clé traduction

        if not preset_name: return False, msg_err_nom_vide
        if not gen_data or not preview_image_pil: return False, msg_err_donnees

        with self._managed_db_connection() as conn:
            if conn is None:
                return False, msg_err_connexion

            try:
                # Préparation de l'image preview
                preview_image_pil.thumbnail((256, 256))
                buffer = BytesIO()
                try: preview_image_pil.save(buffer, format="WEBP", quality=85)
                except Exception: preview_image_pil.save(buffer, format="PNG")
                image_blob = sqlite3.Binary(buffer.getvalue())

                # Préparation des données
                db_data = gen_data.copy()
                db_data.update({
                    'name': preset_name, 'preview_image': image_blob,
                    'preset_type': 'gen_image', 'last_used': datetime.now().isoformat()
                })

                # Sérialisation JSON pour styles et loras
                if 'styles' in db_data and isinstance(db_data['styles'], list): db_data['styles'] = json.dumps(db_data['styles'])
                if 'loras' in db_data and isinstance(db_data['loras'], list): db_data['loras'] = json.dumps(db_data['loras'])

                # --- CONVERSION SEED EN TEXTE ---
                if 'seed' in db_data:
                    db_data['seed'] = str(db_data['seed']) # Convertir en string

                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(presets)")
                valid_columns = {row['name'] for row in cursor.fetchall()}

                # Filtrer les données pour ne garder que les colonnes valides
                filtered_data = {k: v for k, v in db_data.items() if k in valid_columns}

                expected_columns = [
                    'name', 'preset_type', 'created_at', 'last_used', 'rating',
                    'model', 'vae', 'original_prompt', 'prompt', 'negative_prompt',
                    'styles', 'custom_pipeline_id', 'guidance_scale', 'num_steps', 'sampler_key', 'seed',
                    'width', 'height', 'loras', 'preview_image', 'notes'
                ]
                if 'created_at' not in filtered_data:
                     filtered_data['created_at'] = datetime.now().isoformat()

                values_to_insert = []
                for col in expected_columns:
                     values_to_insert.append(filtered_data.get(col))

                columns_str = ', '.join(expected_columns)
                placeholders = ', '.join('?' * len(expected_columns))
                sql = f"INSERT OR REPLACE INTO presets ({columns_str}) VALUES ({placeholders})"

                cursor.execute(sql, values_to_insert)
                conn.commit()

                log_message = translate("log_preset_saved", self.translations).format(name=preset_name)
                print(txt_color("[INFO DB] ", "ok"), log_message)
                return True, msg_success

            except sqlite3.IntegrityError as integrity_err:
                log_message = translate("log_warn_rename_name_exists", self.translations).format(name=preset_name)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_message)
                return False, translate("erreur_nom_preset_existe_rename", self.translations).format(preset_name)
            except sqlite3.Error as db_err:
                log_message = translate("log_erreur_db_save", self.translations).format(name=preset_name, error=db_err)
                print(txt_color("[ERREUR DB] ", "erreur"), log_message)
                traceback.print_exc()
                if "too large" in str(db_err):
                    return False, msg_err_seed_overflow
                return False, msg_err_db
            except Exception as e:
                log_message = translate("log_erreur_general_save", self.translations).format(name=preset_name, error=e)
                print(txt_color("[ERREUR] ", "erreur"), log_message)
                traceback.print_exc()
                return False, msg_err_general


# Dans la classe PresetManager

    # --- MODIFIER LA SIGNATURE ---
    def load_presets_for_display(self, preset_type='gen_image', search_term="", sort_by="Nom A-Z",
                                 selected_models=None, selected_samplers=None, selected_loras=None):
        """
        Charge les presets pour affichage, en appliquant les filtres.
        """
        presets_data = []
        with self._managed_db_connection() as conn:
            if conn is None:
                return []

            try:
                cursor = conn.cursor()
                query = """
                    SELECT id, name, rating, preview_image, created_at, last_used, notes,
                           model, sampler_key, loras
                    FROM presets
                    WHERE preset_type = ?
                """
                params = [preset_type]

                if search_term:
                    query += " AND name LIKE ?"
                    params.append(f"%{search_term}%")

                if selected_models and isinstance(selected_models, list):
                    placeholders = ','.join('?' * len(selected_models))
                    query += f" AND model IN ({placeholders})"
                    params.extend(selected_models)

                if selected_samplers and isinstance(selected_samplers, list):
                    placeholders = ','.join('?' * len(selected_samplers))
                    query += f" AND sampler_key IN ({placeholders})"
                    params.extend(selected_samplers)

                order_clause = " ORDER BY "
                if sort_by == "Nom A-Z": order_clause += "name ASC"
                elif sort_by == "Nom Z-A": order_clause += "name DESC"
                elif sort_by == "Date Création": order_clause += "created_at DESC"
                elif sort_by == "Date Utilisation": order_clause += "last_used DESC NULLS LAST"
                elif sort_by == "Note": order_clause += "rating DESC"
                else: order_clause += "name ASC"
                query += order_clause

                cursor.execute(query, params)
                results_sql = cursor.fetchall()

                if selected_loras and isinstance(selected_loras, list):
                    filtered_presets_data = []
                    for row in results_sql:
                        loras_str = row['loras']
                        preset_loras_names = set()
                        if loras_str:
                            try:
                                loras_list_data = json.loads(loras_str)
                                if isinstance(loras_list_data, list):
                                    for lora_item in loras_list_data:
                                        if isinstance(lora_item, dict) and 'name' in lora_item and lora_item['name']:
                                            preset_loras_names.add(lora_item['name'])
                            except (json.JSONDecodeError, TypeError):
                                pass 
                        if set(selected_loras).issubset(preset_loras_names):
                            filtered_presets_data.append(row)
                    presets_data = filtered_presets_data
                else:
                    presets_data = results_sql

                log_message = translate("log_presets_loaded_display_filtered", self.translations).format(
                    count=len(presets_data),
                    type=preset_type,
                    search=search_term or "N/A",
                    models=selected_models or "Tous",
                    samplers=selected_samplers or "Tous",
                    loras=selected_loras or "Tous"
                )
                print(txt_color("[INFO DB]", "info"), log_message)

            except sqlite3.Error as e:
                log_message = translate("log_erreur_load_display", self.translations).format(type=preset_type, error=e)
                print(txt_color("[ERREUR DB]", "erreur"), log_message)
                traceback.print_exc()
        return presets_data # Retourne la liste (potentiellement filtrée par Python)

    def load_preset_data(self, preset_id):
        """
        Charge les données complètes d'un preset. Inclut maintenant 'original_prompt'.
        Utilise self.translations.
        """
        preset_dict = None
        with self._managed_db_connection() as conn:
            if conn is None:
                return None

            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM presets WHERE id = ?", (preset_id,))
                preset_data = cursor.fetchone()

                if preset_data:
                    preset_dict = dict(preset_data)
                    log_message = translate("log_preset_data_loading", self.translations).format(id=preset_id, name=preset_dict.get('name', ''))
                    print(txt_color("[INFO DB] ", "info"), log_message)
                    try:
                        # Utiliser la même connexion pour l'UPDATE
                        conn.execute("UPDATE presets SET last_used = ? WHERE id = ?", (datetime.now().isoformat(), preset_id))
                        conn.commit()
                    except sqlite3.Error as update_err:
                        log_warn_message = translate("log_warn_update_last_used", self.translations).format(id=preset_id, error=update_err)
                        print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)

                    try:
                        styles_str = preset_dict.get('styles')
                        preset_dict['styles'] = json.loads(styles_str) if styles_str else []
                    except (json.JSONDecodeError, TypeError) as e:
                        log_warn_message = translate("log_warn_deserialize_styles", self.translations).format(id=preset_id, error=e)
                        print(txt_color("[AVERTISSEMENT] ", "erreur"), log_warn_message)
                        preset_dict['styles'] = []
                    try:
                        loras_str = preset_dict.get('loras')
                        preset_dict['loras'] = json.loads(loras_str) if loras_str else []
                    except (json.JSONDecodeError, TypeError) as e:
                        log_warn_message = translate("log_warn_deserialize_loras", self.translations).format(id=preset_id, error=e)
                        print(txt_color("[AVERTISSEMENT] ", "erreur"), log_warn_message)
                        preset_dict['loras'] = []

                    try:
                        seed_str = preset_dict.get('seed')
                        preset_dict['seed'] = int(seed_str) if seed_str is not None else -1
                    except (ValueError, TypeError):
                         print(txt_color("[AVERTISSEMENT]", "erreur"), f"Impossible de convertir le seed '{seed_str}' en entier pour le preset {preset_id}. Utilisation de -1.")
                         preset_dict['seed'] = -1
                else:
                    log_warn_message = translate("log_warn_preset_not_found_load", self.translations).format(id=preset_id)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)

            except sqlite3.Error as e:
                log_message = translate("log_erreur_load_data", self.translations).format(id=preset_id, error=e)
                print(txt_color("[ERREUR DB] ", "erreur"), log_message)
                traceback.print_exc()
        return preset_dict 

    def delete_preset(self, preset_id):
        """
        Supprime un preset. Utilise self.translations.
        """
        # Utilise self.translations pour les messages
        msg_err_connexion = translate("erreur_db_connexion", self.translations)
        msg_success = translate("preset_supprime", self.translations)
        msg_err_not_found = translate("erreur_preset_non_trouve", self.translations)
        msg_err_db = translate("erreur_suppression_preset", self.translations)

        with self._managed_db_connection() as conn:
            if conn is None:
                return False, msg_err_connexion

            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    log_message = translate("log_preset_deleted", self.translations).format(id=preset_id)
                    print(txt_color("[INFO DB] ", "ok"), log_message)
                    return True, msg_success
                else:
                    log_warn_message = translate("log_warn_preset_not_found_delete", self.translations).format(id=preset_id)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                    return False, msg_err_not_found

            except sqlite3.Error as e:
                log_message = translate("log_erreur_delete", self.translations).format(id=preset_id, error=e)
                print(txt_color("[ERREUR DB] ", "erreur"), log_message)
                traceback.print_exc()
                return False, msg_err_db

    def update_preset_rating(self, preset_id, new_rating):
        """
        Met à jour la note. Utilise self.translations.
        """
        # Utilise self.translations pour les messages
        msg_err_connexion = translate("erreur_db_connexion", self.translations)
        msg_err_not_found = translate("erreur_preset_non_trouve", self.translations)
        msg_err_invalid = translate("erreur_note_invalide", self.translations)
        msg_err_db = translate("erreur_db_maj_note", self.translations)

        with self._managed_db_connection() as conn:
            if conn is None:
                return False, msg_err_connexion

            try:
                rating_value = int(new_rating)
                if not (0 <= rating_value <= 5): rating_value = 0

                cursor = conn.cursor()
                cursor.execute("UPDATE presets SET rating = ? WHERE id = ?", (rating_value, preset_id))
                conn.commit()

                if cursor.rowcount > 0:
                    log_message = translate("log_rating_updated", self.translations).format(id=preset_id, rating=rating_value)
                    print(txt_color("[INFO DB] ", "info"), log_message)
                    return True, ""
                else:
                    log_warn_message = translate("log_warn_preset_not_found_rating", self.translations).format(id=preset_id)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                    return False, msg_err_not_found

            except (ValueError, TypeError):
                log_message = translate("log_erreur_invalid_rating_value", self.translations).format(id=preset_id, value=new_rating)
                print(txt_color("[ERREUR] ", "erreur"), log_message)
                return False, msg_err_invalid
            except sqlite3.Error as e:
                log_message = translate("log_erreur_db_update_rating", self.translations).format(id=preset_id, error=e)
                print(txt_color("[ERREUR DB] ", "erreur"), log_message)
                traceback.print_exc()
                return False, msg_err_db



    def rename_preset(self, preset_id, new_name):
        """
        Renomme un preset existant.
        Args:
            preset_id (int): L'ID du preset à renommer.
            new_name (str): Le nouveau nom souhaité pour le preset.
        Returns:
            (bool, str): Tuple indiquant le succès/échec et un message traduit.
        """
        # Messages pour l'UI
        msg_err_nom_vide = translate("erreur_nom_preset_vide", self.translations)
        msg_err_connexion = translate("erreur_db_connexion", self.translations)
        msg_success = translate("preset_renomme_succes", self.translations).format(new_name) # Nouvelle clé de traduction
        msg_err_not_found = translate("erreur_preset_non_trouve", self.translations)
        msg_err_db = translate("erreur_db_rename", self.translations) # Nouvelle clé
        msg_err_name_exists = translate("erreur_nom_preset_existe_rename", self.translations).format(new_name) # Nouvelle clé

        if not new_name:
            return False, msg_err_nom_vide

        with self._managed_db_connection() as conn:
            if conn is None:
                return False, msg_err_connexion

            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM presets WHERE name = ? AND id != ?", (new_name, preset_id))
                existing = cursor.fetchone()
                if existing:
                    log_warn_message = translate("log_warn_rename_name_exists", self.translations).format(name=new_name)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                    return False, msg_err_name_exists

                cursor.execute("UPDATE presets SET name = ? WHERE id = ?", (new_name, preset_id))
                conn.commit()

                if cursor.rowcount > 0:
                    log_message = translate("log_preset_renamed", self.translations).format(id=preset_id, new_name=new_name)
                    print(txt_color("[INFO DB] ", "ok"), log_message)
                    return True, msg_success
                else:
                    log_warn_message = translate("log_warn_preset_not_found_rename", self.translations).format(id=preset_id)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                    return False, msg_err_not_found

            except sqlite3.IntegrityError: 
                 log_warn_message = translate("log_warn_rename_name_exists", self.translations).format(name=new_name)
                 print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                 return False, msg_err_name_exists
            except sqlite3.Error as e:
                log_message = translate("log_erreur_db_rename", self.translations).format(id=preset_id, error=e)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                traceback.print_exc()
                return False, msg_err_db

    def get_distinct_preset_filters(self):
        """
        Récupère les listes uniques de modèles, samplers et noms de LoRAs utilisés dans les presets.
        """
        filters = {'models': [], 'samplers': [], 'loras': set()} # Utiliser un set pour les LoRAs

        with self._managed_db_connection() as conn:
            if conn is None:
                return {'models': [], 'samplers': [], 'loras': []}

            try:
                cursor = conn.cursor()

                cursor.execute("SELECT DISTINCT model FROM presets WHERE model IS NOT NULL AND model != '' ORDER BY model ASC")
                filters['models'] = [row['model'] for row in cursor.fetchall()]

                cursor.execute("SELECT DISTINCT sampler_key FROM presets WHERE sampler_key IS NOT NULL AND sampler_key != '' ORDER BY sampler_key ASC")
                filters['samplers'] = [row['sampler_key'] for row in cursor.fetchall()]

                cursor.execute("SELECT loras FROM presets WHERE loras IS NOT NULL AND loras != '[]' AND loras != ''")
                loras_json_list = cursor.fetchall()

                for row in loras_json_list:
                    loras_str = row['loras']
                    try:
                        loras_list_data = json.loads(loras_str)
                        if isinstance(loras_list_data, list):
                            for lora_item in loras_list_data:
                                if isinstance(lora_item, dict) and 'name' in lora_item and lora_item['name']:
                                    filters['loras'].add(lora_item['name'])
                    except (json.JSONDecodeError, TypeError) as e:
                        print(txt_color("[AVERTISSEMENT DB]", "erreur"), f"Impossible de parser le JSON LoRA: {loras_str[:50]}... Erreur: {e}")

                filters['loras'] = sorted(list(filters['loras']))

                log_message = translate("log_filters_loaded", self.translations).format(
                    models=len(filters['models']),
                    samplers=len(filters['samplers']),
                    loras=len(filters['loras'])
                )
                print(txt_color("[INFO DB]", "info"), log_message)

            except sqlite3.Error as e:
                log_message = translate("log_erreur_load_filters", self.translations).format(error=e)
                print(txt_color("[ERREUR DB]", "erreur"), log_message)
                traceback.print_exc()
                return {'models': [], 'samplers': [], 'loras': []}
        return filters

    # --- Gestion de l'état des modules ---

    def _load_module_states(self):
        """Charge l'état des modules depuis le fichier de configuration JSON."""
        try:
            if os.path.exists(self.module_config_path):
                with open(self.module_config_path, 'r', encoding='utf-8') as f:
                    states = json.load(f)
                    # S'assurer que c'est un dictionnaire, sinon retourner un dictionnaire vide
                    if isinstance(states, dict):
                        print(txt_color("[INFO] ", "info"), translate("log_module_states_loaded", self.translations))
                        return states
                    else:
                        print(txt_color("[AVERTISSEMENT] ", "erreur"), translate("log_warn_module_config_invalid_format", self.translations))
                        return {}
            else:
                print(txt_color("[INFO] ", "info"), translate("log_module_config_not_found_creating", self.translations))
                return {} # Retourne un dict vide si le fichier n'existe pas encore
        except (json.JSONDecodeError, IOError) as e:
            log_message = translate("log_erreur_load_module_states", self.translations).format(error=e)
            print(txt_color("[ERREUR] ", "erreur"), log_message)
            traceback.print_exc()
            return {} # En cas d'erreur, retourne un dict vide

    def _save_module_states_to_file(self):
        """Sauvegarde l'état actuel des modules dans le fichier de configuration JSON."""
        try:
            with open(self.module_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.loaded_module_states, f, indent=4, ensure_ascii=False)
            print(txt_color("[INFO] ", "ok"), translate("log_module_states_saved", self.translations))
        except IOError as e:
            log_message = translate("log_erreur_save_module_states", self.translations).format(error=e)
            print(txt_color("[ERREUR] ", "erreur"), log_message)
            traceback.print_exc()

    def save_module_state(self, module_name, is_active):
        """
        Sauvegarde l'état (activé/désactivé) d'un module spécifique.

        Args:
            module_name (str): Le nom du module.
            is_active (bool): True si le module est actif, False sinon.
        """
        self.loaded_module_states[module_name] = is_active
        self._save_module_states_to_file()
        log_message = translate("log_module_state_updated", self.translations).format(module_name=module_name, state="activé" if is_active else "désactivé")
        print(txt_color("[INFO] ", "info"), log_message)

    def get_module_state(self, module_name, default_active=False):
        """Récupère l'état d'un module, avec une valeur par défaut si non trouvé."""
        return self.loaded_module_states.get(module_name, default_active)
