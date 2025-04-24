# presets/presets_Manager.py

import sqlite3
import os
import json
from datetime import datetime
from io import BytesIO
from PIL import Image
import traceback

# Import direct depuis Utils (sans try/except comme vous l'avez suggéré)
from Utils.utils import txt_color, translate

# Chemin vers la base de données
DB_FILE_PATH = os.path.join(os.path.dirname(__file__), 'presets.db')

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
        self._init_preset_db() # Initialiser la DB lors de la création de l'instance

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

    def _init_preset_db(self):
        """
        Initialise la base de données et crée la table 'presets' si elle n'existe pas.
        Méthode interne appelée par __init__.
        """
        conn = self._get_db_connection() # Utilise la méthode interne
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
                    model TEXT, vae TEXT, original_prompt TEXT, prompt TEXT, negative_prompt TEXT, styles TEXT,
                    guidance_scale REAL, num_steps INTEGER, sampler_key TEXT, seed INTEGER,
                    width INTEGER, height INTEGER, loras TEXT, preview_image BLOB, notes TEXT
                );
            """)
            conn.commit()
            print(txt_color("[INFO DB] ", "ok"), translate("db_init_success", self.translations))
            cursor.execute("PRAGMA table_info(presets)")
            columns = [column['name'] for column in cursor.fetchall()]
            if 'original_prompt' not in columns:
                print(txt_color("[INFO DB] ", "info"), translate("db_adding_original_prompt_column", self.translations)) # Nouvelle clé de traduction
                cursor.execute("ALTER TABLE presets ADD COLUMN original_prompt TEXT")
                conn.commit()
                print(txt_color("[INFO DB] ", "ok"), translate("db_original_prompt_column_added", self.translations))
        except sqlite3.Error as e:
            log_message = translate("erreur_init_table", self.translations).format(error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

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

        conn = self._get_db_connection() # Utilise la méthode interne
        if conn is None: return False, msg_err_connexion

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

            # Assurer que toutes les colonnes attendues sont présentes (avec None si manquant)
            # Utiliser une liste définie des colonnes pour garantir l'ordre
            # (Assurez-vous que cette liste correspond à votre table)
            expected_columns = [
                'name', 'preset_type', 'created_at', 'last_used', 'rating',
                'model', 'vae', 'original_prompt', 'prompt', 'negative_prompt', 'styles',
                'guidance_scale', 'num_steps', 'sampler_key', 'seed',
                'width', 'height', 'loras', 'preview_image', 'notes'
            ]
            # Ajouter created_at si ce n'est pas déjà fait par la DB
            if 'created_at' not in filtered_data:
                 filtered_data['created_at'] = datetime.now().isoformat()

            # Préparer les valeurs dans l'ordre attendu
            values_to_insert = []
            for col in expected_columns:
                 values_to_insert.append(filtered_data.get(col)) # Utiliser .get pour éviter KeyError

            columns_str = ', '.join(expected_columns)
            placeholders = ', '.join('?' * len(expected_columns))
            sql = f"INSERT OR REPLACE INTO presets ({columns_str}) VALUES ({placeholders})"

            cursor.execute(sql, values_to_insert)
            conn.commit()

            # Log console unifié
            log_message = translate("log_preset_saved", self.translations).format(name=preset_name)
            print(txt_color("[INFO DB] ", "ok"), log_message)
            return True, msg_success

        except sqlite3.IntegrityError as integrity_err:
            # Gérer spécifiquement l'erreur de nom unique
            log_message = translate("log_warn_rename_name_exists", self.translations).format(name=preset_name) # Réutiliser clé
            print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_message)
            return False, translate("erreur_nom_preset_existe_rename", self.translations).format(preset_name) # Réutiliser clé
        except sqlite3.Error as db_err:
            # Log console unifié
            log_message = translate("log_erreur_db_save", self.translations).format(name=preset_name, error=db_err)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
            # Vérifier si c'est une erreur d'overflow pour le seed (même si on convertit en texte)
            if "too large" in str(db_err):
                return False, msg_err_seed_overflow
            return False, msg_err_db
        except Exception as e:
            # Log console unifié
            log_message = translate("log_erreur_general_save", self.translations).format(name=preset_name, error=e)
            print(txt_color("[ERREUR] ", "erreur"), log_message)
            traceback.print_exc()
            return False, msg_err_general
        finally:
            if conn: conn.close()


# Dans la classe PresetManager

    # --- MODIFIER LA SIGNATURE ---
    def load_presets_for_display(self, preset_type='gen_image', search_term="", sort_by="Nom A-Z",
                                 selected_models=None, selected_samplers=None, selected_loras=None):
        """
        Charge les presets pour affichage, en appliquant les filtres.
        """
        conn = self._get_db_connection()
        if conn is None: return []

        presets_data = []
        try:
            cursor = conn.cursor()
            # --- AJOUTER model, sampler_key, loras à SELECT ---
            query = """
                SELECT id, name, rating, preview_image, created_at, last_used, notes,
                       model, sampler_key, loras
                FROM presets
                WHERE preset_type = ?
            """
            params = [preset_type]

            # --- Filtre par terme de recherche ---
            if search_term:
                query += " AND name LIKE ?"
                params.append(f"%{search_term}%")

            # --- Filtre par modèles (si une liste est fournie) ---
            if selected_models and isinstance(selected_models, list):
                placeholders = ','.join('?' * len(selected_models))
                query += f" AND model IN ({placeholders})"
                params.extend(selected_models)

            # --- Filtre par samplers (si une liste est fournie) ---
            if selected_samplers and isinstance(selected_samplers, list):
                placeholders = ','.join('?' * len(selected_samplers))
                query += f" AND sampler_key IN ({placeholders})" # Utiliser sampler_key
                params.extend(selected_samplers)

            # --- Clause ORDER BY (inchangée) ---
            order_clause = " ORDER BY "
            # ... (logique de tri identique) ...
            if sort_by == "Nom A-Z": order_clause += "name ASC"
            elif sort_by == "Nom Z-A": order_clause += "name DESC"
            elif sort_by == "Date Création": order_clause += "created_at DESC"
            elif sort_by == "Date Utilisation": order_clause += "last_used DESC NULLS LAST"
            elif sort_by == "Note": order_clause += "rating DESC"
            else: order_clause += "name ASC" # Défaut
            query += order_clause

            # --- Exécution de la requête SQL (sans filtre LoRA pour l'instant) ---
            cursor.execute(query, params)
            results_sql = cursor.fetchall() # Récupérer tous les résultats correspondants

            # --- Filtrage Python pour les LoRAs (si nécessaire) ---
            if selected_loras and isinstance(selected_loras, list):
                presets_data = [] # Commencer avec une liste vide
                for row in results_sql:
                    loras_str = row['loras']
                    preset_loras_names = set() # Noms des LoRAs pour ce preset
                    if loras_str:
                        try:
                            loras_list_data = json.loads(loras_str)
                            if isinstance(loras_list_data, list):
                                for lora_item in loras_list_data:
                                    if isinstance(lora_item, dict) and 'name' in lora_item and lora_item['name']:
                                        preset_loras_names.add(lora_item['name'])
                        except (json.JSONDecodeError, TypeError):
                            pass # Ignorer les JSON invalides lors du filtrage

                    # Vérifier si TOUS les LoRAs sélectionnés sont présents dans ce preset
                    if set(selected_loras).issubset(preset_loras_names):
                        presets_data.append(row) # Garder ce preset
            else:
                # Si pas de filtre LoRA, utiliser tous les résultats SQL
                presets_data = results_sql

            # Log console unifié
            log_message = translate("log_presets_loaded_display_filtered", self.translations).format( # Nouvelle clé
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
        finally:
            if conn: conn.close()
        return presets_data # Retourne la liste (potentiellement filtrée par Python)

    def load_preset_data(self, preset_id):
        """
        Charge les données complètes d'un preset. Inclut maintenant 'original_prompt'.
        Utilise self.translations.
        """
        conn = self._get_db_connection()
        if conn is None: return None

        preset_dict = None
        try:
            cursor = conn.cursor()
            # SELECT * récupère toutes les colonnes, y compris original_prompt si elle existe
            cursor.execute("SELECT * FROM presets WHERE id = ?", (preset_id,))
            preset_data = cursor.fetchone() # preset_data est un objet sqlite3.Row

            if preset_data:
                preset_dict = dict(preset_data) # Convertit en dictionnaire standard
                # 'original_prompt' sera une clé dans preset_dict si la colonne existe

                # ... (log et mise à jour last_used inchangés) ...
                log_message = translate("log_preset_data_loading", self.translations).format(id=preset_id, name=preset_dict.get('name', ''))
                print(txt_color("[INFO DB] ", "info"), log_message)
                try:
                    cursor.execute("UPDATE presets SET last_used = ? WHERE id = ?", (datetime.now(), preset_id))
                    conn.commit()
                except sqlite3.Error as update_err:
                    log_warn_message = translate("log_warn_update_last_used", self.translations).format(id=preset_id, error=update_err)
                    print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)

                # Désérialisation JSON (inchangée)
                try:
                    # Utiliser .get() pour éviter KeyError si la colonne n'existe pas (ancien schéma)
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

                # --- Conversion du seed (lu comme TEXT) en int ---
                try:
                    seed_str = preset_dict.get('seed')
                    preset_dict['seed'] = int(seed_str) if seed_str is not None else -1 # Ou une autre valeur par défaut
                except (ValueError, TypeError):
                     print(txt_color("[AVERTISSEMENT]", "erreur"), f"Impossible de convertir le seed '{seed_str}' en entier pour le preset {preset_id}. Utilisation de -1.")
                     preset_dict['seed'] = -1
                # --- Fin Conversion Seed ---

            else:
                log_warn_message = translate("log_warn_preset_not_found_load", self.translations).format(id=preset_id)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)

        except sqlite3.Error as e:
            log_message = translate("log_erreur_load_data", self.translations).format(id=preset_id, error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
        finally:
            if conn: conn.close()
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

        conn = self._get_db_connection() # Utilise la méthode interne
        if conn is None: return False, msg_err_connexion

        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
            conn.commit()

            if cursor.rowcount > 0:
                # Log console unifié - Utilise self.translations
                log_message = translate("log_preset_deleted", self.translations).format(id=preset_id)
                print(txt_color("[INFO DB] ", "ok"), log_message)
                return True, msg_success
            else:
                # Log console unifié - Utilise self.translations
                log_warn_message = translate("log_warn_preset_not_found_delete", self.translations).format(id=preset_id)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                return False, msg_err_not_found

        except sqlite3.Error as e:
            # Log console unifié - Utilise self.translations
            log_message = translate("log_erreur_delete", self.translations).format(id=preset_id, error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
            return False, msg_err_db
        finally:
            if conn: conn.close()

    def update_preset_rating(self, preset_id, new_rating):
        """
        Met à jour la note. Utilise self.translations.
        """
        # Utilise self.translations pour les messages
        msg_err_connexion = translate("erreur_db_connexion", self.translations)
        msg_err_not_found = translate("erreur_preset_non_trouve", self.translations)
        msg_err_invalid = translate("erreur_note_invalide", self.translations)
        msg_err_db = translate("erreur_db_maj_note", self.translations)

        conn = self._get_db_connection() # Utilise la méthode interne
        if conn is None: return False, msg_err_connexion

        try:
            rating_value = int(new_rating)
            if not (0 <= rating_value <= 5): rating_value = 0

            cursor = conn.cursor()
            cursor.execute("UPDATE presets SET rating = ? WHERE id = ?", (rating_value, preset_id))
            conn.commit()

            if cursor.rowcount > 0:
                # Log console unifié - Utilise self.translations
                log_message = translate("log_rating_updated", self.translations).format(id=preset_id, rating=rating_value)
                print(txt_color("[INFO DB] ", "info"), log_message)
                return True, ""
            else:
                # Log console unifié - Utilise self.translations
                log_warn_message = translate("log_warn_preset_not_found_rating", self.translations).format(id=preset_id)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                return False, msg_err_not_found

        except (ValueError, TypeError):
            # Log console unifié - Utilise self.translations
            log_message = translate("log_erreur_invalid_rating_value", self.translations).format(id=preset_id, value=new_rating)
            print(txt_color("[ERREUR] ", "erreur"), log_message)
            return False, msg_err_invalid
        except sqlite3.Error as e:
            # Log console unifié - Utilise self.translations
            log_message = translate("log_erreur_db_update_rating", self.translations).format(id=preset_id, error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
            return False, msg_err_db
        finally:
            if conn: conn.close()



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

        conn = self._get_db_connection()
        if conn is None:
            return False, msg_err_connexion

        try:
            cursor = conn.cursor()
            # 1. Vérifier si le nouveau nom existe déjà pour un AUTRE preset
            cursor.execute("SELECT id FROM presets WHERE name = ? AND id != ?", (new_name, preset_id))
            existing = cursor.fetchone()
            if existing:
                # Log console
                log_warn_message = translate("log_warn_rename_name_exists", self.translations).format(name=new_name)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                return False, msg_err_name_exists

            # 2. Si le nom est libre ou appartient déjà à ce preset_id, effectuer la mise à jour
            cursor.execute("UPDATE presets SET name = ? WHERE id = ?", (new_name, preset_id))
            conn.commit()

            if cursor.rowcount > 0:
                # Log console
                log_message = translate("log_preset_renamed", self.translations).format(id=preset_id, new_name=new_name)
                print(txt_color("[INFO DB] ", "ok"), log_message)
                return True, msg_success
            else:
                # Log console
                log_warn_message = translate("log_warn_preset_not_found_rename", self.translations).format(id=preset_id)
                print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
                return False, msg_err_not_found

        except sqlite3.IntegrityError: # Devrait être attrapé par la vérification précédente, mais par sécurité
             log_warn_message = translate("log_warn_rename_name_exists", self.translations).format(name=new_name)
             print(txt_color("[AVERTISSEMENT DB] ", "erreur"), log_warn_message)
             return False, msg_err_name_exists
        except sqlite3.Error as e:
            # Log console
            log_message = translate("log_erreur_db_rename", self.translations).format(id=preset_id, error=e)
            print(txt_color("[ERREUR DB] ", "erreur"), log_message)
            traceback.print_exc()
            return False, msg_err_db
        finally:
            if conn:
                conn.close()

    def get_distinct_preset_filters(self):
        """
        Récupère les listes uniques de modèles, samplers et noms de LoRAs utilisés dans les presets.
        """
        conn = self._get_db_connection()
        if conn is None:
            return {'models': [], 'samplers': [], 'loras': []}

        filters = {'models': [], 'samplers': [], 'loras': set()} # Utiliser un set pour les LoRAs

        try:
            cursor = conn.cursor()

            # Récupérer les modèles distincts
            cursor.execute("SELECT DISTINCT model FROM presets WHERE model IS NOT NULL AND model != '' ORDER BY model ASC")
            filters['models'] = [row['model'] for row in cursor.fetchall()]

            # Récupérer les samplers distincts
            cursor.execute("SELECT DISTINCT sampler_key FROM presets WHERE sampler_key IS NOT NULL AND sampler_key != '' ORDER BY sampler_key ASC")
            filters['samplers'] = [row['sampler_key'] for row in cursor.fetchall()] # Utiliser sampler_key

            # Récupérer tous les LoRAs (chaînes JSON) et les traiter en Python
            cursor.execute("SELECT loras FROM presets WHERE loras IS NOT NULL AND loras != '[]' AND loras != ''")
            loras_json_list = cursor.fetchall()

            for row in loras_json_list:
                loras_str = row['loras']
                try:
                    loras_list_data = json.loads(loras_str)
                    if isinstance(loras_list_data, list):
                        for lora_item in loras_list_data:
                            if isinstance(lora_item, dict) and 'name' in lora_item and lora_item['name']:
                                filters['loras'].add(lora_item['name']) # Ajouter au set
                except (json.JSONDecodeError, TypeError) as e:
                    # Log discret si un JSON est invalide
                    print(txt_color("[AVERTISSEMENT DB]", "erreur"), f"Impossible de parser le JSON LoRA: {loras_str[:50]}... Erreur: {e}")

            # Convertir le set de LoRAs en liste triée
            filters['loras'] = sorted(list(filters['loras']))

            log_message = translate("log_filters_loaded", self.translations).format( # Nouvelle clé de traduction
                models=len(filters['models']),
                samplers=len(filters['samplers']),
                loras=len(filters['loras'])
            )
            print(txt_color("[INFO DB]", "info"), log_message)

        except sqlite3.Error as e:
            log_message = translate("log_erreur_load_filters", self.translations).format(error=e) # Nouvelle clé
            print(txt_color("[ERREUR DB]", "erreur"), log_message)
            traceback.print_exc()
            # Retourner des listes vides en cas d'erreur DB
            return {'models': [], 'samplers': [], 'loras': []}
        finally:
            if conn:
                conn.close()

        return filters
