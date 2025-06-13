# Utils/llm_prompter_util.py
from transformers import AutoModelForCausalLM, AutoTokenizer # <-- MODIFIÉ
import torch
import traceback
from Utils.utils import txt_color, translate

# Global model and tokenizer instances
llm_model_prompter = None
llm_tokenizer_prompter = None # <-- AJOUTÉ
# llm_device_prompter is handled by device_map="auto" for Transformers

def init_llm_prompter(model_path: str, # Hugging Face model name or path
                      translations=None) -> bool: # Ajout du type de retour
    """
    Initialise le modèle de langage (Transformers) pour l'amélioration des prompts.
    Retourne True en cas de succès, False sinon.
    """
    global llm_model_prompter, llm_tokenizer_prompter

    if llm_model_prompter is not None and llm_tokenizer_prompter is not None:
        msg = translate("llm_prompter_already_initialized", translations) if translations else "LLM Prompter déjà initialisé."
        print(f"{txt_color('[INFO]', 'info')} {msg}")
        return True # Déjà initialisé = succès

    print(f"{txt_color('[INFO]', 'info')} Initialisation du LLM Prompter avec le modèle Transformers {model_path}...")

    try:
        llm_tokenizer_prompter = AutoTokenizer.from_pretrained(model_path)
        llm_model_prompter = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto", # Using "auto" for flexibility
            device_map="cpu"    # Force CPU usage
        )
        # Ensure pad_token is set for generation, Qwen models might not have it by default
        if llm_tokenizer_prompter.pad_token is None:
            llm_tokenizer_prompter.pad_token = llm_tokenizer_prompter.eos_token
            print(f"{txt_color('[INFO]', 'info')} Tokenizer pad_token non défini. Utilisation de eos_token comme pad_token.")

        print(f"{txt_color('[OK]', 'ok')} LLM Prompter (Transformers: {model_path}) initialisé avec succès.")
        return True

    except Exception as e:
        error_msg = str(e)
        # Check for common model not found indicators
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower() or "repository" in error_msg.lower() :
             print(f"{txt_color('[ERREUR]', 'erreur')} Erreur: Le modèle Transformers '{model_path}' n'a pas été trouvé ou n'est pas accessible.")
             print(f"{txt_color('[ERREUR]', 'erreur')} Assurez-vous que le nom du modèle est correct et que vous avez une connexion internet si le modèle doit être téléchargé.")
             print(f"{txt_color('[ERREUR]', 'erreur')} Détails de l'erreur: {error_msg}")
        else:
             print(f"{txt_color('[ERREUR]', 'erreur')} Erreur inattendue lors de l'initialisation du LLM Prompter Transformers: {error_msg}")
        traceback.print_exc()
        llm_model_prompter = None
        llm_tokenizer_prompter = None
        return False

def unload_llm_prompter(translations=None):
    """
    Décharge le modèle LLM Prompter de la mémoire.
    """
    global llm_model_prompter, llm_tokenizer_prompter
    if llm_model_prompter is None and llm_tokenizer_prompter is None:
        msg = translate("llm_prompter_not_loaded_nothing_to_unload", translations) if translations else "LLM Prompter n'est pas chargé, rien à décharger."
        print(f"{txt_color('[INFO]', 'info')} {msg}")
        return

    print(f"{txt_color('[INFO]', 'info')} Déchargement du LLM Prompter...")
    del llm_model_prompter
    del llm_tokenizer_prompter
    llm_model_prompter = None
    llm_tokenizer_prompter = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{txt_color('[INFO]', 'info')} Cache CUDA vidé.")
    print(f"{txt_color('[OK]', 'ok')} LLM Prompter déchargé avec succès.")

def generate_enhanced_prompt(base_prompt: str, model_path: str, translations=None, max_new_tokens: int = 400) -> str: # Augmentation de max_new_tokens
    """
    Génère un prompt enrichi à partir d'un prompt de base en utilisant le LLM Transformers.
    Tente d'initialiser le LLM si ce n'est pas déjà fait.
    """
    global llm_model_prompter, llm_tokenizer_prompter # Nécessaire si init_llm_prompter est appelé ici

    if llm_model_prompter is None or llm_tokenizer_prompter is None:
        # Tentative d'initialisation à la volée
        print(f"{txt_color('[INFO]', 'info')} LLM Prompter non chargé. Tentative d'initialisation pour l'amélioration du prompt...")
        if not init_llm_prompter(model_path, translations):
            # L'initialisation a échoué
            msg_err = translate("erreur_llm_prompter_init_failed_on_demand", translations) if translations else "Erreur: LLM Prompter (Transformers) n'a pas pu être initialisé à la demande."
            print(f"{txt_color('[ERREUR]', 'erreur')} {msg_err}")
            return base_prompt # Retourne le prompt de base si l'initialisation échoue
        # Si l'initialisation réussit, llm_model_prompter et llm_tokenizer_prompter sont maintenant définis.

    # Instruction pour le modèle Qwen.
    # "Réponds UNIQUEMENT avec le prompt amélioré, sans texte additionnel." est crucial.
    # Ajout de "en anglais" car le traitement ultérieur semble viser l'anglais pour SDXL.
    user_message_content = f"Génère un prompt détaillé et imaginatif pour un générateur d'images à partir de l'idée suivante : \"{base_prompt}\". Réponds UNIQUEMENT avec le prompt amélioré, sans texte additionnel, en anglais."


    messages = [
        {"role": "user", "content": user_message_content}
    ]

    try:
        # Prepare model input using the tokenizer's chat template
        text_input_for_model = llm_tokenizer_prompter.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, # Important for instruct/chat models
            enable_thinking=True # For Qwen's thinking process
        )
        model_inputs = llm_tokenizer_prompter([text_input_for_model], return_tensors="pt").to(llm_model_prompter.device)

        # Define generation parameters
        temperature = 0.7 
        top_p = 0.9
        
        pad_token_id_for_generation = llm_tokenizer_prompter.pad_token_id
        if pad_token_id_for_generation is None:
            pad_token_id_for_generation = llm_tokenizer_prompter.eos_token_id

        generated_ids = llm_model_prompter.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id_for_generation,
            eos_token_id=llm_tokenizer_prompter.eos_token_id
        )
        
        # Decode the output, excluding input tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        final_content_after_thinking = ""

        # Attempt 1: Parse using the special token ID for </think> (151668).
        # This token ID is specific to some Qwen models for the </think> tag.
        qwen_think_end_token_id = 151668
        content_start_index_in_ids_after_token = 0 # Default to start if token not found

        try:
            # Find the index *after* the last occurrence of the </think> token.
            # The index points to the start of the actual content after the token.
            content_start_index_in_ids_after_token = len(output_ids) - output_ids[::-1].index(qwen_think_end_token_id)
            
            # Decode the segment identified by token parsing
            decoded_segment = llm_tokenizer_prompter.decode(output_ids[content_start_index_in_ids_after_token:], skip_special_tokens=True).strip()
            final_content_after_thinking = decoded_segment
            print(f"{txt_color('[DEBUG]', 'info')} Found and processed Qwen </think> token (ID: {qwen_think_end_token_id}). Content starts after token.")

        except ValueError:
            # Qwen </think> token (151668) not found. Decode the full output and try string parsing.
            print(f"{txt_color('[DEBUG]', 'info')} Qwen </think> token (ID: {qwen_think_end_token_id}) not found. Decoding full output for string-based parsing.")
            full_decoded_output = llm_tokenizer_prompter.decode(output_ids, skip_special_tokens=True).strip()

            # Attempt 2: String-based parsing on the full_decoded_output.
            think_start_tag_str = "<think>"
            think_end_tag_str = "</think>"
            
            # Check for the </think> string tag
            last_think_end_pos = full_decoded_output.rfind(think_end_tag_str)
            
            if last_think_end_pos != -1:
                # </think> string found. Take content after it.
                final_content_after_thinking = full_decoded_output[last_think_end_pos + len(think_end_tag_str):].strip()
                print(f"{txt_color('[DEBUG]', 'info')} Found '{think_end_tag_str}' string. Extracted content after it.")
            elif full_decoded_output.lower().strip().startswith(think_start_tag_str):
                # Starts with "<think>" string, but no corresponding "</think>" (token or string) was found.
                # This indicates a truncated or malformed thinking block.
                print(f"{txt_color('[WARN]', 'warning')} Output starts with '{think_start_tag_str}' but no corresponding '{think_end_tag_str}' (token or string) was found. Enhancement failed, returning base prompt.")
                return base_prompt # Fallback to base_prompt
            else:
                # No <think> start, no </think> token, no </think> string. Assume all output is content.
                final_content_after_thinking = full_decoded_output
                print(f"{txt_color('[DEBUG]', 'info')} No </think> token or string, and does not start with '{think_start_tag_str}'. Assuming all output is content.")

        raw_generated_output = final_content_after_thinking # This is now the potentially cleaned content
        cleaned_prompt = raw_generated_output # Start cleaning from this point

        # Remove common conversational preambles if the model didn't strictly follow "UNIQUEMENT"
        preambles_to_remove = [
            "Voici le prompt amélioré :", "Here is the enhanced prompt:",
            "Enhanced prompt:", "Prompt amélioré :", "Okay, here's the prompt:",
            "Sure, here is the detailed and imaginative prompt:",
            "Here's a detailed and imaginative prompt:",
            # Add more based on observed Qwen outputs
        ]
        for preamble in preambles_to_remove:
            if cleaned_prompt.lower().startswith(preamble.lower()):
                cleaned_prompt = cleaned_prompt[len(preamble):].strip()
                break

        if cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"'):
            cleaned_prompt = cleaned_prompt[1:-1].strip()
        elif cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'"):
            cleaned_prompt = cleaned_prompt[1:-1].strip()

        # If cleaning results in an empty string or no change, revert to base_prompt or raw output
        if cleaned_prompt and cleaned_prompt.strip() and cleaned_prompt.strip().lower() != base_prompt.strip().lower():
            final_prompt = cleaned_prompt.strip()
        elif raw_generated_output and raw_generated_output.strip() and raw_generated_output.strip().lower() != base_prompt.strip().lower():
            # If cleaning failed but raw output is different and not empty
            final_prompt = raw_generated_output.strip() 
            print(f"{txt_color('[WARN]', 'warning')} LLM output cleaning was not effective, using raw output.")
        else:
            final_prompt = base_prompt # Fallback to base_prompt if enhancement fails or is identical

    except Exception as e:
        err_msg_template = "Erreur lors de la génération du prompt enrichi avec Transformers: {error}"
        err_msg = translate("erreur_generation_prompt_enrichi_transformers", translations) if translations else err_msg_template.format(error=e)
        print(f"{txt_color('[ERREUR]', 'erreur')} {err_msg}")
        traceback.print_exc()
        return base_prompt

    print(f"{txt_color('[INFO]', 'info')} Prompt de base: \"{base_prompt}\" -> Prompt enrichi (Transformers brut): \"{raw_generated_output}\" -> Prompt final: \"{final_prompt}\"")
    return final_prompt
