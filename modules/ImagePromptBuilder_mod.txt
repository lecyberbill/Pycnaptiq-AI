import os
import json
import gradio as gr
import random # Import random module

from Utils.utils import txt_color, translate

MODULE_NAME = "ImagePromptBuilder"
DATA_DIR = os.path.join(os.path.dirname(__file__), "ImagePromptBuilder_data")

def _load_data_from_json(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERREUR] Fichier de données introuvable : {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"[ERREUR] Erreur de décodage JSON pour le fichier : {filepath}")
        return []

module_json_path = os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}_mod.json")

try:
    with open(module_json_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
except FileNotFoundError:
    print(f"[ERREUR] Fichier JSON du module {MODULE_NAME} introuvable.")
    module_data = {"name": MODULE_NAME} 
except json.JSONDecodeError:
    print(f"[ERREUR] Erreur de décodage JSON pour le module {MODULE_NAME}.")
    module_data = {"name": MODULE_NAME} 

def initialize(global_translations, model_manager_instance, gestionnaire_instance, global_config=None):
    print(txt_color("[OK] ", "ok"), f"Initialisation module {module_data.get('name', MODULE_NAME)}")
    return ImagePromptBuilderModule(global_translations, global_config)

class ImagePromptBuilderModule:
    def __init__(self, global_translations, global_config=None):
        self.global_translations = global_translations
        self.global_config = global_config
        self.module_translations = {}

        # I. Sujet Principal & Actions
        # Load the nested subject details and extract sub-categories
        subject_details_data = _load_data_from_json("subject_details.json")
        self.subject_character_types = subject_details_data.get("character_types", [])
        self.subject_creatures_entities = subject_details_data.get("creatures_entities", [])
        self.subject_clothing_styles = subject_details_data.get("clothing_styles", [])
        self.subject_physical_attributes = subject_details_data.get("physical_attributes", [])
        self.subject_visual_supernatural_attributes = subject_details_data.get("visual_supernatural_attributes", [])
        self.actions_verbs = _load_data_from_json("actions_verbs.json")

        # II. Style Visuel & Artistique
        self.medium_technique = _load_data_from_json("medium_technique.json")
        self.artistic_movements = _load_data_from_json("artistic_movements.json")
        self.visual_effects = _load_data_from_json("visual_effects.json")

        # III. Environnement & Atmosphère
        self.locations = _load_data_from_json("locations.json")
        self.time_of_day = _load_data_from_json("time_of_day.json")
        self.weather_conditions = _load_data_from_json("weather_conditions.json")
        self.mood_emotions = _load_data_from_json("mood_emotions.json")

        # IV. Composition & Perspective
        self.composition_arrangement = _load_data_from_json("composition_arrangement.json")
        self.perspectives = _load_data_from_json("perspectives.json")

        # V. Détails Techniques & Qualité
        self.resolutions_quality = _load_data_from_json("resolutions_quality.json")
        self.rendering_engines = _load_data_from_json("rendering_engines.json")
        self.lighting_options = _load_data_from_json("lighting_options.json")

        # VI. Univers & Références
        self.franchises_universes = _load_data_from_json("franchises_universes.json")
        self.artist_references = _load_data_from_json("artist_references.json")

        # VII. Prompts Négatifs
        self.undesired_elements = _load_data_from_json("undesired_elements.json")

    def create_tab(self, module_translations):
        self.module_translations = module_translations

        with gr.Tab(translate("image_prompt_builder_tab_name", self.module_translations)) as tab:
            gr.Markdown(f"## {translate('image_prompt_builder_tab_title', self.module_translations)}")

            with gr.Column():
                self.final_prompt_textbox = gr.Textbox(
                    label=translate("generated_prompt_label", self.module_translations),
                    info=translate("generated_prompt_info", self.module_translations),
                    lines=5,
                    interactive=True,
                    value="",
                    elem_id="final_prompt_output"
                )
                self.subject_input = gr.Textbox(
                    label=translate("subject_input_label", self.module_translations),
                    info=translate("subject_input_info", self.module_translations),
                    placeholder=translate("subject_input_placeholder", self.module_translations),
                    lines=1,
                )

                gr.Markdown(f"### {translate('prompt_helpers_title', self.module_translations)}")

                with gr.Accordion(translate("section_subject_actions_title", self.module_translations), open=False):
                    with gr.Row():
                        self.subject_character_types_dropdown = gr.Dropdown(
                            label=translate("subject_character_types_label", self.module_translations),
                            choices=self.subject_character_types,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.subject_creatures_entities_dropdown = gr.Dropdown(
                            label=translate("subject_creatures_entities_label", self.module_translations),
                            choices=self.subject_creatures_entities,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                    with gr.Row():
                        self.subject_clothing_styles_dropdown = gr.Dropdown(
                            label=translate("subject_clothing_styles_label", self.module_translations),
                            choices=self.subject_clothing_styles,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.subject_physical_attributes_dropdown = gr.Dropdown(
                            label=translate("subject_physical_attributes_label", self.module_translations),
                            choices=self.subject_physical_attributes,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                    with gr.Row():
                        self.subject_visual_supernatural_attributes_dropdown = gr.Dropdown(
                            label=translate("subject_visual_supernatural_attributes_label", self.module_translations),
                            choices=self.subject_visual_supernatural_attributes,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.actions_verbs_dropdown = gr.Dropdown(
                            label=translate("actions_verbs_label", self.module_translations),
                            choices=self.actions_verbs,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_visual_artistic_style_title", self.module_translations), open=False):
                    with gr.Row():
                        self.medium_technique_dropdown = gr.Dropdown(
                            label=translate("medium_technique_label", self.module_translations),
                            choices=self.medium_technique,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.artistic_movements_dropdown = gr.Dropdown(
                            label=translate("artistic_movements_label", self.module_translations),
                            choices=self.artistic_movements,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                    with gr.Row():
                        self.visual_effects_dropdown = gr.Dropdown(
                            label=translate("visual_effects_label", self.module_translations),
                            choices=self.visual_effects,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_environment_atmosphere_title", self.module_translations), open=False):
                    with gr.Row():
                        self.location_context_dropdown = gr.Dropdown(
                            label=translate("location_context_label", self.module_translations),
                            choices=self.locations,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.time_of_day_dropdown = gr.Dropdown(
                            label=translate("time_of_day_label", self.module_translations),
                            choices=self.time_of_day,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                    with gr.Row():
                        self.weather_conditions_dropdown = gr.Dropdown(
                            label=translate("weather_conditions_label", self.module_translations),
                            choices=self.weather_conditions,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.mood_emotion_dropdown = gr.Dropdown(
                            label=translate("mood_emotion_label", self.module_translations),
                            choices=self.mood_emotions,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_composition_perspective_title", self.module_translations), open=False):
                    with gr.Row():
                        self.composition_arrangement_dropdown = gr.Dropdown(
                            label=translate("composition_arrangement_label", self.module_translations),
                            choices=self.composition_arrangement,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.perspective_dropdown = gr.Dropdown(
                            label=translate("perspective_label", self.module_translations),
                            choices=self.perspectives,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_technical_quality_title", self.module_translations), open=False):
                    with gr.Row():
                        self.resolution_quality_dropdown = gr.Dropdown(
                            label=translate("resolution_quality_label", self.module_translations),
                            choices=self.resolutions_quality,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.rendering_engines_dropdown = gr.Dropdown(
                            label=translate("rendering_label", self.module_translations),
                            choices=self.rendering_engines,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                    with gr.Row():
                        self.lighting_options_dropdown = gr.Dropdown(
                            label=translate("lighting_label", self.module_translations),
                            choices=self.lighting_options,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_universe_references_title", self.module_translations), open=False):
                    with gr.Row():
                        self.franchises_universes_dropdown = gr.Dropdown(
                            label=translate("license_label", self.module_translations),
                            choices=self.franchises_universes,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                        self.artist_references_dropdown = gr.Dropdown(
                            label=translate("artist_references_label", self.module_translations),
                            choices=self.artist_references,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )

                with gr.Accordion(translate("section_negative_prompts_title", self.module_translations), open=False):
                    with gr.Row():
                        self.undesired_elements_dropdown = gr.Dropdown(
                            label=translate("undesired_elements_label", self.module_translations),
                            choices=self.undesired_elements,
                            multiselect=True,
                            value=[],
                            interactive=True
                        )
                
                # Button to clear all selections
                self.clear_button = gr.Button(
                    translate("clear_all_selections", self.module_translations),
                    variant="secondary"
                )

                # Random Prompt Button
                self.random_prompt_button = gr.Button(
                    translate("random_prompt_button", self.module_translations),
                    variant="primary"
                )

            # Event listeners for prompt generation
            all_input_components = [
                self.subject_input,
                self.subject_character_types_dropdown, self.subject_creatures_entities_dropdown,
                self.subject_clothing_styles_dropdown, self.subject_physical_attributes_dropdown,
                self.subject_visual_supernatural_attributes_dropdown, self.actions_verbs_dropdown,
                self.medium_technique_dropdown, self.artistic_movements_dropdown, self.visual_effects_dropdown,
                self.location_context_dropdown, self.time_of_day_dropdown, self.weather_conditions_dropdown, self.mood_emotion_dropdown,
                self.composition_arrangement_dropdown, self.perspective_dropdown,
                self.resolution_quality_dropdown, self.rendering_engines_dropdown, self.lighting_options_dropdown,
                self.franchises_universes_dropdown, self.artist_references_dropdown,
                self.undesired_elements_dropdown
            ]

            for comp in all_input_components:
                comp.change(
                    fn=self._update_prompt,
                    inputs=all_input_components,
                    outputs=self.final_prompt_textbox
                )
            
            # Clear all selections button functionality
            self.clear_button.click(
                fn=self._clear_all_selections,
                outputs=[
                    self.subject_input,
                    self.subject_character_types_dropdown, self.subject_creatures_entities_dropdown,
                    self.subject_clothing_styles_dropdown, self.subject_physical_attributes_dropdown,
                    self.subject_visual_supernatural_attributes_dropdown, self.actions_verbs_dropdown,
                    self.medium_technique_dropdown, self.artistic_movements_dropdown, self.visual_effects_dropdown,
                    self.location_context_dropdown, self.time_of_day_dropdown, self.weather_conditions_dropdown, self.mood_emotion_dropdown,
                    self.composition_arrangement_dropdown, self.perspective_dropdown,
                    self.resolution_quality_dropdown, self.rendering_engines_dropdown, self.lighting_options_dropdown,
                    self.franchises_universes_dropdown, self.artist_references_dropdown,
                    self.undesired_elements_dropdown,
                    self.final_prompt_textbox
                ]
            )

            # Random prompt button functionality
            self.random_prompt_button.click(
                fn=self._randomize_prompt_fields,
                outputs=[
                    self.subject_input,
                    self.subject_character_types_dropdown, self.subject_creatures_entities_dropdown,
                    self.subject_clothing_styles_dropdown, self.subject_physical_attributes_dropdown,
                    self.subject_visual_supernatural_attributes_dropdown, self.actions_verbs_dropdown,
                    self.medium_technique_dropdown, self.artistic_movements_dropdown, self.visual_effects_dropdown,
                    self.location_context_dropdown, self.time_of_day_dropdown, self.weather_conditions_dropdown, self.mood_emotion_dropdown,
                    self.composition_arrangement_dropdown, self.perspective_dropdown,
                    self.resolution_quality_dropdown, self.rendering_engines_dropdown, self.lighting_options_dropdown,
                    self.franchises_universes_dropdown, self.artist_references_dropdown,
                    # Note: undesired_elements_dropdown is intentionally excluded from randomization
                    self.final_prompt_textbox
                ]
            )


        return tab

    def _update_prompt(self, subject, subject_character_types, subject_creatures_entities,
                       subject_clothing_styles, subject_physical_attributes,
                       subject_visual_supernatural_attributes, actions_verbs,
                       medium_technique, artistic_movements, visual_effects,
                       location_context, time_of_day, weather_conditions, mood_emotion,
                       composition_arrangement, perspective,
                       resolution_quality, rendering_engines, lighting_options,
                       franchises_universes, artist_references,
                       undesired_elements):
        parts = []
        if subject:
            parts.append(subject.strip())
        
        # Add selected dropdown values for new subject sub-categories, ensuring they are lists
        if isinstance(subject_character_types, list) and subject_character_types:
            parts.extend(subject_character_types)
        if isinstance(subject_creatures_entities, list) and subject_creatures_entities:
            parts.extend(subject_creatures_entities)
        if isinstance(subject_clothing_styles, list) and subject_clothing_styles:
            parts.extend(subject_clothing_styles)
        if isinstance(subject_physical_attributes, list) and subject_physical_attributes:
            parts.extend(subject_physical_attributes)
        if isinstance(subject_visual_supernatural_attributes, list) and subject_visual_supernatural_attributes:
            parts.extend(subject_visual_supernatural_attributes)
        if isinstance(actions_verbs, list) and actions_verbs:
            parts.extend(actions_verbs)
        
        if isinstance(medium_technique, list) and medium_technique:
            parts.extend(medium_technique)
        if isinstance(artistic_movements, list) and artistic_movements:
            parts.extend(artistic_movements)
        if isinstance(visual_effects, list) and visual_effects:
            parts.extend(visual_effects)
            
        if isinstance(location_context, list) and location_context:
            parts.extend(location_context)
        if isinstance(time_of_day, list) and time_of_day:
            parts.extend(time_of_day)
        if isinstance(weather_conditions, list) and weather_conditions:
            parts.extend(weather_conditions)
        if isinstance(mood_emotion, list) and mood_emotion:
            parts.extend(mood_emotion)

        if isinstance(composition_arrangement, list) and composition_arrangement:
            parts.extend(composition_arrangement)
        if isinstance(perspective, list) and perspective:
            parts.extend(perspective)

        if isinstance(resolution_quality, list) and resolution_quality:
            parts.extend(resolution_quality)
        if isinstance(rendering_engines, list) and rendering_engines:
            parts.extend(rendering_engines)
        if isinstance(lighting_options, list) and lighting_options:
            parts.extend(lighting_options)
            
        if isinstance(franchises_universes, list) and franchises_universes:
            parts.extend(franchises_universes)
        if isinstance(artist_references, list) and artist_references:
            parts.extend(artist_references)

        if isinstance(undesired_elements, list) and undesired_elements:
            # Negative prompts usually go at the end, often with a different separator or structure
            # For simplicity, we'll just add them to the main prompt with a comma for now.
            # In a real scenario, you might want to separate positive and negative prompts.
            parts.extend(["NOT " + item for item in undesired_elements]) # Example: Add "NOT" for negative elements
            
        return ", ".join(parts)

    def _clear_all_selections(self):
        return (
            "", # subject_input
            [], # subject_character_types_dropdown
            [], # subject_creatures_entities_dropdown
            [], # subject_clothing_styles_dropdown
            [], # subject_physical_attributes_dropdown
            [], # subject_visual_supernatural_attributes_dropdown
            [], # actions_verbs_dropdown
            [], # medium_technique_dropdown
            [], # artistic_movements_dropdown
            [], # visual_effects_dropdown
            [], # location_context_dropdown
            [], # time_of_day_dropdown
            [], # weather_conditions_dropdown
            [], # mood_emotion_dropdown
            [], # composition_arrangement_dropdown
            [], # perspective_dropdown
            [], # resolution_quality_dropdown
            [], # rendering_engines_dropdown
            [], # lighting_options_dropdown
            [], # franchises_universes_dropdown
            [], # artist_references_dropdown
            [], # undesired_elements_dropdown
            ""  # final_prompt_textbox
        )

    def _randomize_prompt_fields(self):
        # Randomly select a subject if not already provided
        random_subject = random.choice(_load_data_from_json("random_subjects.json"))

        # Randomly select 0 to N items from each list, where N is a varied range
        def get_random_choices(choices_list, max_selections_factor=0.2, min_selections=0, max_selections_override=None):
            if not choices_list:
                return []
            
            num_available = len(choices_list)
            
            if max_selections_override is not None:
                max_to_select = max_selections_override
            else:
                max_to_select = max(1, int(num_available * max_selections_factor)) # At least 1 if factor is low

            num_choices = random.randint(min_selections, min(max_to_select, num_available))
            return random.sample(choices_list, num_choices)

        random_character_types = get_random_choices(self.subject_character_types, max_selections_factor=0.15, max_selections_override=3)
        random_creatures_entities = get_random_choices(self.subject_creatures_entities, max_selections_factor=0.15, max_selections_override=3)
        random_clothing_styles = get_random_choices(self.subject_clothing_styles, max_selections_factor=0.15, max_selections_override=3)
        random_physical_attributes = get_random_choices(self.subject_physical_attributes, max_selections_factor=0.15, max_selections_override=3)
        random_visual_supernatural_attributes = get_random_choices(self.subject_visual_supernatural_attributes, max_selections_factor=0.15, max_selections_override=3)
        random_actions_verbs = get_random_choices(self.actions_verbs, max_selections_factor=0.1, max_selections_override=2)
        random_medium_technique = get_random_choices(self.medium_technique, max_selections_factor=0.1, max_selections_override=2)
        random_artistic_movements = get_random_choices(self.artistic_movements, max_selections_factor=0.05, max_selections_override=1)
        random_visual_effects = get_random_choices(self.visual_effects, max_selections_factor=0.1, max_selections_override=3)
        random_locations = get_random_choices(self.locations, max_selections_factor=0.05, max_selections_override=2)
        random_time_of_day = get_random_choices(self.time_of_day, max_selections_factor=0.2, max_selections_override=1)
        random_weather_conditions = get_random_choices(self.weather_conditions, max_selections_factor=0.15, max_selections_override=1)
        random_mood_emotions = get_random_choices(self.mood_emotions, max_selections_factor=0.1, max_selections_override=2)
        random_composition_arrangement = get_random_choices(self.composition_arrangement, max_selections_factor=0.1, max_selections_override=2)
        random_perspectives = get_random_choices(self.perspectives, max_selections_factor=0.1, max_selections_override=2)
        random_resolutions_quality = get_random_choices(self.resolutions_quality, max_selections_factor=0.1, max_selections_override=3)
        random_rendering_engines = get_random_choices(self.rendering_engines, max_selections_factor=0.05, max_selections_override=1)
        random_lighting_options = get_random_choices(self.lighting_options, max_selections_factor=0.1, max_selections_override=2)
        random_franchises_universes = get_random_choices(self.franchises_universes, max_selections_factor=0.02, max_selections_override=1)
        random_artist_references = get_random_choices(self.artist_references, max_selections_factor=0.02, max_selections_override=1)

        # Call _update_prompt with the newly generated random values
        updated_prompt = self._update_prompt(
            random_subject,
            random_character_types, random_creatures_entities,
            random_clothing_styles, random_physical_attributes,
            random_visual_supernatural_attributes, random_actions_verbs,
            random_medium_technique, random_artistic_movements, random_visual_effects,
            random_locations, random_time_of_day, random_weather_conditions, random_mood_emotions,
            random_composition_arrangement, random_perspectives,
            random_resolutions_quality, random_rendering_engines, random_lighting_options,
            random_franchises_universes, random_artist_references,
            [] # undesired_elements is intentionally left empty for random generation
        )

        return (
            random_subject,
            random_character_types, random_creatures_entities,
            random_clothing_styles, random_physical_attributes,
            random_visual_supernatural_attributes, random_actions_verbs,
            random_medium_technique, random_artistic_movements, random_visual_effects,
            random_locations, random_time_of_day, random_weather_conditions, random_mood_emotions,
            random_composition_arrangement, random_perspectives,
            random_resolutions_quality, random_rendering_engines, random_lighting_options,
            random_franchises_universes, random_artist_references,
            updated_prompt
        )
