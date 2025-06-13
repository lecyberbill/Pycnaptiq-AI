/**
 * @file js_modules.js
 * @description
 * Ce fichier centralise les scripts JavaScript utilisés par les différents modules
 * de l'application CyberBill SDXL. L'objectif est de fournir un point d'entrée unique
 * pour le chargement et la gestion des fonctionnalités JavaScript côté client
 * qui interagissent avec l'interface Gradio et les opérations backend.
 *
 * Contexte d'utilisation :
 * - **cyberbill_SDXL.py** : Le script principal de l'application Gradio. Il peut
 *   charger ce fichier JavaScript globalement via les options de l'interface Gradio
 *   ou spécifiquement pour certains composants.
 * - **utils.py (fonction load_modules_js)** : Cette fonction dans les utilitaires Python
 *   est responsable de la lecture du contenu de ce fichier pour l'injecter
 *   dans l'interface Gradio.
 * - **Modules individuels (_mod.py)** : Les modules peuvent s'appuyer sur les fonctions
 *   définies ici pour des interactions courantes (ex: copier dans le presse-papier,
 *   afficher/masquer des éléments, déclencher des événements Gradio).
 *   La délégation d'événements est privilégiée pour gérer dynamiquement les éléments
 *   ajoutés par Gradio.
 */
() => {
            // --- Fonctions Utilitaires (locales) ---
            function toggleMeta(id) {
                // Vérifie si l'ID est fourni
                if (!id) {
                    console.error("toggleMeta appelé sans ID");
                    return;
                }
                const meta = document.getElementById(id);
                if (meta) {
                    // Ajoute ou retire la classe 'active' pour afficher/masquer
                    meta.classList.toggle("active");
                } else {
                    // Avertit si l'élément overlay n'est pas trouvé
                    console.error("Élément overlay non trouvé pour l'ID:", id);
                }
            }
            
            function copyToClipboard(textId) {
                if (!textId) {
                    console.error("copyToClipboard appelé sans ID");
                    return;
                }

                const textElement = document.getElementById(textId);
                if (textElement) {
                    const text = textElement.innerText;

                    navigator.clipboard.writeText(text).then(() => {
                        alert("Prompt copié !");

                        const gradioInput = document.querySelector('#promt_input textarea');
                        if (gradioInput) {
                            // Focus pour activer le champ
                            gradioInput.focus();

                            // Efface l'ancien texte
                            gradioInput.value = '';
                            gradioInput.dispatchEvent(new Event('input', { bubbles: true }));

                            // Simule la frappe lettre par lettre
                            let index = 0;
                            const typeLetter = () => {
                                if (index < text.length) {
                                    gradioInput.value += text[index];
                                    gradioInput.dispatchEvent(new Event('input', { bubbles: true }));
                                    index++;
                                    setTimeout(typeLetter, 3); // vitesse de "frappe"
                                }
                            };
                            typeLetter();

                        } else {
                            console.warn("Champ Gradio avec elem_id='promt_input' non trouvé.");
                        }

                    }).catch(err => {
                        console.error("Erreur lors de la copie dans le presse-papiers:", err);
                        alert("Erreur lors de la copie.");
                    });
                } else {
                    console.error("Élément texte non trouvé pour l'ID:", textId);
                }
            }
            // --- Fonctions (civitai_downloader_mod.py) ---

            function handleCivitaiDetailsClick(event) {
                const button = event.target.closest('.civitai-downloader-details-btn');
                if (!button) {
                    return; // Click was not on a details button or its child
                }

                const modelName = button.getAttribute('data-model-name');
                if (!modelName) {
                    console.error("Civitai Downloader JS: data-model-name attribute not found on button.");
                    return;
                }

                // Find the hidden Gradio textbox that acts as a trigger
                // Gradio Textbox renders a textarea inside its main div.
                const triggerTextbox = document.querySelector('#civitai_downloader_hidden_model_trigger textarea');

                if (triggerTextbox) {
                    triggerTextbox.value = modelName;
                    // Dispatch an 'input' event to notify Gradio of the change
                    const inputEvent = new Event('input', { bubbles: true });
                    triggerTextbox.dispatchEvent(inputEvent);

                    // Scroll to the details section
                    const detailsSection = document.getElementById('civitai_downloader_details_container');
                    if (detailsSection) {
                        // Using 'start' to align the top of the element with the top of the visible area of the scrollable ancestor.
                        detailsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    } else {
                        console.warn("Civitai Downloader JS: Details section with ID 'civitai_downloader_details_container' not found for scrolling.");
                    }
                } else {
                    console.error("Civitai Downloader JS: Hidden trigger textbox with ID 'civitai_downloader_hidden_model_trigger' (or its textarea) not found.");
                }
            }
            // --- Délégation d'Événements (civitai_downloader_mod.py) ---
            console.log("Civitai Downloader: Initializing JS event listeners...");
            document.body.addEventListener('click', handleCivitaiDetailsClick);
            console.log("Civitai Downloader: JS event listeners initialized.");

            // --- Délégation d'Événements ---
            // Ajoute un seul écouteur sur un parent stable (body)
            console.log("Initialisation des écouteurs d'événements JS pour la galerie Gradio...");
            document.body.addEventListener('click', function(event) {
                // Trouve le bouton le plus proche correspondant aux sélecteurs
                const metaButton = event.target.closest('.meta-button');
                const closeButton = event.target.closest('.close-btn');
                const copyButton = event.target.closest('.copy-btn');

                // Agit en fonction du bouton cliqué
                if (metaButton) {
                    // Récupère l'ID de l'overlay depuis l'attribut data-*
                    const overlayId = metaButton.getAttribute('data-overlay-id');
                    toggleMeta(overlayId);
                } else if (closeButton) {
                    // Récupère l'ID de l'overlay depuis l'attribut data-*
                    const overlayId = closeButton.getAttribute('data-overlay-id');
                    toggleMeta(overlayId); // Le bouton fermer utilise la même fonction
                } else if (copyButton) {
                    // Récupère l'ID du texte à copier depuis l'attribut data-*
                    const textId = copyButton.getAttribute('data-text-id');
                    copyToClipboard(textId);
                }
            });
            console.log("Écouteurs d'événements JS prêts.");

        }