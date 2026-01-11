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
    document.body.addEventListener('click', function (event) {
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

    // --- OpenPose Interactive Editor ---

    class OpenPoseEditor {
        constructor(canvasId, suffix = "") {
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) return;
            this.suffix = suffix;
            this.ctx = this.canvas.getContext('2d');
            this.bgImage = null; // Background image
            this.people = [];
            this.selectedJoint = null;
            this.draggedPerson = null;
            this.draggedJoint = null;

            // COCO Keypoints structure (18 joints)
            this.jointLabels = ["Nose", "Neck", "R-Sho", "R-Elb", "R-Wri", "L-Sho", "L-Elb", "L-Wri", "R-Hip", "R-Kne", "R-Ank", "L-Hip", "L-Kne", "L-Ank", "R-Eye", "L-Eye", "R-Ear", "L-Ear"];

            // Limbs (Connections between joint indices)
            this.limbs = [
                [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
            ];

            // Colors per limb (Standard OpenPose palette)
            this.colors = [
                [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
            ];

            this.initEvents();
            this.addPerson(); // Initial person
            this.render();
        }

        addPerson() {
            const cx = this.canvas.width / 2;
            const cy = this.canvas.height / 2;
            // Standard standing pose keypoints
            const joints = [
                [cx, cy - 100], [cx, cy - 60], [cx - 30, cy - 50], [cx - 40, cy], [cx - 45, cy + 40],
                [cx + 30, cy - 50], [cx + 40, cy], [cx + 45, cy + 40], [cx - 20, cy + 20], [cx - 20, cy + 70],
                [cx - 20, cy + 120], [cx + 20, cy + 20], [cx + 20, cy + 70], [cx + 20, cy + 120],
                [cx - 5, cy - 105], [cx + 5, cy - 105], [cx - 15, cy - 105], [cx + 15, cy - 105]
            ];
            this.people.push({ joints: joints });
            this.render();
        }

        resetPose() {
            this.people = [];
            this.addPerson();
        }

        clear() {
            this.people = [];
            this.render();
        }

        initEvents() {
            this.canvas.addEventListener('mousedown', (e) => {
                const { x, y } = this.getMousePos(e);
                // Find closest joint
                let found = false;
                for (let pIdx = 0; pIdx < this.people.length; pIdx++) {
                    const person = this.people[pIdx];
                    for (let jIdx = 0; jIdx < person.joints.length; jIdx++) {
                        const joint = person.joints[jIdx];
                        const dist = Math.hypot(joint[0] - x, joint[1] - y);
                        if (dist < 15) {
                            this.draggedPerson = person;
                            this.draggedJoint = jIdx;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            });

            window.addEventListener('mousemove', (e) => {
                if (this.draggedPerson && this.draggedJoint !== null) {
                    const { x, y } = this.getMousePos(e);
                    this.draggedPerson.joints[this.draggedJoint] = [x, y];
                    this.render();
                }
            });

            window.addEventListener('mouseup', () => {
                this.draggedPerson = null;
                this.draggedJoint = null;
            });
        }

        getMousePos(e) {
            const rect = this.canvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
                y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
            };
        }

        render() {
            this.ctx.fillStyle = 'black';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Draw Background Image if any
            if (this.bgImage) {
                // simple contain fit
                const wrate = this.canvas.width / this.bgImage.width;
                const hrate = this.canvas.height / this.bgImage.height;
                const ratio = Math.min(wrate, hrate);
                const nw = this.bgImage.width * ratio;
                const nh = this.bgImage.height * ratio;
                const nx = (this.canvas.width - nw) / 2;
                const ny = (this.canvas.height - nh) / 2;
                this.ctx.drawImage(this.bgImage, nx, ny, nw, nh);
            }

            this.people.forEach(person => {
                // Draw Limbs
                this.limbs.forEach((limb, i) => {
                    const start = person.joints[limb[0]];
                    const end = person.joints[limb[1]];
                    const color = this.colors[i % this.colors.length];

                    this.ctx.beginPath();
                    this.ctx.moveTo(start[0], start[1]);
                    this.ctx.lineTo(end[0], end[1]);
                    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    this.ctx.lineWidth = 4;
                    this.ctx.stroke();
                });

                // Draw Joints
                person.joints.forEach((joint, i) => {
                    this.ctx.beginPath();
                    this.ctx.arc(joint[0], joint[1], i < 14 ? 5 : 3, 0, Math.PI * 2);
                    this.ctx.fillStyle = i < 14 ? 'white' : 'gray';
                    this.ctx.fill();
                });
            });
        }

        apply() {
            const dataUrl = this.canvas.toDataURL('image/png');
            console.log(`[OpenPose] Applying pose for suffix "${this.suffix}". Data URL length: ${dataUrl.length}`);

            const bufferElem = document.getElementById(`openpose_data_buffer${this.suffix}`);
            if (!bufferElem) {
                console.error(`[OpenPose] FATAL: Buffer element "openpose_data_buffer${this.suffix}" NOT FOUND in DOM.`);
                return;
            }

            let bufferTextarea = bufferElem.tagName === 'TEXTAREA' ? bufferElem : bufferElem.querySelector('textarea');
            if (bufferTextarea) {
                bufferTextarea.value = dataUrl;
                // Dispatch multiple events for maximum coverage
                bufferTextarea.dispatchEvent(new Event('input', { bubbles: true }));
                bufferTextarea.dispatchEvent(new Event('change', { bubbles: true }));

                setTimeout(() => {
                    let triggerBtn = document.getElementById(`openpose_apply_trigger${this.suffix}`);
                    if (!triggerBtn) {
                        console.error(`[OpenPose] Trigger element "openpose_apply_trigger${this.suffix}" NOT FOUND.`);
                        return;
                    }

                    // Agressively find the button
                    if (triggerBtn.tagName !== 'BUTTON') {
                        const inner = triggerBtn.querySelector('button');
                        if (inner) triggerBtn = inner;
                    }

                    if (triggerBtn) {
                        triggerBtn.click();
                    } else {
                        console.error(`[OpenPose] Could not find clickable button for trigger.`);
                    }
                }, 300);
            } else {
                console.error(`[OpenPose] Textarea NOT FOUND inside buffer element.`);
            }
        }

        setBackground(dataUrl) {
            if (!dataUrl || dataUrl.trim() === "" || dataUrl === "null") {
                this.bgImage = null;
                this.render();
                return;
            }
            const img = new Image();
            img.onload = () => {
                this.bgImage = img;
                this.render();
            };
            img.onerror = (e) => {
                console.error("[OpenPose] Failed to load background image guide.", e);
                this.bgImage = null;
                this.render();
            };
            img.src = dataUrl;
        }
    }

    let opEditors = {};

    function initOpEditor(suffix) {
        if (!opEditors[suffix]) {
            opEditors[suffix] = new OpenPoseEditor(`openpose_canvas${suffix}`, suffix);

            // Background Observer
            const bgBuffer = document.getElementById(`openpose_bg_buffer${suffix}`);
            if (bgBuffer) {
                const textarea = bgBuffer.querySelector('textarea');
                if (textarea) {
                    // Gradio updates usually happen by setting the value property.
                    // We can observe the value by wrapping the setter or using an interval as fallback,
                    // but MutationObserver on 'value' attribute might not work if it's set via property.
                    // Let's use a simple interval for the background buffer check once initialized.
                    setInterval(() => {
                        const currentVal = textarea.value || "";
                        if (opEditors[suffix] && currentVal !== (opEditors[suffix].lastBgValue || "")) {
                            opEditors[suffix].lastBgValue = currentVal;
                            opEditors[suffix].setBackground(currentVal);
                        }
                    }, 500);
                }
            }
        }
    }

    document.body.addEventListener('click', function (event) {
        // OpenPose Editor Toggles and Actions
        const btnOpen = event.target.closest('[id^="openpose_editor_btn_main"]');
        if (btnOpen) {
            const suffix = btnOpen.id.replace('openpose_editor_btn_main', '');
            const container = document.getElementById(`openpose_container${suffix}`);
            if (container) {
                const isHidden = container.style.display === 'none';
                container.style.display = isHidden ? 'block' : 'none';
                if (isHidden) {
                    initOpEditor(suffix);
                }
            }
        }

        if (event.target.id.startsWith('op_apply')) {
            const suffix = event.target.id.replace('op_apply', '');
            if (opEditors[suffix]) opEditors[suffix].apply();
        } else if (event.target.id.startsWith('op_add')) {
            const suffix = event.target.id.replace('op_add', '');
            if (opEditors[suffix]) opEditors[suffix].addPerson();
        } else if (event.target.id.startsWith('op_reset')) {
            const suffix = event.target.id.replace('op_reset', '');
            if (opEditors[suffix]) opEditors[suffix].resetPose();
        } else if (event.target.id.startsWith('op_clear')) {
            const suffix = event.target.id.replace('op_clear', '');
            if (opEditors[suffix]) opEditors[suffix].clear();
        }
    });

    // Special check for background buffer changes even if container is hidden initially
    function setupBgObservers() {
        const bgBuffers = document.querySelectorAll('[id^="openpose_bg_buffer"]');
        if (bgBuffers.length === 0 && !window.op_warned) {
            console.warn("[OpenPose] Critical: No background buffers found in DOM.");
            window.op_warned = true;
        }
        bgBuffers.forEach(buf => {
            const suffix = buf.id.replace('openpose_bg_buffer', '');
            const textarea = buf.querySelector('textarea');
            if (textarea && textarea.value && textarea.value.trim() !== "" && textarea.value.startsWith('data:image')) {
                if (!opEditors[suffix]) {
                    initOpEditor(suffix);
                } else if (textarea.value !== (opEditors[suffix].lastBgValue || "")) {
                    opEditors[suffix].lastBgValue = textarea.value;
                    opEditors[suffix].setBackground(textarea.value);
                }
            }
        });
    }
    // Periodic check because of Gradio dynamic loading
    setInterval(setupBgObservers, 2000);

    console.log("Système OpenPose prêt.");
}