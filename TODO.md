# 📋 TODO List - Pycnaptiq-AI

Voici une liste d'idées et d'améliorations pour les prochaines versions de **Pycnaptiq-AI**.

## 🛠️ Robustesse et Qualité de Vie (QoL)
- [x] **Mise en place de l'infrastructure de tests automatisés**
    - Création du dossier `tests/` et `conftest.py`.
    - Implémentation de tests unitaires pour `core/config.py`, `Utils/preset_handlers.py`, et `core/sdxl_logic.py`.
    - **Pour lancer les tests :** `.\venv\Scripts\python.exe -m pytest tests/`
- [X] **Gestion des Erreurs UI** : Ajouter des notifications (toasts) dans Gradio pour les erreurs de chargement ou de génération.
- [x] **Logs Améliorés** : Système de logging structuré (console couleur + rotation de fichiers 5Mo).

## ✨ Nouvelles Fonctionnalités Créatives
- [x] **ControlNet / IP-Adapter** : Intégrer le contrôle précis des formes et des styles pour SDXL.
- [ ] **Génération Vidéo** : Ajouter des modules pour la vidéo (SVD, Mochi).
- [x] **Upscale par Tiles** : Permettre une mise à l'échelle très haute résolution (4K+) sans saturation de la VRAM.

## 🚀 Optimisation Technique
- [ ] **Quantisation (GGUF/EXL2)** : Supporter les modèles compressés pour réduire l'empreinte VRAM.
- [ ] **API Headless** : Permettre d'utiliser l'application comme un serveur distant via une API REST.
- [ ] **Inférence Turbo** : Intégrer des optimisations comme TensorRT pour accélérer la génération sur Nvidia.

## 📦 Écosystème et Distribution
- [ ] **Gestionnaire d'Extensions** : Interface pour installer de nouveaux modules depuis GitHub/Civitai.
- [ ] **Dockerisation** : Créer un conteneur pour simplifier l'installation multi-plateforme.
- [ ] **Documentation Développeur** : Créer un guide pour aider les autres à créer leurs propres modules.

---
*Dernière mise à jour : 2026-01-07*
