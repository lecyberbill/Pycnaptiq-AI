# cyberbill générateur d'image 🚀

## 📌 Prérequis
- **CUDA 12.6** installé ✅
- **Carte Nvidia RTX** : Non testé sur d'autres cartes.
- **8 Go de VRAM recommandés** : Optimisation non disponible pour les petites cartes graphiques.

## 📥 Installation
1. **Téléchargez le projet**  
   - Choisissez le fichier `zip` ou `cyberbill_SDXL.zip` et décompressez-le dans le répertoire de votre choix.

2. **Installez CUDA 12.6** via [ce lien](https://developer.nvidia.com/cuda-12-6-0-download-archive).

3. **Lancez le script `install.bat`**  
   - Cela configure l'environnement nécessaire pour le logiciel.

4. **Démarrez l'application avec `start.bat`**  
   - Double-cliquez sur ce fichier pour lancer l'interface utilisateur.


## ▶️ Lancement de l’application
1. **Double-cliquez sur `start.bat`** 🎯  
   - Cela active l’environnement et lance l’application.
2. **Ou utilisez la ligne de commande :**
   ```sh
   venv\Scripts\activate
   python cyberbill_SDXL.py

   
   
## ▶️ Utilisation
### 🌟 Étapes essentielles
1. **Charger un modèle SDXL**  
   - Placez les fichiers `.safetensors` dans `models/checkpoints`.
   - Cliquez sur **"Lister les modèles"**, puis sélectionnez le modèle souhaité.

   NOTE : Le logiciel est fourni **sans modèle**.
	- Si aucun modèle n'est trouvé au lancement, le programme vous demandera si vous souhaitez en charger un. Répondez par o ou n (oui ou non). Le modèle sera alors chargé. Il s'agit d'un modèle générique qui donne de bons résultats : MegaChonkXL.
	- Par ailleurs, vous pouvez télécharger vos propres modèles sur différentes sources (modèle checkpoints SDXL 1.0 au format .safetensors à placer dans le répertoire `/models/checkpoints`).

		Exemples de sites : [civitai.com](https://civitai.com/) | [lexica.art](https://lexica.art/) | [huggingface.co](https://huggingface.co)

2. **Configurer vos paramètres**  
   - **VAE** :
     - Placez vos fichiers `.safetensors` dans `/models/vae/`. Les fichiers dans leur grande majorité sont fournis avec un VAE intégré, il est généralement pas necessaire d'en télécharger un... mais au cas où !
     - Le VAE transforme l'image latente en une version complète et détaillée.
   - **Sampler** :
     - Sélectionnez un algorithme pour guider la génération de l'image (Euler, DDIM, etc.).
   - **Guidage** :
     - Détermine la fidélité de l'image au prompt :
       - *3-7* : Résultats créatifs.
       - *10-20* : Résultats précis.
   - **Étapes** :
     - Recommandé : environ 30 pour un équilibre qualité/vitesse.
   - **Seed** :
     - Utilisez -1 pour un seed aléatoire ou définissez un seed fixe pour reproduire des résultats.
   - **Dimensions** :
     - Sélectionnez un format prédéfini compatible avec le modèle.
   - **Nombre d'images** :
     - Sélectionnez le nombre d'image à générer.

3. **Ajouter un prompt**
   - Entrez un texte décrivant l'image souhaitée.
   - Activez "Traduire en anglais" pour automatiser la traduction.
   - En cochant la case générer un prompt à partir d'une image, vous pouvez coller ou uploader une image à partir de votre disque, et un prompt sera alors proposé.

4. **Générer des images**
   - Cliquez sur **"Générer"**. Les images sont enregistrées dans le dossier `output` avec un rapport HTML.

## Capture de l'interface

Le générateur d'image, prompt calculé à partir de l'image, ajout d'un lora
![image](https://github.com/user-attachments/assets/7e03b05f-028b-46e6-93de-ab2b7ce12280)

L'Inpainting, définir une zone de l'image à modifier, ici un visage d'une peronne de 80 ans à la place d'une jeune femme
![image](https://github.com/user-attachments/assets/d60b8d1b-8e77-4988-abe7-3f81ca0f4a34)

[MODULE] Amélioration d'image avec AuraSR (https://github.com/fal-ai/aura-sr)
![image](https://github.com/user-attachments/assets/4f188555-de5b-47ca-ae07-e24083894eef)

[MODULE] Cicitai browser 
![image](https://github.com/user-attachments/assets/506ab5fa-eacd-4f9b-be93-2c35b157cbc6)

[MODULE] Retouche d'image
![image](https://github.com/user-attachments/assets/2e31935f-8f0d-445d-a123-9784033f7042)

[MODULE] Image to Image (ici prompt et style sélectionné)
![image](https://github.com/user-attachments/assets/a3493385-5b48-40eb-82e8-75932d540253)

[MODULE] Remove Background basé sur RemBG https://github.com/danielgatis/rembg 
![iamge](https://github.com/user-attachments/assets/15717a23-9828-4e14-8a78-465110b22f76)


## ▶️ Configuration avancée

### 🌟 Fichier de configuration : `config.json`

Le fichier `config.json`, situé dans le dossier `/config`, permet de personnaliser les paramètres principaux de l'application. Voici une version détaillée :

```json
{
    "AUTHOR": "Cyberbill_SDXL",
    "MODELS_DIR": "models\\checkpoints",
    "VAE_DIR": "models\\vae",
    "INPAINT_MODELS_DIR": "models\\inpainting",
    "LORAS_DIR": "models\\loras",
    "SAVE_DIR": "output",
    "IMAGE_FORMAT": "webp",
    "DEFAULT_MODEL": "your_default_modele.safetensors",
    "NEGATIVE_PROMPT": "deformed, ugly, blurry, pixelated, grainy, poorly drawn, artifacts, errors, duplicates, missing, inconsistent, unrealistic, bad anatomy, severed hands, severed heads, crossed eyes, poor quality, low resolution, washed out, overexposed, underexposed, noise, flat, lacking details, generic, amateur",
    "FORMATS": [
        {"dimensions": "704*1408", "orientation": "Portrait"},
        {"dimensions": "704*1344", "orientation": "Portrait"},
        {"dimensions": "768*1344", "orientation": "Portrait"},
        {"dimensions": "768*1280", "orientation": "Portrait"},
        {"dimensions": "832*1216", "orientation": "Portrait"},
        {"dimensions": "832*1152", "orientation": "Portrait"},
        {"dimensions": "896*1152", "orientation": "Portrait"},
        {"dimensions": "896*1088", "orientation": "Portrait"},
        {"dimensions": "960*1088", "orientation": "Portrait"},
        {"dimensions": "960*1024", "orientation": "Portrait"},
        {"dimensions": "1024*1024", "orientation": "Carré"},
        {"dimensions": "1024*960", "orientation": "Paysage"},
        {"dimensions": "1088*960", "orientation": "Paysage"},
        {"dimensions": "1088*896", "orientation": "Paysage"},
        {"dimensions": "1408*704", "orientation": "Paysage"},
        {"dimensions": "1344*704", "orientation": "Paysage"},
        {"dimensions": "1344*768", "orientation": "Paysage"},
        {"dimensions": "1280*768", "orientation": "Paysage"},
        {"dimensions": "1216*832", "orientation": "Paysage"},
        {"dimensions": "1152*832", "orientation": "Paysage"},
        {"dimensions": "1152*896", "orientation": "Paysage"}
    ],
    "OPEN_BROWSER": "Yes",
    "GRADIO_THEME": "Defaut",
    "SHARE": "False",
    "LANGUAGE": "en"
}
```
### 🛠️ Champs principaux :

- **`AUTHOR`** : Nom ou auteur du fichier de configuration.
- **`MODELS_DIR`** : Répertoire où sont stockés les modèles de base SDXL.
- **`VAE_DIR`** : Emplacement pour les VAE personnalisés.
- **`INPAINT_MODELS_DIR`** : Chemin vers les modèles dédiés à l'inpainting.
- **`LORAS_DIR`** : Emplacement pour charger les fichiers LoRA au format `.safetensors`.
- **`SAVE_DIR`** : Dossier où sont sauvegardées les images générées.
- **`IMAGE_FORMAT`** : Format des fichiers image : `webp`, `jpeg`, ou `png`.
- **`DEFAULT_MODEL`** : Modèle chargé par défaut au démarrage.
- **`NEGATIVE_PROMPT`** : Prompt négatif générique appliqué par défaut, utile pour exclure des éléments indésirables dans les résultats générés.
- **`FORMATS`** : Dimensions des images, spécifiées en multiples de 4, avec des orientations comme `Portrait`, `Carré`, et `Paysage`.
- **`OPEN_BROWSER`** :  
  - `Yes` ouvre l'application directement dans le navigateur par défaut.  
  - `No` désactive l'ouverture automatique du navigateur.
- **`GRADIO_THEME`** : Personnalisez l'apparence de l'interface utilisateur grâce aux thèmes disponibles.
- **`SHARE`** :  
  - `True` permet de partager l'application en ligne via Gradio.  
  - `False` limite l'utilisation au local.
- **`LANGUAGE`** : Langue de l'interface utilisateur (`en` pour anglais, `fr` pour français).

### 🌟 Options supplémentaires en détail

- **`FORMATS`** : Détermine les dimensions des images. Chaque option doit respecter des multiples de 4 pour assurer une compatibilité optimale.  
  - **Exemple** :  
    - Portrait : `704*1408`, `768*1280`  
    - Carré : `1024*1024`  
    - Paysage : `1408*704`, `1280*768`

- **`OPEN_BROWSER`** :  
  - `Yes` : Ouvre l'application directement dans le navigateur par défaut.  
  - `No` : Désactive l'ouverture automatique du navigateur.

- **`GRADIO_THEME`** : Définit l'apparence de l'interface utilisateur.  
  - **Thèmes disponibles** :  
    - `Base` : Minimaliste avec une couleur primaire bleue.  
    - `Default` : Thème par défaut (orange et gris).  
    - `Origin` : Inspiré des versions classiques de Gradio.  
    - `Citrus` : Jaune vibrant avec des effets 3D sur les boutons.  
    - `Monochrome` : Noir et blanc avec un style classique.  
    - `Soft` : Tons violets avec des bords arrondis.  
    - `Glass` : Effet visuel "verre" avec des dégradés bleus.  
    - `Ocean` : Tons bleu-vert avec des transitions horizontales.

- **`SHARE`** :  
  - `True` : Permet de partager l'application en ligne via Gradio.  
  - `False` : Restreint l'application à un usage local uniquement.

- **`LANGUAGE`** : Définit la langue utilisée dans l'interface utilisateur.  
  - `en` : Anglais  
  - `fr` : Français

NOTE : 
C:\dossier\de\modeles
Vous devrez l'écrire comme ceci :
C:\\\dossier\\\de\\\modeles
pour c:\repertoire\mes_modeles\checkpoints il faudra écrire c:\\\repertoire\\\mes_modeles\\\checkpoints

 

## Savoir plus sur le choix des Samplers :
    EulerDiscreteScheduler (Rapide et détaillé): Un sampler Euler classique, rapide et qui produit des images détaillées. Bon point de départ et souvent utilisé pour son efficacité. Vous l'avez déjà.
    DDIMScheduler (Rapide et créatif): DDIM (Denoising Diffusion Implicit Models) est plus rapide que les méthodes classiques et peut être plus créatif, offrant parfois des résultats plus variés et surprenants. Peut être un bon choix pour l'exploration rapide.
    DPMSolverMultistepScheduler (Rapide et de haute qualité): Une version optimisée et plus rapide des solveurs DPM. Offre un bon compromis entre vitesse et qualité d'image, souvent considéré comme un des meilleurs choix pour la vitesse sans sacrifier trop la qualité.

Samplers de Haute Qualité et Photorealistic (pour un rendu détaillé et réaliste):

    DPM++ 2M Karras (Photoréaliste et détaillé): Un sampler très performant pour obtenir des images photoréalistes et très détaillées. Le "Karras" indique l'utilisation d'un schéma de bruitage amélioré (Karras noise schedule) qui améliore la qualité. Vous l'avez déjà et c'est un excellent choix.
    PNDMScheduler (Stable et photoréaliste): PNDM (Pseudo Numerical Methods for Diffusion Models) est stable et tend à produire des images photoréalistes avec moins de bruit. Peut être un bon choix si vous recherchez un rendu plus propre.
    DPM++ SDE Karras (Photoréaliste et avec réduction du bruit): Combine les avantages de DPM++ avec une méthode SDE (Stochastic Differential Equations) et le bruitage Karras. Très efficace pour réduire le bruit et obtenir un rendu photoréaliste de haute qualité.
    DPM++ 2M SDE Karras (Combine photoréalisme et réduction du bruit): Une autre variante de DPM++ SDE Karras qui combine photoréalisme et réduction du bruit, possiblement avec des caractéristiques légèrement différentes de la version simple DPM++ SDE Karras.
    KDPM2DiscreteScheduler (Détaillé et net): Une autre variante de KDPM qui tend à produire des images très détaillées et nettes. Bon choix si vous recherchez la précision.

Samplers Artistiques et Fluides (pour un rendu plus pictural ou stylisé):

    Euler Ancestral (Artistique et fluide): Un sampler Euler Ancestral qui produit des images plus fluides et artistiques. "Ancestral" signifie qu'il ajoute du bruit à chaque étape de débruitage, ce qui peut donner un aspect plus pictural. Vous l'avez déjà et c'est un bon choix pour des styles artistiques.
    KDPM2AncestralDiscreteScheduler (Artistique et net): Combine les caractéristiques de KDPM2 (détaillé et net) avec l'approche Ancestral (artistique). Peut offrir un bon compromis entre détail et style artistique.
    HeunDiscreteScheduler (Bon compromis vitesse/qualité): Heun est un sampler qui essaie de trouver un bon équilibre entre vitesse et qualité, et peut parfois produire des résultats avec un aspect plus doux ou "peint".
    LMSDiscreteScheduler (Équilibré et polyvalent): LMS (Linear Multistep Method) est un sampler plus polyvalent qui peut donner de bons résultats dans divers styles. Il est souvent considéré comme un bon choix général, ni trop rapide ni trop lent, ni trop spécialisé dans un style particulier.

## Liste des Samplers pour la Génération d'Images

Cette section décrit les différents samplers disponibles pour la génération d'images dans votre outil.  Le choix du sampler peut grandement influencer le style, la qualité et la vitesse de génération de l'image.

---

### Samplers Rapides et Efficaces

Ces samplers sont idéaux pour les itérations rapides, les tests, ou les systèmes moins puissants. Ils offrent une bonne vitesse de génération d'image.

*   **EulerDiscreteScheduler (Rapide et détaillé):**  Sampler Euler classique, connu pour sa rapidité et sa capacité à produire des images détaillées. Un bon point de départ et souvent utilisé pour son efficacité.

*   **DDIMScheduler (Rapide et créatif):**  DDIM (Denoising Diffusion Implicit Models) est plus rapide que les méthodes classiques et peut être plus créatif, offrant des résultats variés et parfois surprenants.  Bon pour l'exploration rapide et la génération d'images originales.

*   **DPMSolverMultistepScheduler (Rapide et de haute qualité):** Version optimisée et rapide des solveurs DPM. Offre un excellent compromis entre vitesse et qualité d'image. Souvent considéré comme l'un des meilleurs choix pour une génération rapide sans trop sacrifier la qualité.

---

### Samplers de Haute Qualité et Photoréalistes

Ces samplers sont conçus pour produire des images de la plus haute qualité, avec un rendu photoréaliste et très détaillé. Ils peuvent être plus lents, mais offrent un niveau de détail et de réalisme supérieur.

*   **DPM++ 2M Karras (Photoréaliste et détaillé):** Sampler très performant pour obtenir des images photoréalistes et extrêmement détaillées.  L'indication "Karras" signifie qu'il utilise un schéma de bruitage amélioré (Karras noise schedule) qui optimise la qualité de l'image.  Excellent choix pour le photoréalisme.

*   **PNDMScheduler (Stable et photoréaliste):**  PNDM (Pseudo Numerical Methods for Diffusion Models) est stable et tend à générer des images photoréalistes avec moins de bruit.  Bon choix si vous recherchez un rendu plus propre et réaliste.

*   **DPM++ SDE Karras (Photoréaliste et avec réduction du bruit):** Combine les avantages de DPM++ avec une méthode SDE (Stochastic Differential Equations) et le bruitage Karras. Très efficace pour réduire le bruit et obtenir un rendu photoréaliste de très haute qualité.

*   **DPM++ 2M SDE Karras (Combine photoréalisme et réduction du bruit):** Variante de DPM++ SDE Karras qui combine également photoréalisme et réduction du bruit. Peut présenter des nuances légèrement différentes par rapport à la version simple DPM++ SDE Karras.

*   **KDPM2DiscreteScheduler (Détaillé et net):** Variante de KDPM qui produit des images très détaillées et nettes. Idéal si la précision et la netteté des détails sont primordiales.

---

### Samplers Artistiques et Fluides

Ces samplers sont plus orientés vers un rendu artistique, pictural ou stylisé. Ils peuvent produire des images avec un aspect plus doux, fluide ou "peint".

*   **Euler Ancestral (Artistique et fluide):** Sampler Euler Ancestral qui génère des images plus fluides et avec un aspect artistique. L'approche "Ancestral" ajoute du bruit à chaque étape de débruitage, ce qui contribue à un rendu plus pictural.  Excellent pour les styles artistiques et créatifs.

*   **KDPM2AncestralDiscreteScheduler (Artistique et net):** Combine les caractéristiques de KDPM2 (détaillé et net) avec l'approche Ancestral (artistique). Offre un bon équilibre entre détails précis et style artistique.

*   **HeunDiscreteScheduler (Bon compromis vitesse/qualité):** Sampler Heun qui cherche un bon équilibre entre vitesse et qualité. Peut produire des résultats avec un aspect plus doux ou "peint".  Un bon choix polyvalent pour différents styles.

*   **LMSDiscreteScheduler (Équilibré et polyvalent):** LMS (Linear Multistep Method) est un sampler polyvalent qui peut donner de bons résultats dans divers styles d'images.  Considéré comme un bon choix général, ni trop rapide ni trop spécialisé dans un style particulier.

---

### Samplers "Abrégés" ou Variantes

Ces samplers sont souvent des versions abrégées ou des variantes d'autres samplers, offrant des comportements similaires ou légèrement modifiés.

*   **Euler A (Euler Ancestral, version abrégée):**  Raccourci pour Euler Ancestral. Se comporte de manière très similaire à Euler Ancestral et peut être utilisé de manière interchangeable.

*   **LMS (Linear Multistep Method, version abrégée):** Raccourci pour LMSDiscreteScheduler. Similaire en comportement à LMSDiscreteScheduler.

*   **PLMS (P-sampler - Pseudo Linear Multistep Method):** Variante de LMS qui peut présenter des caractéristiques légèrement différentes en termes de stabilité ou de style. Peut être intéressant à expérimenter si vous utilisez déjà LMS.

*   **DEISMultistepScheduler (Excellent pour les détails fins):**  DEIS (Denoising Estimator Implicit Solvers) est conçu pour exceller dans la préservation des détails fins. Choix idéal si la précision des détails est primordiale et que vous travaillez sur des images complexes.

---

**Note Importante:**

*   Les descriptions ci-dessus sont des généralisations basées sur les caractéristiques typiques de chaque sampler. Les résultats réels peuvent varier en fonction du modèle utilisé, du prompt, des paramètres de génération et d'autres facteurs.
*   L'expérimentation est la clé !  N'hésitez pas à tester différents samplers pour voir ceux qui correspondent le mieux à votre style et à vos besoins spécifiques.


## ▶️ Modules Supplémentaires

### 🌟 Aperçu des Modules
L'application **cyberbill_SDXL** propose plusieurs modules complémentaires qui s'activent automatiquement lorsqu'ils sont placés dans le répertoire `/modules`. Ces modules enrichissent les fonctionnalités de base et permettent aux utilisateurs de personnaliser leur expérience.

### 📚 Liste des Modules Disponibles
1. **Retouche d'image**  
   - Fournit des outils basiques pour modifier ou améliorer vos créations.
   - Compatible avec les images générées par l'application ou externes.

2. **Upscaling**  
   - Améliore la résolution des images grâce à SDXL.
   - Idéal pour des rendus nets et détaillés.

3. **Amélioration d'image**  
   - Utilise AuraSR pour optimiser la clarté et les détails de vos images.

4. **Suppression d'arrière-plan**  
   - Basé sur RemBG, ce module isole rapidement le sujet de l'image en supprimant son arrière-plan.

5. **Navigation sur Civitai**  
   - Permet de parcourir la bibliothèque Civitai pour découvrir de nouveaux modèles et prompts.
   - Option de copier des prompts intéressants directement depuis l'interface.

6. **Module de test**  
   - Fournit un squelette de base pour aider les utilisateurs à créer leurs propres modules.
   - Idéal pour les développeurs souhaitant expérimenter ou personnaliser leurs fonctionnalités.

---

### 🛠️ Activation des Modules
- **Placement automatique** : Placez le module désiré dans le dossier `/modules`. L'application détecte automatiquement sa présence et l'active.
- **Interface utilisateur** : Les modules activés seront accessibles depuis le menu principal ou des onglets spécifiques. Relancer l'application pour une prise en compte

---

### 🌈 Configuration des Modules
Certains modules proposent des options de configuration avancées :
- **Module d'upscaling** :  
  - Ajustez la résolution cible directement dans les paramètres de l'application.
  
- **Suppression d'arrière-plan** :  
  
- **Retouche d'image** :  
  - Permet d'importer des images externes et d'appliquer des filtres rapidement.

---

### 🔧 Développement de Modules Personnalisés
Le module de test inclus fournit un cadre pratique pour développer vos propres modules. Voici comment procéder :
1. **Structure du module** :  
   - Chaque module doit inclure un fichier principal nommé `monModule_mod.py` et des dépendances spécifiques.

2. **Configuration** :  
   - Utilisez le fichier `monMocule_mod.json` du module pour définir ses comportements et paramètres et les traductions.

3. **Documentation** :  
   - Ajoutez des instructions claires dans le dossier du module pour guider les utilisateurs.

---
