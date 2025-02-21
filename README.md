# cyberbill générateur d'image 🚀

## 📌 Prérequis
- **CUDA 12.6** installé ✅
- **Carte Nvidia RTX** pas testé sur d'autres cartes.
- **Testé sur GTX 1650 Ti** avec 4 go de vram mais ne fonctionne pas, si quelqu'un à des idées...
- **8 go de vram recommandés** pour le moment je n'ai pas encore optimisé pour les petites cartes graphique GTX etc.


## 📥 Installation
1. **Télécharger (code et choisir zip ou cyberbill_SDXL.zip) décompresser à l'endroit voulu
2. **Téléchargez et installez** [CUDA 12.6] https://developer.nvidia.com/cuda-12-6-0-download-archive
3. **Lancez** `install.bat`
4. **Lancez start.bat**
4. **Profitez !** 🎨



## ▶️ Lancement de l’application
1. **Double-cliquez sur `start.bat`** 🎯  
   - Cela active l’environnement et lance l’application.
2. **Ou utilisez la ligne de commande :**
   ```sh
   cd venv\Scripts\
   activate.bat
   python cyberbill_SDXL.py
   ```
   
   
## ▶️ Utilisation
1. **Charger un modèle impérativement pour utiliser la génération d'image**
   - Le logiciel est fourni **sans modèle**.
     - Si aucun modèle n'est trouvé au lancement, le programme vous demandera si vous souhaitez en charger un. Répondez par o ou n (oui ou non). Le modèle sera alors chargé. Il s'agit d'un modèle générique qui donne de bons résultats : MegaChonkXL.
     - Par ailleurs, vous pouvez télécharger vos propres modèles sur différentes sources (modèle checkpoints SDXL 1.0 au format .safetensors à placer dans le répertoire `/models/checkpoints`).

     Exemples de sites : [civitai.com](https://civitai.com/) | [lexica.art](https://lexica.art/) | [huggingface.co](https://huggingface.co)

2. Au lancement de l'application, cliquez sur "Lister les modèles".
3. Les modèles présents dans le dossier models seront affichés.
4. **[FACULTATIF]** Choisir un VAE (Auto-Encodeur Variationnel). Placez vos VAE (.safetensors uniquement) dans `/models/vae/`. *Le VAE prend l'image générée dans l'espace latent et la "décompresse" pour la rendre visible et détaillée. C'est comme si vous demandiez au peintre de transformer la version miniature du tableau en une œuvre d'art complète.* SDXL est livré avec un VAE intégré, ce qui signifie que vous n'avez pas besoin de télécharger ou d'installer de VAE supplémentaire. Cependant, il existe également des VAE personnalisés que vous pouvez utiliser pour obtenir des résultats différents.
5. **[FACULTATIF]** Choisir un sampler. *En termes simples, le sampler est l'algorithme qui guide le processus de transformation du bruit aléatoire en une image cohérente.*
6. Cliquez sur **"Charger le modèle"**.

7. **[FACULTATIF]** Cocher la case Lora pour utiliser un Lora :
   - En cochant la case, de nouvelles options s'offrent à vous. Vous pourrez choisir un Lora (compatible SDXL 1.0), lui donner un poids, et le décharger quand vous n'en aurez plus besoin. **ATTENTION** Le fait de changer de modèle décharge automatiquement le Lora, il faudra donc le recharger. Il est possible de charger plusieurs Lora, mais pour le moment, il n'est pas encore possible de leur donner un poids individuellement.
8. **Taper un prompt**. Le compteur de tokens vous indique la longueur à ne pas dépasser.
   - Cocher la case Traduire en anglais vous permet de taper votre prompt en français et de le faire traduire.
   - Générer un prompt à partir d'une image permet de générer automatiquement un prompt. Il s'uffit alors de faire glisser une image ou d'en coller une depuis le presse papier, ou encore de faire une photo à partir de votre webcam.
9. **Régler les paramètres**
   - Guidage :
     - *En termes simples, le guidage détermine à quel point l'image générée est fidèle au prompt.*
     - **Valeurs faibles** (par exemple, 3-7) : L'image aura plus de liberté créative et pourra s'éloigner du prompt. Cela peut donner des résultats plus surprenants et artistiques, mais aussi moins précis par rapport à la description.
     - **Valeurs élevées** (par exemple, 10-20) : L'image sera plus étroitement liée au prompt et essaiera de le suivre de plus près. Cela peut donner des résultats plus précis et détaillés, mais aussi potentiellement plus rigides et moins créatifs.
   - Étapes :
     - **Impact du nombre d'étapes** :
       - **Qualité de l'image** : En général, un nombre d'étapes plus élevé tend à produire des images de meilleure qualité, avec plus de détails, moins de bruit et une meilleure fidélité au prompt (la description textuelle). Cependant, au-delà d'un certain point, l'amélioration de la qualité devient marginale, voire négligeable.
       - **Vitesse de génération** : Un nombre d'étapes plus élevé signifie un temps de génération plus long. Il existe donc un compromis entre la qualité de l'image et la vitesse de génération.
     - **Nombre d'étapes recommandé pour SDXL** :
       - Pour SDXL, un nombre d'étapes d'échantillonnage d'environ 30 est souvent considéré comme un bon équilibre entre qualité et vitesse. Au-delà de 30, chaque étape supplémentaire offre un rendement décroissant en termes d'amélioration de la qualité. Il est rare de voir des améliorations significatives au-delà de 50 étapes.
10. Choisir un format pour les dimensions de votre image.
11. **Seed**. La valeur par défaut -1 génère un seed aléatoire.
     - **Reproductibilité** : Si vous utilisez le même seed, le même prompt et les mêmes autres paramètres, vous obtiendrez exactement la même image à chaque fois. Cela est extrêmement utile pour affiner un résultat particulier, expérimenter avec d'autres paramètres tout en conservant la même base, ou partager vos créations avec d'autres en leur permettant de les reproduire à l'identique.
     - **Variété** : En changeant le seed, vous obtiendrez une image différente, même avec le même prompt. Cela vous permet d'explorer un large éventail de possibilités créatives à partir d'une même idée de base.
12. **Nombre d'images** permet de lancer plusieurs images avec le même prompt.
13. **Générer ou arrêter**.
    - Générer produit l'image, vous retrouverez l'image ainsi qu'un rapport au format HTML dans le dossier output.
14. **Activer la retouche d'image** :
    - Cocher cette case ouvre un accès à des outils basiques pour retoucher des images. Il est possible de retoucher une image générée. Pour cela, faites un clic droit sur l'image et sélectionnez "copier l'image", puis collez l'image dans la zone "Sélectionner une image" en cliquant sur l'icône presse-papier 📋.

**Au premier lancement si il n'y a pas de modèle l'application vous invite à en télécharger un :**
L'avertissement sur Triton n'est pas bloquant. Triton est une bibliothèque qui permet d'optimiser les calculs d'inférence. Triton ne fonctionne pas sur Windows
![image](https://github.com/user-attachments/assets/c12ebfa5-779a-47d8-8f53-5b2df4cfd484)
![image](https://github.com/user-attachments/assets/9f9a192e-f04e-4c0f-9c77-9fd81507aa08)

L'application au lancement :


![image](https://github.com/user-attachments/assets/5d9cd3b8-e4ce-4dfb-9a0c-91f42149b6bc)

Choisir le modèle et cliquer sur charger, attendre que le modèle soit charger pour utiliser la génération d'image :


![image](https://github.com/user-attachments/assets/ad69af2a-a2c8-4195-93e9-a5020a7df058)

Indique que le modèle est chargé : 


![image](https://github.com/user-attachments/assets/2ddf3d56-8d27-4be0-98f7-ce960f3d909b)

On tappe un prompt que l'on peut traduire en cochant la case :


![image](https://github.com/user-attachments/assets/2324a269-d88e-416a-b950-0bc3cd9aff91)

On régle les paramètre comme indiqué plus haut :


![image](https://github.com/user-attachments/assets/a1948d2d-f1c3-4ed8-a4db-c0cec87e0f1d)

On clique sur générer : 


![image](https://github.com/user-attachments/assets/471e1820-4f4c-4a10-a67e-29a257d8cfd9)

![image](https://github.com/user-attachments/assets/a8f33682-5c67-45d1-bc69-f3e3cb9527e8)

Les images d'affichent au fur et à mesure de la production :


![image](https://github.com/user-attachments/assets/b0986577-329f-4ff6-833a-2d1145a765a4)
![image](https://github.com/user-attachments/assets/b1dfcc4a-af49-4a03-aeb8-c829f45fd07a)

Un fichier `rapport.html` est créé dans le même répertoire que les images. Il permettra de consulter les images produites et de mémoriser les paramètres.

![image](https://github.com/user-attachments/assets/e37d134b-84c1-44a1-a816-fabf4a44460c)

![image](https://github.com/user-attachments/assets/3d790458-f301-4d02-9c79-d8f7a66c2f77)
Cliquer sur l'image pour l'agrandir


**Prompt à partir d'une image :**
Glisser une image dans la zone dédiée, ou coller-la :


![image](https://github.com/user-attachments/assets/e0e9428b-7c67-47b6-a1dd-54f77ae667d9)


le prompt se génère automatiquement  :


![image](https://github.com/user-attachments/assets/2873fcef-e131-4367-882c-55f0054302e7)


Activer la retouche d'images :
Cocher la case correspondante pour voir apparaître quelques outils pour retoucher une image, il est possible de copier (ctrl + c) à partir de la galerie d'image générée vers le module de retouche en collant l'image (ctrl + v) 


![image](https://github.com/user-attachments/assets/7d57ba30-b863-49ee-8fa4-f61ef1fb2ad2)



## ▶️ Configuration avancée
Il est possible de modifier le fichier de configuration.

1. Allez dans le dossier `./config` et ouvrez le fichier `config.json` dans un éditeur de texte simple.
2. Pour les petites configurations, je recommande fortement d'utiliser des tailles d'images de 512 x 512 maximum. Des images plus grandes feront planter la génération.

Le fichier se présente ainsi :
 
```json
{
    "AUTHOR": "Cyberbill_SDXL",
    "MODELS_DIR": "models\\checkpoints",
    "VAE_DIR": "models\\vae",
	"LORAS_DIR": "models\\loras",
	"SAVE_DIR": "output",
    "IMAGE_FORMAT": "webp",
	"DEFAULT_MODEL": "votre_modele_par_defaut_au_lancement.safetensors",
	"NEGATIVE_PROMPT": "udeformed, ugly, blurry, pixelated, grainy, poorly drawn, artifacts, errors, duplicates, missing, inconsistent, unrealistic, bad anatomy, severed hands, severed heads, crossed eyes, poor quality, low resolution, washed out, overexposed, underexposed, noise, flat, lacking details, generic, amateur",
    "FORMATS": [
        "704*1408", "704*1344", "768*1344", "768*1280", "832*1216",
        "832*1152", "896*1152", "896*1088", "960*1088", "960*1024",
        "1024*1024", "1024*960", "1088*960", "1088*896"
    ],
	"OPEN_BROWSER": "Yes",
	"GRADIO_THEME": "Defaut",
	"SHARE":"False" 
}



```
#changer les repertoirs par défauts des modèles et de la sortie des images
**MODELS_DIR, VAE_DIR, SAVE_DIR, REPORT_PATH
Personnalisation du stockage des modèles

Vous pouvez personnaliser l'emplacement où sont stockés vos modèles, vos vae, vos loaras et vos images. Veuillez noter qu'il est nécessaire d'échapper le caractère \.
Exemple
Au lieu d'utiliser un chemin de fichier comme ceci :
C:\dossier\de\modeles
Vous devrez l'écrire comme ceci :
C:\\dossier\\de\\modeles
pour c:\repertoire\mes_modeles\checkpoints il faudra écrire c:\\repertoire\\mes_modeles\\checkpoints


**MODELS_DIR** : endroit où sont stockés les modèles de base SDXL 1.0

**VAE_DIR** : endroit où sont stockés les VAE (attention uniquement SDXL 1.0)

**LORAS_DIR** : endroit où sont stockés vos Loras (attention uniquement SDXL 1.0)

**SAVE_DIR** : endroit où sont stockées les photos produites. Un répertoire à la date du jour sera créé pour stocker les photos.

**NEGATIVE_PROMPT** : permet de changer le prompt négatif. Il sera utilisé pour toutes les images. C'est un choix que j'ai fait de mettre un prompt négatif générique.

**IMAGE_FORMAT** correspond au type de fichier : webp | jpeg | png

**FORMATS** correspond à la taille (impérativement des multiples de 4). Il est conseillé d'utiliser des résolutions proches de 1024x1024 pixels.

**OPEN_BROWSER** si oui ou yes l'application ouvrira automatiquement le navigateur par défault.

**GRADIO_THEME** : permet de choisir le thème de l'application parmi les thèmes suivants :


	- Base: Thème minimaliste avec une couleur primaire bleue.
 
	- Default: Thème par défaut de Gradio 5, orange et gris.
 
	- Origin: Similaire au style de Gradio 4, couleurs plus sobres.
 
	- Citrus: Thème jaune avec effets 3D sur les boutons.
 
	- Monochrome: Thème noir et blanc avec des polices de caractères de style journal.
 
	- Soft: Thème violet avec bords arrondis et étiquettes mises en évidence.
 
	- Glass: Thème bleu avec effet de verre grâce à des dégradés verticaux.
 
	- Ocean: Thème bleu-vert avec dégradés horizontaux.

**SHARE** : Si vous mettez True, alors un lien sera créé pour utiliser l'application depuis un autre ordinateur. **ATTENTION** : Pour le moment, il n'y a pas de système de queue. Ainsi, si vous partagez le lien et que plusieurs personnes utilisent le logiciel, il y a un gros risque de bug.
 

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


