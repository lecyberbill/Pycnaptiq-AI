# cyberbill générateur d'image 🚀

## 📌 Prérequis
- **CUDA 11.8** installé ✅
- **Carte Nvidia RTX** pas testé sur d'autres cartes.
- **8 go de vram recommandés** pour le moment je n'ai pas encore optimisé pour les petites cartes graphique GTX etc.


## 📥 Installation
1. **Télécharger (code et choisir zip) décompresser à l'endroit voulu
2. **Téléchargez et installez** [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
3. **Lancez** `install.bat`
4. **Lancez start.bat**
4. **Profitez !** 🎨



## ▶️ Lancement de l’application
1. **Double-cliquez sur `start.bat`** 🎯  
   - Cela active l’environnement et lance l’application.
2. **Ou utilisez la ligne de commande :**
   ```sh
   venv\Scripts\activate
   python cyberbill_SDXL.py
   ```
   
   
## ▶️ Utilisation
1. **Charger un modèle impérativement pour utiliser la génération d'image**
	- Le logiciel est fourni **sans modèle**, vous pouvez télécharger des modèles sur différentes sources (modèle checkpoints SDXL 1.0 au format .safetensors à placer dans le repertoire /models/checkpoints)
		Exemples de sites : https://civitai.com/ | https://lexica.art/ | https://huggingface.co
 	- D'abord cliquer sur **"Lister les modèles"**
	- Choisir un modèle (placer vos modèles **.safetensors** uniquement dans /models/checkpoints/ )
		https://civitai.com est une bonne source pour se procurer des modèle (SDXL 1.0)
	- Choisir un VAE (Auto-Encodeur Variationnel, placer vos vae **.safetensors** uniquemen dans /models/vae/ ) 
		*Le VAE prend l'image générée dans l'espace latent et la "décompresse" pour la rendre visible et détaillée. C'est comme si vous demandiez au peintre de transformer la version miniature du tableau en une œuvre d'art complète
		*SDXL est livré avec un VAE intégré, ce qui signifie que vous n'avez pas besoin de télécharger ou d'installer de VAE supplémentaire. Cependant, il existe également des VAE personnalisés que vous pouvez utiliser pour obtenir des résultats différents
	- Choisir un sampler. 
		*En termes simples, le sampler est l'algorithme qui guide le processus de transformation du bruit aléatoire en une image cohérente
	- Cliqer sur **"Charger le modèle"**
2. **Taper un prompt le compteur de tokens vous indique la longueur à ne pas dépasser**
	- Cocher la case Traduire en anglais vous permez de taper votre prompt en français et de le faire Traduire
	- Générer un prompt à partir d'une image permet de générer automatiquement un prompt
3. **Régler les paramètres**
	- Guidage :
		*En termes simples, le guidage détermine à quel point l'image générée est fidèle au prompt.
		*Valeurs faibles (par exemple, 3-7) : L'image aura plus de liberté créative et pourra s'éloigner du prompt. Cela peut donner des résultats plus surprenants et artistiques, mais aussi moins précis par rapport à la description.
		*Valeurs élevées (par exemple, 10-20) : L'image sera plus étroitement liée au prompt et essaiera de le suivre de plus près. Cela peut donner des résultats plus précis et détaillés, mais aussi potentiellement plus rigides et moins créatifs.
	- Etapes : 
		**Impact du nombre d'étapes :
		*Qualité de l'image: En général, un nombre d'étapes plus élevé tend à produire des images de meilleure qualité, avec plus de détails, moins de bruit et une meilleure fidélité au prompt (la description textuelle). Cependant, au-delà d'un certain point, l'amélioration de la qualité devient marginale, voire négligeable.
		*Vitesse de génération: Un nombre d'étapes plus élevé signifie un temps de génération plus long. Il existe donc un compromis entre la qualité de l'image et la vitesse de génération.
		**Nombre d'étapes recommandé pour SDXL :
		*Pour SDXL, un nombre d'étapes d'échantillonnage d'environ 30 est souvent considéré comme un bon équilibre entre qualité et vitesse. Au-delà de 30, chaque étape supplémentaire offre un rendement décroissant en termes d'amélioration de la qualité. Il est rare de voir des améliorations significatives au-delà de 50 étapes.
4. **Choisir un format pour les dimensions de votre image
5. **Seed la valeur par défaut -1 génére un seed aléatoire 
		*Reproductibilité : Si vous utilisez le même seed, le même prompt et les mêmes autres paramètres, vous obtiendrez exactement la même image à chaque fois. Cela est extrêmement utile pour affiner un résultat particulier, expérimenter avec d'autres paramètres tout en conservant la même base, ou partager vos créations avec d'autres en leur permettant de les reproduire à l'identique.
		*Variété : En changeant le seed, vous obtiendrez une image différente, même avec le même prompt. Cela vous permet d'explorer un large éventail de possibilités créatives à partir d'une même idée de base.
6. **Nombre d'image permet de lancer plusieurs images avec le même prompt
7. **Générer ou arrêter
	- Générer, génère l'image, vous retrouverez l'image ainsi qu'un rapport au format html dans le dossier output
8. **Activer la retouche d'image
	- cocher cette case ouvre un accès à des outils basiques pour retoucher des images, il est possible de retoucher une image générée, pour cela faire un clic droit sur l'image et "copier l'image", coller l'image dans la zone "Sélectionner une image" en cliquant sur l'icone presse papier 📋

## ▶️ Configuration avancée
Il est possible de modifier le fichier de configuration
Allez dans le dossier /config ouvrez le fichier config.json dans un éditeur de texte simple

le fichier se présente ainsi : 
```json
{
    "MODELS_DIR": "models\\checkpoints",
    "VAE_DIR": "models\\vae",
	"SAVE_DIR": "output",
    "IMAGE_FORMAT": "webp",
	"NEGATIVE_PROMPT": "udeformed, ugly, blurry, pixelated, grainy, poorly drawn, artifacts, errors, duplicates, missing, inconsistent, unrealistic, bad anatomy, severed hands, severed heads, crossed eyes, poor quality, low resolution, washed out, overexposed, underexposed, noise, flat, lacking details, generic, amateur",
    "FORMATS": [
        "704*1408", "704*1344", "768*1344", "768*1280", "832*1216",
        "832*1152", "896*1152", "896*1088", "960*1088", "960*1024",
        "1024*1024", "1024*960", "1088*960", "1088*896"
    ],
	"GRADIO_THEME": "Defaut"
}

```
#changer les repertoirs par défauts des modèles et de la sortie des images
**MODELS_DIR, VAE_DIR, SAVE_DIR, REPORT_PATH
Vous pouvez personnaliser l'endroit où sont stokés les modèles, attention il faut **"écahpper"** le \ exemple :

pour c:\repertoire\mes_modeles\checkpoints il faudra écrire c:\\repertoire\\mes_modeles\\checkpoints


**MODELS_DIR** : endroit où sont stoké les modèles de base SDXL 1.0
**VAE_DIR** : endroit où sont stoké les VAE (attention uniquement SDXL 1.0)
**SAVE_DIR** : endroit où sont stoké les photos produites, un repertoire à la date du jour sera créé pour stoker les photos
**NEGATIVE_PROMPT** : permet de changer le prompt négatif, il sera utiliser pour toutes les images, c'est un choix que j'ai de mettre un prompt négatif générique.
**IMAGE_FORMAT** correspond au type de fichier, webp | jpeg | png
**FORMATS** correspond à la taille impérativement des multiples de 4,  il est conseillé d'utiliser des résolutions proches de 1024x1024 pixels
**GRADIO_THEME** : permet de choisir le theme de l'application parmis les thèmes suivant : 
	Base: Thème minimaliste avec une couleur primaire bleue.
	Default: Thème par défaut de Gradio 5, orange et gris.
	Origin: Similaire au style de Gradio 4, couleurs plus sobres.
	Citrus: Thème jaune avec effets 3D sur les boutons.
	Monochrome: Thème noir et blanc avec des polices de caractères de style journal.
	Soft: Thème violet avec bords arrondis et étiquettes mises en évidence.
	Glass: Thème bleu avec effet de verre grâce à des dégradés verticaux.
	Ocean: Thème bleu-vert avec dégradés horizontaux.
