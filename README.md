# cyberbill image generator 🚀

This development was heavily inspired by the excellent software Fooocus https://github.com/lllyasviel/Fooocus, whose latest version dates back to August 2024.
Although many forks have appeared, I wanted to create a complete software almost from scratch, drawing from libraries like Gradio, Diffusers, Hugging Face (Transformers, Compel), ONNX Runtime, Rembg, etc. It also integrates specific models and techniques for image enhancement, such as ModelScope for colorization, Diffusers LDM for upscaling, and OneRestore for image restoration. It is therefore a coherent assembly of various sources, and the work of many teams whom I warmly thank.


Passionate about image generation and AI, I heavily relied on Gemini to help me... being a beginner, I learned a lot while designing this software. Like having a teacher by my side, with good foundations and determination, one can have fun and contribute to the community, however small the contribution may be.

## 📌 Prerequisites
- **CUDA 12.8** installed ✅
- **Nvidia RTX Card**: Not tested on other cards.
- **8 GB of VRAM recommended**: Optimization not available for smaller graphics cards.

## 📥 Installation
1.  **Download the project**
    - Choose the `zip` file or `cyberbill_SDXL.zip` and unzip it into the directory of your choice.

2.  **Install CUDA 12.8** via this [link] (https://developer.nvidia.com/cuda-downloads).

3.  **Run the `install.bat` script**
    - This sets up the necessary environment for the software.

4.  **Start the application with `start.bat`**
    - Double-click this file to launch the user interface.

## ▶️ Launching the application
1.  **Double-click on `start.bat`** 🎯
    - This activates the environment and launches the application.
2.  **Or use the command line:**
    ```sh
    venv\Scripts\activate
    python cyberbill_SDXL.py
    ```



## ▶️ Usage
### 🌟 Essential Steps
1.  **Load an SDXL model**
    - Place the `.safetensors` files in `models/checkpoints`.
    - Click on **"List models"**, then select the desired model.

    NOTE: The software is provided **without a model**.
    - If no model is found at launch, the program will ask if you want to load one. Answer with y or n (yes or no). The model will then be loaded. This is a generic model that yields good results: MegaChonkXL.
    - Alternatively, you can download your own models from various sources (SDXL 1.0 checkpoint models in `.safetensors` format to be placed in the `/models/checkpoints` directory).

        Example sites: [civitai.com](https://civitai.com/) | [lexica.art](https://lexica.art/) | [huggingface.co](https://huggingface.co)

2.  **Configure your settings**
    - **VAE**:
        - Place your `.safetensors` files in `/models/vae/`. The vast majority of files come with an integrated VAE, so it's generally not necessary to download one... but just in case!
        - The VAE transforms the latent image into a complete and detailed version.
    - **Sampler**:
        - Select an algorithm to guide image generation (Euler, DDIM, etc.).
    - **Guidance (CFG Scale)**:
        - Determines the image's fidelity to the prompt:
            - *3-7*: Creative results.
            - *10-20*: Precise results.
    - **Steps**:
        - Recommended: around 30 for a balance between quality/speed.
    - **Seed**:
        - Use -1 for a random seed or set a fixed seed to reproduce results.
    - **Dimensions**:
        - Select a predefined format compatible with the model.
    - **Number of images**:
        - Select the number of images to generate.

3.  **Add a prompt**
    - Enter text describing the desired image.
    - Activate "Translate to English" to automate the translation.
    - By checking the box "generate a prompt from an image", you can paste or upload an image from your disk, and a prompt will then be suggested.

4.  **Generate images**
    - Click on **"Generate"**. Images are saved in the `output` folder along with an HTML report.

### 🤖 AI-Powered Prompt Enhancement (LLM) (New in 1.8.9)

The application now features an optional capability to enhance your prompts using a Language Model (LLM).

- **Activation**: Check the "Enhance prompt with AI" checkbox located below the main prompt input field.
- **How it Works**: When this option is enabled, your base prompt is sent to an LLM (by default, `Qwen/Qwen3-0.6B` by QwenAI, configurable in `config.json` via the `LLM_PROMPTER_MODEL_PATH` key).
- **Objective**: The LLM is instructed to generate a more detailed, descriptive, and imaginative version of your initial idea, specifically in **English**, to optimize results with image generation models like Stable Diffusion XL.
- **Resources**: To conserve your graphics card's resources (VRAM) for image generation, the LLM is configured to run on the **CPU**.
- **Output**: The AI-enhanced prompt will replace your initial prompt in the text field and will be used for generation. If the enhancement fails or does not produce a significantly different result, your original prompt will be retained.

This feature aims to help you explore new creative avenues and achieve richer, more detailed images without needing to formulate complex prompts yourself.

    - Click on **"Generate"**. Images are saved in the `output` folder along with an HTML report.

### 🚀 New: Batch Generation (tab) and batch runner (Beta 1.8)

1.  **Create a Batch Definition:**
    *   Go to the **"Batch Generator"** tab.
    *   Configure the parameters (model, VAE, prompt, styles, LoRAs, etc.) for a task.
    *   Use the "Translate Prompt to English" checkbox if needed.
    *   Click **"Add Task to Batch"**. Repeat for all desired tasks.
    *   Review the task list in the table.
    *   Click **"Generate JSON"**. The JSON file will be automatically saved in the directory specified by `SAVE_BATCH_JSON_PATH` in `config.json` (default: `Output\json_batch_files`) with a name like `batch_001.json`.

2.  **Run the Batch:**
    *   Go back to the main **"Image Generation"** tab.
    *   Expand the **"Batch Runner"** accordion.
    *   Click the file input area (or use the File Explorer if available) to **load the generated JSON file** (e.g., `batch_001.json`) from the save directory.
    *   Click **"Run Batch"**.
    *   The application will process each task sequentially, displaying progress and results. You can stop the process using the **"Stop Batch"** button.

## Interface Capture
Image generator, Prompt Enhancement (LLM)
![image](https://github.com/user-attachments/assets/db8810cc-0e92-4059-b9c9-5276ae9e7716)


Image generator, prompt calculated from the image, adding a LoRA

![Capture d'écran 2025-04-24 073557](https://github.com/user-attachments/assets/b3455d1c-308c-4907-8aa6-970d0b92ce7b)

Civitai Downloader Module
![image](https://github.com/user-attachments/assets/436461cd-6408-48d1-a030-22bab26cf5b4)


Watermark Module 
![image](https://github.com/user-attachments/assets/967733a0-7d9e-4316-add6-b8375007bf09)


New Image Enhancement tab (Beta 1.8.5)
![image](https://github.com/user-attachments/assets/edc0eae6-a2c2-4dd6-a767-96b691624010)

New Sana Sprint tab added in Beta 1.8.6 
![image](https://github.com/user-attachments/assets/aa41bd18-a2e1-45ef-b981-947c05e3d0f7)


Batch runner since Beta version 1.8 :

![image](https://github.com/user-attachments/assets/fc718e44-be0d-42c7-9361-699fa2d21b89)


Batch generator since Beta version 1.8 :

![image](https://github.com/user-attachments/assets/80956917-47e0-4875-8fb0-e4fe8fe7e8a6)


Presets since Beta version 1.7, it is possible to save presets,

![Capture d'écran 2025-04-24 074037](https://github.com/user-attachments/assets/cb6dea51-7c86-4c52-9ad4-584573fc91f8)
once the image is produced, give it a name and a rating (optional), and save your creation's data to keep track of it.

Inpainting, define an area of the image to modify, here an 80-year-old person's face instead of a young woman's

![image](https://github.com/user-attachments/assets/d60b8d1b-8e77-4988-abe7-3f81ca0f4a34)


[MODULE] Civitai Downloader (New in Beta 1.8.7 - Screenshot may need update)
<!-- This screenshot might be for an older/different Civitai feature. Update if new UI. -->
![image](https://github.com/user-attachments/assets/506ab5fa-eacd-4f9b-be93-2c35b157cbc6)

[MODULE] Image Editing

![image](https://github.com/user-attachments/assets/2e31935f-8f0d-445d-a123-9784033f7042)

[MODULE] Image to Image (here prompt and style selected)

![image](https://github.com/user-attachments/assets/a3493385-5b48-40eb-82e8-75932d540253)

[MODULE] Remove Background based on RemBG https://github.com/danielgatis/rembg

![iamge](https://github.com/user-attachments/assets/15717a23-9828-4e14-8a78-465110b22f76)



## ▶️ Advanced Configuration

### 🌟 Configuration File: `config.json`

The `config.json` file, located in the `/config` folder, allows you to customize the main settings of the application. Here is a detailed version:

```json
{
    "AUTHOR": "Cyberbill_SDXL",
    "MODELS_DIR": "models\\checkpoints",
    "VAE_DIR": "models\\vae",
    "INPAINT_MODELS_DIR": "models\\inpainting",
	"LORAS_DIR": "models\\loras",
	"SAVE_DIR": "Output",
    "SAVE_BATCH_JSON_PATH": "Output\\json_batch_files", 
    "LLM_PROMPTER_MODEL_PATH": "Qwen/Qwen3-0.6B",
    "IMAGE_FORMAT": "webp",
	"DEFAULT_MODEL": "your_default_modele.safetensors",
    "CIVITAI_API_KEY": "", // Optional: Your Civitai API key. Leave empty if not used.
	"NEGATIVE_PROMPT": "udeformed, ugly, blurry, pixelated, grainy, poorly drawn, artifacts, errors, duplicates, missing, inconsistent, unrealistic, bad anatomy, severed hands, severed heads, crossed eyes, poor quality, low resolution, washed out, overexposed, underexposed, noise, flat, lacking details, generic, amateur",
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
        {"dimensions": "1024*1024", "orientation": "Square"},
        {"dimensions": "1024*960", "orientation": "Landscape"},
        {"dimensions": "1088*960", "orientation": "Landscape"},
        {"dimensions": "1088*896", "orientation": "Landscape"},
        {"dimensions": "1408*704", "orientation": "Landscape"},
        {"dimensions": "1344*704", "orientation": "Landscape"},
        {"dimensions": "1344*768", "orientation": "Landscape"},
        {"dimensions": "1280*768", "orientation": "Landscape"},
        {"dimensions": "1216*832", "orientation": "Landscape"},
        {"dimensions": "1152*832", "orientation": "Landscape"},
        {"dimensions": "1152*896", "orientation": "Landscape"}
	],
	"OPEN_BROWSER": "Yes",
	"GRADIO_THEME": "Default",
	"SHARE":"No",
    "LANGUAGE": "en",
	"PRESETS_PER_PAGE": 12,
	"PRESET_COLS_PER_ROW":4
}
```

### 🛠️ Main Fields:

- **`AUTHOR`**: Name or author of the configuration file.
- **`MODELS_DIR`**: Directory where base SDXL models are stored.
- **`VAE_DIR`**: Location for custom VAEs.
- **`INPAINT_MODELS_DIR`**: Path to models dedicated to inpainting.
- **`LORAS_DIR`**: Location to load LoRA files in `.safetensors` format.
- **`SAVE_DIR`**: Folder where generated images are saved.
- **`SAVE_BATCH_JSON_PATH`**: Folder where generated batch JSON files are automatically saved (New in Beta 1.8).
- **`LLM_PROMPTER_MODEL_PATH`**: (New in 1.8.9) Path or Hugging Face name of the Language Model (LLM) used for prompt enhancement. Default: "Qwen/Qwen3-0.6B".
- **`IMAGE_FORMAT`**: Image file format: `webp`, `jpeg`, or `png`.
- **`DEFAULT_MODEL`**: Model loaded by default at startup.
- **`CIVITAI_API_KEY`**: (Optional) Your Civitai API key. If provided, it will be used by the Civitai Downloader module to access models or information requiring authentication. Leave empty ("") if you don't have one or don't want to use it.
- **`NEGATIVE_PROMPT`**: Generic negative prompt applied by default, useful for excluding unwanted elements in generated results.
- **`FORMATS`**: Image dimensions, specified in multiples of 4, with orientations like `Portrait`, `Square`, and `Landscape`.
- **`OPEN_BROWSER`**:
  - `Yes` opens the application directly in the default browser.
  - `No` disables automatic browser opening.
- **`GRADIO_THEME`**: Customize the user interface appearance with available themes.
- **`SHARE`**:
  - `True` allows sharing the application online via Gradio.
  - `False` limits usage to local only.
- **`LANGUAGE`**: User interface language (`en` for English, `fr` for French).

### 🌟 Additional Options in Detail

- **`FORMATS`**: Determines image dimensions. Each option must respect multiples of 4 for optimal compatibility.
  - **Example**:
    - Portrait: `704*1408`, `768*1280`
    - Square: `1024*1024`
    - Landscape: `1408*704`, `1280*768`

- **`OPEN_BROWSER`**:
  - `Yes`: Opens the application directly in the default browser.
  - `No`: Disables automatic browser opening.

- **`GRADIO_THEME`**: Defines the user interface appearance.
  - **Available Themes**:
    - `Base`: Minimalist with a blue primary color.
    - `Default`: Default theme (orange and gray).
    - `Origin`: Inspired by classic Gradio versions.
    - `Citrus`: Vibrant yellow with 3D effects on buttons.
    - `Monochrome`: Black and white with a classic style.
    - `Soft`: Purple tones with rounded edges.
    - `Glass`: "Glass" visual effect with blue gradients.
    - `Ocean`: Blue-green tones with horizontal transitions.

- **`SHARE`**:
  - `True`: Allows sharing the application online via Gradio.
  - `False`: Restricts the application to local use only.

- **`LANGUAGE`**: Defines the language used in the user interface.
  - `en`: English
  - `fr`: French

- **`PRESETS`**: Ability to adjust the display of presets, number per page, and number per column. Ensure the number of presets per column is a multiple of the number of presets per page.
  - `PRESETS_PER_PAGE`: 12,
  - `PRESET_COLS_PER_ROW`: 4

NOTE:
For paths like `C:\path\to\models`, you need to write it like this:
`C:\\path\\to\\models`
So, for `c:\directory\my_models\checkpoints`, you should write `c:\\directory\\my_models\\checkpoints`

## Learn More About Choosing Samplers:

    EulerDiscreteScheduler (Fast and detailed): A classic Euler sampler, fast and produces detailed images. Good starting point and often used for its efficiency. You already have it.
    DDIMScheduler (Fast and creative): DDIM (Denoising Diffusion Implicit Models) is faster than classic methods and can be more creative, sometimes offering more varied and surprising results. Can be a good choice for rapid exploration.
    DPMSolverMultistepScheduler (Fast and high-quality): An optimized and faster version of DPM solvers. Offers a good compromise between speed and image quality, often considered one of the best choices for speed without sacrificing too much quality.

High-Quality and Photorealistic Samplers (for detailed and realistic rendering):

    DPM++ 2M Karras (Photorealistic and detailed): A high-performing sampler for obtaining photorealistic and highly detailed images. "Karras" indicates the use of an improved noise schedule (Karras noise schedule) which enhances quality. You already have it, and it's an excellent choice.
    PNDMScheduler (Stable and photorealistic): PNDM (Pseudo Numerical Methods for Diffusion Models) is stable and tends to produce photorealistic images with less noise. Can be a good choice if you are looking for a cleaner rendering.
    DPM++ SDE Karras (Photorealistic and with noise reduction): Combines the advantages of DPM++ with an SDE (Stochastic Differential Equations) method and Karras noise. Very effective for reducing noise and achieving high-quality photorealistic rendering.
    DPM++ 2M SDE Karras (Combines photorealism and noise reduction): Another variant of DPM++ SDE Karras that combines photorealism and noise reduction, possibly with slightly different characteristics from the simple DPM++ SDE Karras version.
    KDPM2DiscreteScheduler (Detailed and sharp): Another KDPM variant that tends to produce very detailed and sharp images. Good choice if you are looking for precision.

Artistic and Fluid Samplers (for a more pictorial or stylized rendering):

    Euler Ancestral (Artistic and fluid): An Euler Ancestral sampler that produces more fluid and artistic images. "Ancestral" means it adds noise at each denoising step, which can give a more pictorial look. You already have it, and it's a good choice for artistic styles.
    KDPM2AncestralDiscreteScheduler (Artistic and sharp): Combines the characteristics of KDPM2 (detailed and sharp) with the Ancestral approach (artistic). Can offer a good compromise between detail and artistic style.
    HeunDiscreteScheduler (Good speed/quality compromise): Heun is a sampler that tries to find a good balance between speed and quality, and can sometimes produce results with a softer or "painted" look.
    LMSDiscreteScheduler (Balanced and versatile): LMS (Linear Multistep Method) is a more versatile sampler that can yield good results in various styles. It is often considered a good general choice, neither too fast nor too slow, nor too specialized in a particular style.

## List of Samplers for Image Generation

This section describes the different samplers available for image generation in your tool. The choice of sampler can greatly influence the style, quality, and speed of image generation.

---

### Fast and Efficient Samplers

These samplers are ideal for rapid iterations, testing, or less powerful systems. They offer good image generation speed.

*   **EulerDiscreteScheduler (Fast and detailed):** Classic Euler sampler, known for its speed and ability to produce detailed images. A good starting point and often used for its efficiency.

*   **DDIMScheduler (Fast and creative):** DDIM (Denoising Diffusion Implicit Models) is faster than traditional methods and can be more creative, offering varied and sometimes surprising results. Good for rapid exploration and generating original images.

*   **DPMSolverMultistepScheduler (Fast and high-quality):** Optimized and fast version of DPM solvers. Offers an excellent compromise between speed and image quality. Often considered one of the best choices for fast generation without sacrificing too much quality.

---

### High-Quality and Photorealistic Samplers

These samplers are designed to produce the highest quality images, with photorealistic and highly detailed rendering. They may be slower but offer a superior level of detail and realism.

*   **DPM++ 2M Karras (Photorealistic and detailed):** High-performing sampler for obtaining photorealistic and extremely detailed images. The "Karras" indication means it uses an improved noise schedule (Karras noise schedule) that optimizes image quality. Excellent choice for photorealism.

*   **PNDMScheduler (Stable and photorealistic):** PNDM (Pseudo Numerical Methods for Diffusion Models) is stable and tends to generate photorealistic images with less noise. Good choice if you are looking for a cleaner and more realistic rendering.

*   **DPM++ SDE Karras (Photorealistic and with noise reduction):** Combines the advantages of DPM++ with an SDE (Stochastic Differential Equations) method and Karras noise. Very effective for reducing noise and achieving very high-quality photorealistic rendering.

*   **DPM++ 2M SDE Karras (Combines photorealism and noise reduction):** Variant of DPM++ SDE Karras that also combines photorealism and noise reduction. May present slightly different nuances compared to the simple DPM++ SDE Karras version.

*   **KDPM2DiscreteScheduler (Detailed and sharp):** KDPM variant that produces very detailed and sharp images. Ideal if precision and sharpness of details are paramount.

---

### Artistic and Fluid Samplers

These samplers are more oriented towards an artistic, pictorial, or stylized rendering. They can produce images with a softer, fluid, or "painted" look.

*   **Euler Ancestral (Artistic and fluid):** Euler Ancestral sampler that generates more fluid images with an artistic look. The "Ancestral" approach adds noise at each denoising step, contributing to a more pictorial rendering. Excellent for artistic and creative styles.

*   **KDPM2AncestralDiscreteScheduler (Artistic and sharp):** Combines the characteristics of KDPM2 (detailed and sharp) with the Ancestral approach (artistic). Offers a good balance between precise details and artistic style.

*   **HeunDiscreteScheduler (Good speed/quality compromise):** Heun sampler that seeks a good balance between speed and quality. Can produce results with a softer or "painted" look. A good versatile choice for different styles.

*   **LMSDiscreteScheduler (Balanced and versatile):** LMS (Linear Multistep Method) is a versatile sampler that can yield good results in various image styles. Considered a good general choice, neither too fast nor too specialized in a particular style.

---

### "Abbreviated" or Variant Samplers

These samplers are often abbreviated versions or variants of other samplers, offering similar or slightly modified behaviors.

*   **Euler A (Euler Ancestral, abbreviated version):** Shortcut for Euler Ancestral. Behaves very similarly to Euler Ancestral and can be used interchangeably.

*   **LMS (Linear Multistep Method, abbreviated version):** Shortcut for LMSDiscreteScheduler. Similar in behavior to LMSDiscreteScheduler.

*   **PLMS (P-sampler - Pseudo Linear Multistep Method):** Variant of LMS that may exhibit slightly different characteristics in terms of stability or style. May be interesting to experiment with if you already use LMS.

*   **DEISMultistepScheduler (Excellent for fine details):** DEIS (Denoising Estimator Implicit Solvers) is designed to excel in preserving fine details. Ideal choice if detail precision is paramount and you are working on complex images.

---

**Important Note:**

*   The descriptions above are generalizations based on the typical characteristics of each sampler. Actual results may vary depending on the model used, the prompt, generation parameters, and other factors.
*   Experimentation is key! Feel free to test different samplers to see which ones best suit your style and specific needs.

## ▶️ Additional Modules

### 🌟 Modules Overview
The **cyberbill_SDXL** application offers several complementary modules that activate automatically when placed in the `/modules` directory. These modules enhance the basic functionalities and allow users to customize their experience.

### 📚 List of Available Modules

1.  **Civitai Downloader** (New in Beta 1.8.7)
    *   Added a dedicated tab to search and download models, LoRAs, VAEs, etc., directly from Civitai.
    *   Supports filtering by model type, sort order, period, and NSFW content.
    *   Includes an interface to view model details, select specific versions and files for download.
    *   Option to use a Civitai API key for extended access.

2.  **Image Watermark** (New in Beta 1.8.7)
    *   Added a new tab for applying text or image watermarks to your generated images.
    *   Supports single image processing and batch processing of images from a folder.
    *   Customizable options for watermark content (text/image), font, size, color, scale, opacity, position (including tiling), margin, and rotation.

3.  **Sana Sprint** (New in Beta 1.8.6)
    - Dedicated tab for fast generation using the Sana Sprint model.
    - Includes image-to-prompt generation.
    - Optimized for speed (fixed steps and size).

4.  **Image Enhancement** (New in Beta 1.8.5)
    - Offers multiple tools in a dedicated tab for post-processing:
        - **Colorization:** Adds color to black and white images using ModelScope.
        - **Upscale (4x):** Increases image resolution by 4x using a Diffusers LDM model.
        - **Restoration:** Automatically detects and fixes degradations like blur and noise using OneRestore.
        - **Auto Retouch:** Applies simple contrast, sharpness, and saturation enhancements.
    - Models are loaded on demand to save VRAM.

5.  **Batch Generator & Runner** (Functionality from Beta 1.8)
    *   **Batch Generator Tab:** Provides a dedicated interface to create and manage lists of generation tasks (batches). Generates JSON files defining the batch.
    *   **Batch Runner (Main Tab):** Loads and executes these batch tasks from a JSON file.

6.  **Image to Image**
    * Allows transforming an existing image using a prompt and styles.
    * Supports processing a single image or a folder containing multiple images (batch processing).
    * Allows browsing a folder to search for images to process.

7.  **Background Removal (RemBG)**
    *   Based on RemBG, this module quickly isolates the subject of the image by removing its background.

8.  **Image Editing**
    *   Provides basic tools to modify or enhance your creations.
    *   Compatible with images generated by the application or external ones.

----

### 🌈 Module Notes
*   **Hugging Face Model Cache:** Models downloaded from Hugging Face (e.g., for colorization, upscaling, translation, image-to-prompt) are typically stored in the local Hugging Face cache. On Windows, this folder is often located at `C:\Users\YOUR_USERNAME\.cache\huggingface`. Managing this cache (size, cleaning) is done through Hugging Face/Transformers tools or environment variables.

*   **Model Management:** Modules like Image Enhancement load their specific models (Colorization, Upscale, Restoration) only when needed and unload them afterward to conserve VRAM. This might involve unloading the main SDXL generation model temporarily.
*   **Dependencies:** Ensure `install.bat` was run correctly to install necessary packages like `modelscope`, `diffusers`, `rembg`, etc.
*   **Configuration:** Most module settings are handled within their respective tabs in the UI. Check `config.json` for global settings like save paths.

----

### 🛠️ Activating Modules
- **Automatic Placement**: Place the desired module in the `/modules` folder. The application automatically detects its presence and activates it (restart application).
- **User Interface**: Activated modules will be accessible from the main menu or specific tabs. Restart the application for changes to take effect.

---

### 🌈 Configuring Modules
Some modules offer advanced configuration options:
- **Upscaling Module**:
  - Adjust the target resolution directly in the application settings.

- **Background Removal**:
  - (Configuration details might be specific to the module)

- **Image Editing**:
  - Allows importing external images and applying filters quickly.

---

### 🔧 Developing Custom Modules
The included test module provides a practical framework for developing your own modules. Here's how to proceed:
1.  **Module Structure**:
    - Each module must include a main file named `myModule_mod.py` and specific dependencies.

2.  **Configuration**:
    - Use the module's `myModule_mod.json` file to define its behaviors, parameters, and translations.

3.  **Documentation**:
    - Add clear instructions in the module's folder to guide users.

---
### ⚙️ Metadata Additions

The application saves generated images with comprehensive metadata for easy management and tracking:

* **XMP Metadata**: Embedded directly within the image file, it includes key information such as the module used, author, model, VAE, generation parameters (steps, guidance, styles, prompt, etc.), image size, generation time, and the original file (in batch mode).
* **HTML Report**: An HTML file is created for each image, presenting the same metadata in a readable and user-friendly format.
* **File Name**: The image file name is constructed descriptively, including elements such as the module used, original file name (if batch), styles applied, generation date and time, and image dimensions.

---
