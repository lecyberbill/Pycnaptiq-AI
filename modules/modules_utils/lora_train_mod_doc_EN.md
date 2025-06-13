# LoRATraining_mod.py Module Documentation

## Table of Contents
1.  [Introduction](#introduction)
2.  [Main Features](#main-features)
3.  [Key Components](#key-components)
    *   [Constants and Global Variables](#constants-and-global-variables)
    *   [Class `DreamBoothDataset`](#class-dreamboothdataset)
    *   Function `collate_fn`
    *   Class `LoRATrainingModule`
        *   Initialization (`__init__`)
        *   User Interface Creation (`create_tab`)
        *   Training Thread Management (`run_preparation_and_training_thread`)
        *   Main Training Logic (`_actual_preparation_and_training`)
        *   Prompt Encoding (`_encode_prompt`)
        *   `time_ids` Computation (`_compute_time_ids`)
4.  Workflow
5.  Dependencies
6.  Configuration
7.  Usage Notes and Best Practices

---

## 1. Introduction

The `LoRATraining_mod.py` module is designed to enable the training of LoRA (Low-Rank Adaptation) models for Stable Diffusion XL (SDXL) directly within the CyberBill Image Generator application. It provides a user interface to configure training parameters, prepare data (including automatic image captioning), and launch the training process. The goal is to simplify the creation of custom LoRAs without requiring complex manual setup of training environments.

---

## 2. Main Features

*   **Automated Data Preparation**:
    *   Cropping source images to the target size.
    *   Automatic image captioning using a Florence-2 model (configurable).
    *   Adding a trigger word to captions.
*   **LoRA Training for SDXL**:
    *   Support for training the UNet and optionally text encoders (CLIPTextModel and CLIPTextModelWithProjection).
    *   Use of Hugging Face's PEFT (Parameter-Efficient Fine-Tuning) library for LoRA adaptation.
    *   Flexible configuration of training hyperparameters (learning rate, batch size, number of epochs, etc.).
    *   Choice of optimizer (AdamW, AdamW8bit, Lion).
    *   Mixed precision support (fp16, bf16, fp32) to optimize memory usage and speed.
*   **Integrated User Interface**:
    *   A dedicated tab in the Gradio application to manage all steps.
    *   Real-time display of training logs.
    *   Buttons to start and stop the training process.
*   **Model Saving**:
    *   Saving LoRA checkpoints at defined intervals.
    *   Saving the final LoRA model in Diffusers-compatible format (`.safetensors`) to the configured LORAS_DIR.
*   **Resource Management**:
    *   Unloading the main SDXL base model before loading components for training to free up VRAM.
    *   GPU memory cleanup after training.

---

## 3. Key Components

### Constants and Global Variables

*   `MODULE_NAME`: "LoRATraining" - Name of the module.
*   `SUPPORTED_IMAGE_EXTENSIONS`: List of supported image extensions for instance data (e.g., `['.png', '.jpg', '.jpeg', '.webp']`).

### Class `DreamBoothDataset`

This class inherits from `torch.utils.data.Dataset` and is responsible for loading and preprocessing images and their captions for training.

*   **`__init__(self, instance_data_root, instance_prompt_fallback, tokenizer_one, tokenizer_two, target_size=1024, center_crop=False)`**
    *   `instance_data_root`: Path to the folder containing instance images and their `.txt` caption files.
    *   `instance_prompt_fallback`: Caption to use if a `.txt` file is missing for an image.
    *   `tokenizer_one`, `tokenizer_two`: The SDXL tokenizers (CLIP ViT-L and OpenCLIP ViT-bigG).
    *   `target_size`: The resolution to which images will be resized and cropped.
    *   `center_crop`: Boolean indicating whether to use center cropping or random cropping.
    *   During initialization, it scans `instance_data_root`, collects image paths, reads associated captions (or uses the fallback), and stores original image sizes.
    *   It also defines image transformations (resize, crop, convert to tensor, normalize).

*   **`__len__(self)`**: Returns the total number of instance images.

*   **`__getitem__(self, index)`**: Returns a dictionary containing a training example:
    *   `instance_images`: The preprocessed image (normalized tensor).
    *   `input_ids_one`, `input_ids_two`: Token IDs of the caption, tokenized by the two respective tokenizers.
    *   `original_size_hw`: Original size of the image (height, width).
    *   `crop_coords_top_left_yx`: Coordinates (y, x) of the top-left corner of the crop applied to the resized image.
    *   Handles image loading errors by returning another random example.

### Function `collate_fn`

This function is used by `torch.utils.data.DataLoader` to assemble individual examples returned by `DreamBoothDataset` into a batch.

*   **`collate_fn(examples)`**:
    *   Takes a list of dictionaries (each dictionary is a dataset example).
    *   Stacks image tensors (`pixel_values`) and token IDs (`input_ids_one`, `input_ids_two`).
    *   Returns a dictionary containing batched tensors and lists of metadata (`original_sizes_hw`, `crop_coords_top_left_yx`).

### Class `LoRATrainingModule`

This is the main class that manages the user interface, training logic, and interaction with other parts of the application.

#### Initialization (`__init__`)

*   **`__init__(self, global_translations, model_manager_instance, gestionnaire_instance, global_config)`**:
    *   Stores instances of `global_translations`, `ModelManager`, `GestionModule`, and `global_config`.
    *   Initializes states like `is_training` (boolean) and `stop_event` (to stop training).
    *   Defines default paths for LoRA projects (`LORA_PROJECTS_DIR`) and final LoRA saving (`LORAS_DIR`) based on `global_config`.
    *   Retrieves the default SDXL model name for training from `global_config`.
    *   Determines the `device` (CPU/GPU) and `torch_dtype` (tensor precision).

#### User Interface Creation (`create_tab`)

*   **`create_tab(self, module_translations_from_gestionnaire)`**:
    *   Builds the Gradio tab for LoRA training.
    *   **Data Preparation (Left Column)**:
        *   `input_images_dir`: Field for the path to the source images folder.
        *   `trigger_word`: Trigger word to add to captions.
        *   `concept`: Concept name for data organization.
        *   `caption_task_dropdown`: Choice of Florence-2 task for captioning.
    *   **Training Parameters (Left Column)**:
        *   `lora_output_name`: Name of the final LoRA file.
        *   `base_model_dropdown`: Selection of the base SDXL model for training.
        *   `training_project_dir`: Root folder for training projects.
        *   `epochs`, `learning_rate_dropdown`, `batch_size`, `resolution`: Basic hyperparameters.
        *   Advanced Options (in accordions):
            *   `network_dim` (LoRA rank), `network_alpha`.
            *   `train_unet_only`, `train_text_encoder`: Checkboxes to specify parts of the model to train.
            *   `optimizer_dropdown`, `lr_scheduler_dropdown`, `mixed_precision_dropdown`.
            *   `save_every_n_epochs`: Frequency for saving checkpoints.
    *   **Controls and Logs (Right Column)**:
        *   `prepare_train_button`: Button to start preparation and training.
        *   `stop_button`: Button to stop ongoing training.
        *   `status_output_html`: HTML area to display status logs.
    *   Connects button events to corresponding functions (`run_preparation_and_training_thread`, `stop_training_wrapper`).

#### Training Thread Management (`run_preparation_and_training_thread`)

*   **`run_preparation_and_training_thread(self, *args)`**:
    *   This function is a generator called when the user clicks "Prepare and Train".
    *   It checks if training is already in progress.
    *   Resets state (`is_training`, `stop_event`, `status_log_list`).
    *   Updates the interface to indicate the start of the process.
    *   Creates a `queue.Queue` to allow the training thread to communicate log messages to the main UI thread.
    *   Launches the `_actual_preparation_and_training` function in a new `threading.Thread`.
    *   Loops to retrieve messages from the `log_queue` and update the log HTML (`status_output_html`) via `yield`.
    *   Waits for the training thread to finish.
    *   Updates the final state of the interface.

#### Main Training Logic (`_actual_preparation_and_training`)

This is the function executed in the separate thread. It contains the business logic for training.

*   **`_actual_preparation_and_training(self, log_queue, input_images_dir, ..., save_every_n_epochs)`**:
    1.  **Input Validation**: Checks that essential paths and parameters are provided.
    2.  **Instance Data Preparation**:
        *   Creates project folders (`training_project_dir / lora_output_name / data / concept_images`).
        *   Iterates through images in `input_images_dir`.
        *   For each image:
            *   Crops it to a fixed size (e.g., 1024x1024) if it's larger. Smaller images are skipped.
            *   Saves the cropped image to the project's data folder.
            *   Generates a caption using `generate_prompt_from_image` (Florence-2).
            *   Adds the `trigger_word` to the caption.
            *   Saves the caption in a corresponding `.txt` file.
        *   Unloads the captioning model (Florence-2) after use.
    3.  **Base Model Loading and PEFT Configuration**:
        *   Determines the loading `torch_dtype` based on the `mixed_precision_choice`.
        *   Loads the `StableDiffusionXLPipeline` from the `.safetensors` file of the selected base model.
        *   Extracts necessary components: `tokenizer_one`, `tokenizer_two`, `text_encoder_one`, `text_encoder_two`, `vae`, `unet`, `noise_scheduler`.
        *   Moves components to the correct `device` and with the correct `dtype`.
        *   Disables gradients (`requires_grad_(False)`) and sets to `eval()` mode models that will not be trained (VAE, and text encoders/UNet if not LoRAfied).
        *   Deletes the full `pipeline` object to free memory.
        *   Configures `LoraConfig` for the UNet (and optionally for text encoders).
        *   Applies `unet.add_adapter(lora_config)` (and similarly for text encoders if `train_text_encoder` is checked).
        *   If `fp16` mixed precision is used, casts LoRA parameters to `fp32` for training stability via `cast_training_params`.
        *   Logs the number of trainable parameters.
    4.  **`DreamBoothDataset` and `DataLoader` Creation**:
        *   Instantiates `DreamBoothDataset` with the prepared data.
        *   Creates a `DataLoader` to iterate over the dataset in batches.
    5.  **Optimizer and Scheduler Configuration**:
        *   Collects parameters requiring gradients (those of LoRA adapters).
        *   Instantiates the chosen optimizer (AdamW, AdamW8bit with `bitsandbytes`, or Lion).
        *   Instantiates a learning rate scheduler (`get_scheduler` from Diffusers).
    6.  **Training Loop**:
        *   Iterates over the number of epochs.
        *   For each epoch, iterates over batches from the `DataLoader`.
        *   **Forward Pass**:
            *   Encodes batch images into latents with the VAE (in `torch.float32` and `torch.no_grad`).
            *   Adds noise to latents according to a random `timestep`.
            *   Encodes batch captions into text embeddings with the two text encoders (with `torch.no_grad` if encoders are not trained).
            *   Computes `add_time_ids` (containing original size, crop coordinates, and target size) for SDXL.
            *   Predicts noise with the (LoRAfied) UNet, passing it noisy latents, timesteps, prompt embeddings, and `added_cond_kwargs` (which include `pooled_prompt_embeds` and `add_time_ids`).
        *   **Loss Calculation**: Computes MSE (Mean Squared Error) loss between the UNet's predicted noise and the original added noise.
        *   **Backward Pass and Optimization**:
            *   Performs backpropagation (`loss.backward()`).
            *   Applies gradient clipping (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients.
            *   Updates weights with the optimizer (`optimizer.step()`).
            *   Updates the learning rate scheduler (`lr_scheduler.step()`).
            *   Uses `torch.amp.GradScaler` if `fp16` mixed precision is enabled.
        *   Logs progress (epoch, step, loss).
        *   **Checkpoint Saving**: If `save_every_n_epochs` is configured, saves LoRA weights of the UNet (and text encoders if trained) to a `checkpoints` subfolder. Uses `unet.save_pretrained(..., adapter_name="default")`.
    7.  **End of Training**:
        *   If training is stopped by the user, logs a message and cleans up.
        *   **Final LoRA Saving**:
            *   Creates the final destination folder in `LORAS_DIR / lora_output_name`.
            *   Retrieves `state_dict` of LoRA adapters from UNet (and text encoders) via `get_peft_model_state_dict`.
            *   Uses `StableDiffusionXLPipeline.save_lora_weights` to save LoRA weights in `.safetensors` format compatible with Diffusers.
        *   Logs the end of training and total time.
    8.  **Cleanup**: Deletes large objects (models, dataloader, etc.) and empties the CUDA cache.

#### Prompt Encoding (`_encode_prompt`)

*   **`_encode_prompt(self, text_encoders, tokenizers, prompt_text_list, text_input_ids_list=None)`**:
    *   Takes as input the two SDXL text encoders, their respective tokenizers, and either a list of prompt strings or a list of pre-tokenized token IDs.
    *   For each encoder:
        *   Tokenizes prompts (if `text_input_ids_list` is not provided).
        *   Passes token IDs through the text encoder.
        *   Extracts embeddings from the penultimate hidden layer (`prompt_embeds_out[2][-2]`).
        *   The second encoder (CLIPTextModelWithProjection) also provides `pooled_prompt_embeds` (`prompt_embeds_out[0]`).
    *   Concatenates embeddings from both encoders to form `final_prompt_embeds`.
    *   Returns `final_prompt_embeds` and `pooled_prompt_embeds`.

#### `time_ids` Computation (`_compute_time_ids`)

*   **`_compute_time_ids(self, original_size_hw, crop_coords_top_left_yx, target_size_hw, device, dtype)`**:
    *   Utility function for SDXL that prepares `add_time_ids`.
    *   These IDs encode the original image size, crop coordinates, and target image size.
    *   Returns a tensor containing this information, formatted for use by the UNet.

---

## 4. Workflow

1.  **User**: Provides the path to source images, a trigger word, a concept name, and configures training parameters via the Gradio interface.
2.  **Click "Prepare and Train"**:
    *   `run_preparation_and_training_thread` is called.
    *   A new thread is launched to execute `_actual_preparation_and_training`.
3.  **Data Preparation (Thread)**:
    *   Images are copied, cropped, and saved in a project folder.
    *   Captions are generated (Florence-2) and saved in `.txt` files, prefixed with the trigger word.
4.  **Model Loading and Configuration (Thread)**:
    *   The base SDXL model is loaded.
    *   LoRA adapters are added to the UNet (and text encoders if specified).
5.  **Training (Thread)**:
    *   `DreamBoothDataset` and `DataLoader` are created.
    *   Optimizer and scheduler are configured.
    *   The main training loop runs, updating LoRA weights.
    *   Checkpoints are saved periodically.
6.  **Final Saving (Thread)**:
    *   The final LoRA is saved in `.safetensors` format in the `LORAS_DIR` directory.
7.  **UI Update (Main Thread)**:
    *   Throughout the process, log messages sent via the `log_queue` by the training thread are displayed in the interface.
    *   Buttons are enabled/disabled accordingly.

---

## 5. Dependencies

*   **PyTorch**: For tensor operations and neural network training.
*   **Diffusers (Hugging Face)**: For SDXL pipelines, models (UNet, VAE, Schedulers), and training utilities.
*   **Transformers (Hugging Face)**: For tokenizers and text encoder models (CLIP).
*   **PEFT (Hugging Face)**: For LoRA implementation (`LoraConfig`, `get_peft_model_state_dict`, etc.).
*   **Gradio**: For creating the user interface.
*   **PIL (Pillow)**: For image manipulation.
*   **bitsandbytes** (Optional): For the AdamW8bit optimizer (memory saving).
*   **lion-pytorch** (Optional): For the Lion optimizer.
*   **tqdm**: For console progress bars.

---

## 6. Configuration

The module uses `global_config` (a dictionary passed during initialization) for certain paths and parameters:

*   `LORA_PROJECTS_DIR`: Root folder where temporary data and checkpoints for each LoRA training project are stored. (Default: "LoRA_Projects_Internal")
*   `LORAS_DIR`: Folder where final LoRA files (`.safetensors`) are saved for use by the application. (Default: "models/loras")
*   `DEFAULT_SDXL_MODEL_FOR_LORA`: Path to the `.safetensors` file of the base SDXL model to be used by default for training.
*   `MODELS_DIR`: Used to locate the base model if `DEFAULT_SDXL_MODEL_FOR_LORA` is a relative filename.

---

## 7. Usage Notes and Best Practices

*   **GPU Memory (VRAM)**: LoRA training, even if "parameter-efficient," can be VRAM-intensive, especially for SDXL.
    *   Use mixed precision (`fp16` or `bf16` if your GPU supports it) to reduce VRAM usage.
    *   The `AdamW8bit` optimizer (via `bitsandbytes`) can also help reduce memory consumption.
    *   Reduce `batch_size` if you encounter "Out of Memory" (OOM) errors. A batch size of 1 is often necessary for GPUs with less VRAM.
    *   `resolution` has a significant impact on VRAM. 1024x1024 is standard for SDXL.
*   **Data Quality**: The quality and consistency of your training images are crucial.
    *   Use clear, well-framed images of the concept you want to train.
    *   Ensure that generated (or manual) captions accurately describe the image content and include the trigger word.
*   **Hyperparameters**:
    *   **Learning Rate**: One of the most important parameters. Start with a value like `1e-4` or `5e-5` and adjust. Too high a rate can "burn" the model (poor quality results), too low a rate can make training very slow.
    *   **Number of Epochs**: Depends on the size of your dataset. For small datasets (10-20 images), a few dozen epochs may suffice. For larger datasets, fewer epochs may be needed. Monitor the loss.
    *   **Network Dimension (Rank)**: A higher rank (e.g., 64, 128) allows the LoRA to learn more details but increases file size and training time. A lower rank (e.g., 8, 16, 32) is lighter.
    *   **Network Alpha**: Often set to half of `network_dim` or equal to `network_dim`. It acts as a scaling factor for LoRA weights.
*   **Trigger Word**: Choose a unique word that is not common to avoid conflicts with existing concepts in the base model.
*   **Saving Checkpoints**: Use `save_every_n_epochs` to save intermediate versions of your LoRA. This allows you to revert if training diverges or if you want to test different stages.
*   **Stopping Training**: The "Stop" button allows for a clean interruption of training. The LoRA will not be saved in its final version if stopped prematurely, but intermediate checkpoints (if enabled) will be kept.
*   **Cleanup**: The project folder (`LORA_PROJECTS_DIR / your_lora_name`) can become large with data and checkpoints. Consider cleaning it up manually after obtaining a satisfactory final LoRA.

