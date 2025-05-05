# test_sana.py
import torch
from diffusers import SanaSprintPipeline
import time
import os

# --- Configuration ---
SANA_MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
PROMPT = "a tiny astronaut hatching from an egg on the moon"
NUM_STEPS = 2 # Keep it low for a quick test
OUTPUT_FILENAME = "sana_sprint_test_output.png"

def main():
    print("--- Starting Sana Sprint Test ---")

    # --- Check Device and Data Type ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use bfloat16 if supported, otherwise float16, fallback to float32
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using device: CUDA with torch.bfloat16")
        else:
            dtype = torch.float16
            print("Using device: CUDA with torch.float16")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using device: CPU with torch.float32")

    # --- Load Pipeline ---
    try:
        print(f"Loading pipeline: {SANA_MODEL_ID}...")
        start_load_time = time.time()
        pipeline = SanaSprintPipeline.from_pretrained(
            SANA_MODEL_ID,
            torch_dtype=dtype
        )
        pipeline.to(device)
        load_time = time.time() - start_load_time
        print(f"Pipeline loaded successfully in {load_time:.2f} seconds.")

    except Exception as e:
        print(f"\n--- ERROR loading pipeline ---")
        print(f"Model ID: {SANA_MODEL_ID}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("-----------------------------")
        return # Stop execution if loading fails

    # --- Generate Image ---
    try:
        print(f"\nGenerating image with prompt: '{PROMPT}'...")
        print(f"Steps: {NUM_STEPS}")
        start_gen_time = time.time()

        # Note: Sana Sprint might not use guidance_scale or negative_prompt effectively,
        # but we pass the prompt directly as per its basic usage.
        image = pipeline(
            prompt=PROMPT,
            num_inference_steps=NUM_STEPS
            # Add other relevant parameters if needed based on pipeline documentation
            # e.g., generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]

        gen_time = time.time() - start_gen_time
        print(f"Image generated successfully in {gen_time:.2f} seconds.")

    except Exception as e:
        print(f"\n--- ERROR during image generation ---")
        print(f"Prompt: {PROMPT}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
        return # Stop execution if generation fails

    # --- Save Image ---
    try:
        print(f"\nSaving image to: {OUTPUT_FILENAME}...")
        image.save(OUTPUT_FILENAME)
        print(f"Image saved successfully.")

    except Exception as e:
        print(f"\n--- ERROR saving image ---")
        print(f"Filename: {OUTPUT_FILENAME}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("-------------------------")
        return

    print("\n--- Sana Sprint Test Completed ---")

if __name__ == "__main__":
    main()
