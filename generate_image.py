import replicate
import time
from PIL import Image
import requests
from io import BytesIO
import os
import base64

def generate_image(prompt, num_outputs, output_format, width, height, main_face_image_path, 
                   true_cfg=1.0, id_weight=1.05, num_steps=20, start_step=0, guidance_scale=4.0, 
                   negative_prompt="bad quality, worst quality, text, signature, watermark, extra limbs, low resolution, partially rendered objects, deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed, blurry", 
                   max_sequence_length=128, output_quality=100):
    # Start the image generation process
    start_time = time.time()

    # Set the output format dynamically
    output_format = output_format.lower()

    # Read the image file and convert it to base64
    with open(main_face_image_path, "rb") as image_file:
        main_face_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    output = replicate.run(
        "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
        input={
            "width": width,
            "height": height,
            "prompt": prompt,
            "true_cfg": true_cfg,
            "id_weight": id_weight,
            "num_steps": num_steps,
            "start_step": start_step,
            "num_outputs": num_outputs,
            "output_format": output_format,
            "guidance_scale": guidance_scale,
            "output_quality": output_quality,
            "main_face_image": f"data:image/jpeg;base64,{main_face_image_base64}",  # Use base64 data URI
            "negative_prompt": negative_prompt,
            "max_sequence_length": max_sequence_length
        }
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert output URLs to PIL Images
    generated_images = []
    for img_url in output:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        generated_images.append(img)

    return generated_images, elapsed_time
