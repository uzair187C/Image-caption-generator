from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import torch
from PIL import Image

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(img):
    image = Image.fromarray(img)
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

# Gradio UI
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="AI Image Caption Generator",
    description="Upload an image and get an AI-generated caption!"
)

iface.launch()
