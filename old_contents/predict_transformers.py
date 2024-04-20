import torch
from PIL import Image
from transformers import AutoProcessor, VipLlavaForConditionalGeneration

model = VipLlavaForConditionalGeneration.from_pretrained("model_weights_hf", device_map="auto", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("model_weights_hf")

prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
question = "Can you please describe this image?"
prompt = prompt.format(question)
image = Image.open('/data/mvtec/bottle/test/good/000.png').convert('RGB')

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=20)
processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)