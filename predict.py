import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

from PIL import Image

import requests
from io import BytesIO

from cog import BasePredictor, Input, Path, ConcatenateIterator

class Predictor(BasePredictor):
    def setup(self, weight_path) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()
    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(weight_path, model_name="vip-llava-7b", model_base=None, load_8bit=False, load_4bit=False)
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
    
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            output_texts = self.model.generate(
                inputs=input_ids,
                tokenizer=self.tokenizer,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            print(output_texts.shape)
            # Process and yield each text output
            prepend_space = False
            for new_text in output_texts:
                # Remove image token and decode
                vocab_size = self.tokenizer.vocab_size
                new_text = [id if 0<id<vocab_size else 0 for id in new_text]
                new_text = self.tokenizer.decode(new_text, skip_special_tokens=True)
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


if __name__ == "__main__":
    predictor = Predictor()
    weight_path = "/home/ueno/vip-llava/checkpoints/rename_full_no_bias/2e-5/30epoch/vip-llava-7b/20240508-153414"
    imagepath = "/home/ueno/InMeMo/pascal-5i/mvtec/JPEGImages/0.jpg"
    category = "bottle"
    predictor.setup(weight_path=weight_path)

    input_text = f"<image>\nThis is an image of {category}. Does this {category} in the image have any defects? If yes, please provide the bounding box coordinate of the region where the defect is located."
    # input_text = f"This is an image of {category}. Does this {category} in the image have any defects? If yes, please provide the bounding box coordinate of the region where the defect is located."
    for i, text in enumerate(predictor.predict(image=imagepath, prompt=input_text, temperature=1e-10)):
        print(f"Prediction {i}: {text}")