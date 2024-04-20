import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

import requests
from io import BytesIO
from cog import BasePredictor, Input, Path, ConcatenateIterator

import os
import csv
import glob
from PIL import Image
from datetime import datetime

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
    # Define
    predictor = Predictor()
    data_paths = glob.glob('/data/mvtec_test_for_icl/*/*/*')
    weight_path = ["checkpoints/rename_view_scraped_whole_bbox/vip-llava-7b/20240419-201050", 'checkpoints/rename_view_scraped_no_bias/vip-llava-7b/20240419-204520', 'checkpoints/rename_full_scraped_whole_bbox/vip-llava-7b/20240420-024724', 'checkpoints/rename_full_scraped_no_bias/vip-llava-7b/20240420-044328']
    output_path = ['./mvtec_result/rename_view_scraped_whole_bbox', './mvtec_result/rename_view_scraped_no_bias', './mvtec_result/rename_full_scraped_whole_bbox', './mvtec_result/rename_full_scraped_no_bias']

    for idx in range(len(weight_path)):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(output_path[idx], timestamp),  exist_ok=True)

        # Setup
        with open(os.path.join(output_path[idx], timestamp, 'output_full_text.csv'), 'w') as f: pass
        predictor.setup(weight_path=weight_path[idx])
        base_input_text = "This is an image of {category}. Does this {category} in the image have any defects? If yes, please provide the bounding box coordinate of the region where the defect is located."
        correct = 0; total = 0
        tmp_category = data_paths[0].split('/')[3]
        tmp_mode = data_paths[0].split('/')[4]

        # Predict
        for data_path in data_paths:
            input_text = base_input_text.format(category=data_path.split('/')[-3])
            # RUN
            for i, text in enumerate(predictor.predict(image=data_path, prompt=input_text, temperature=0.2)):
                # Output category, mode, and accuracy
                if tmp_mode != data_path.split('/')[4]:
                    with open(os.path.join(output_path[idx], timestamp, 'output_full_text.csv'), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([tmp_category, tmp_mode, f'{correct}/{total}'])
                        correct = 0; total = 0; tmp_category = data_path.split('/')[3]; tmp_mode = data_path.split('/')[4]

                print(data_path, end=':')
                print(text.split('ASSISTANT: ')[-1])
                if text.split('ASSISTANT: ')[-1] == 'None.' and data_path.split('/')[4] == 'good':
                    correct += 1
                    total += 1
                    true_or_false = True
                elif text.split('ASSISTANT: ')[-1] != 'None.' and data_path.split('/')[4] != 'good':
                    correct += 1
                    total += 1
                    true_or_false = True
                else:
                    total += 1
                    true_or_false = False
                # Output path, answer, and true or false
                with open(os.path.join(output_path[idx], timestamp, 'output_full_text.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([data_path, text.split('ASSISTANT: ')[-1], true_or_false])
        with open(os.path.join(output_path[idx], timestamp, 'output_full_text.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([data_path.split('/')[3], data_path.split('/')[4], f'{correct}/{total}'])