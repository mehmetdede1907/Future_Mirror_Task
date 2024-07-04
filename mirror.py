#hugging face api: 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

import openai
import os
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime,timedelta
import speech_recognition as sr
import requests
import json
from config import openaikey

openai.api_key = openaikey

# Check for MPS availability
#mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# Load the pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    pipe = pipe.to(device)
    #print('gpu  asdadsfadsf \n')
else:
    device = torch.device("cpu")
    pipe = pipe.to(device)
    print('cpu used')

def enhance_description(input_text,token):
    prompt = f"Describe the apperance of '{input_text}' as it might appear 20 years in the future. Be imaginative, as it might appear 20 years in the future. Use a language such that describes lot with less word"
    #insted we may use Completiton but to determine more parameters ChatCompletition is used (Benefitial for learning) :)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # abvious
        messages=[
            {"role": "system", "content": "You are a imaginative and knowlagable assistant."}, # Tested with this message AI trust himself some examples: for insance use bulletpoints
            {"role": "user", "content": prompt}
        ],

        # to look parameter details chech this link
        #https://platform.openai.com/docs/guides/text-generation/parameter-details

        max_tokens= token,#obvious
        temperature=0.9, #Controls the randomness of the response. about generating tokens
        top_p=0.2, # About selecting token. Value of 1.0 means that no filtering is applied (creative) based on cumulative probability, so all tokens are considered. This does not force the model to include only high-probability tokens, but it allows the model to potentially use any token, depending on the temperature #if top p low high probability tokens are selected
        frequency_penalty= 0, # about repetition of tokens when we have small value we dont mind repetiton for example key words for summirizing
        presence_penalty=0.7 #discourages the model from introducing new concepts that weren't present in the prompt,
    )
    #print(response['choices'][0]['message']['content'].strip())
    return response['choices'][0]['message']['content'].strip()

def generate_image(description):
    image = pipe(description).images[0]
    return image


def future_mirror(input_text):
    start_time = datetime.now()
    description = enhance_description(input_text,75)
    print(description)
    image = generate_image(description)
    os.makedirs('output',exist_ok=True) #if file exist no fileexeption error. false is default (fileexeption error)
    filename = f"output/{input_text.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    image.save(filename) #PIL (Python Imaging Library) Image object that Stable Diffusion pipeline typically returns
    end_time = datetime.now()
    process_time = (end_time - start_time).total_seconds()
    metadata = {
        "original_prompt": input_text,
        "image_size": f"{image.width}x{image.height}",
        "model_used": "runwayml/stable-diffusion-v1-5-fp16",
        "processing_time": f"{process_time:.2f} seconds",
        "description": description

    }
    return description , filename , metadata

def generate_image_dalle(description):
    response = openai.Image.create(
        prompt = description,
        n = 1, # number of image generated
        size = "1024x1024"
    )
    image_url = response['data'][0]['url'] #if we have generated 2 image we could write [1]

    #download image image data is received as raw binary content from an HTTP response
    image_responce = requests.get(image_url) 

    if image_responce.status_code == 200: #means successfull
        filename = f"output/dalle_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        with open(filename,'wb') as f:#binary write mode Since DALL-e downlad image data is received as raw binary content from an HTTP response
            f.write(image_responce.content)
        return filename
    else:
        raise Exception('Failed to download DALL-E generated image')

def future_mirror_dalle(input_text):
    start_time = datetime.now()
    description = enhance_description(input_text,150)
    image_path = generate_image_dalle(description)
    end_time = datetime.now()
    process_time = (end_time - start_time).total_seconds()
    metadata = {
        "original_prompt": input_text,
        "image_size": "1024x1024",
        "model_used": "DALL-E",
        "processing_time": f"{process_time:.2f} seconds",
        "description": description
    }
    return description, image_path, metadata