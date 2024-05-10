import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
import pandas as pd
import os
import sys

class Blip2:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

    def generate_prompted_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        prompt = "A drawing/sketch of a"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        prompted_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return prompted_caption
    
    def generate_answer(self,image_path, question):
        image = Image.open(image_path).convert("RGB")
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return answer

def main(type):
    current_directory = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(current_directory, '../Dataset'))

    # Load the Dataset
    sketch_path = path + "/sketch/"+type+"/"
    question_path = path + "/qa/"+type+".csv"
    qa = pd.read_csv(question_path)
    
    blip2 = Blip2()
    folder_name = ''
    folder_id = 0
    # 이미지를 순회하며 Blip2 모델을 이용해 답변 생성 및 데이터 프레임으로 저장
    df = pd.DataFrame(columns=['image_id', 'question', 'true_answer','blip2_caption', 'blip2_prompted_caption', 'blip2_answer'])
    for i in tqdm(range(len(qa))):
        image_id = qa['image_id'][i]
        question = qa['question'][i]
        true_answer = qa['answer'][i]
        # 폴더 아이디는 folder_name이 달라질 때마다 1씩 증가 
        if folder_name != qa['image_id'][i].split('(')[0]:
            folder_id += 1
            folder_name = qa['image_id'][i].split('(')[0]
        folder_path = str(folder_id) + '.' + folder_name
        image_path = sketch_path + folder_path + '/' + image_id + '.png'
        blip2.image = Image.open(image_path).convert("RGB")  # Update the image for each iteration
        blip2_caption = blip2.generate_caption(image_path)
        blip2_prompted_caption = blip2.generate_prompted_caption(image_path)
        blip2_answer = blip2.generate_answer(image_path, question)
        
        df.loc[i] = [image_id, question, true_answer, blip2_caption, blip2_prompted_caption, blip2_answer]

    # Save the DataFrame
    df.to_csv("SketchVQA/Dataset/blip2/"+type+"_blip2.csv", index=False)

# python 실행 bash, filename을 인자로 받아서 실행
# python Blip2/Blip2.py filename 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('train')
        main('test')