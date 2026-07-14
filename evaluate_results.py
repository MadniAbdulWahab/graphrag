from evaluate import load
import evaluate
import os
import json

from pipeline.utils.load_huggingface_dataset import load_huggingface_dataset

squad_df = load_huggingface_dataset("rajpurkar/squad", split='train')
answers_file = 'output/query_results.json'


# print( squad_df)

def calculate_rouge_score(generated_answer: list, ground_truth_answer: list):
    rouge = evaluate.load('rouge')
    
    return rouge.compute(predictions=generated_answer, references=ground_truth_answer)

def load_answers(file_name: str):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return None

answers = load_answers(answers_file)


generated_answers = []
for answer in answers:
    generated_answers.append(answer['global_answer'])


# print(squad_df)

# print(len(generated_answers))    

ground_truth_answers = []
for i, row in squad_df.iterrows():
    if i == 56:
        break
    ground_truth_answers.append(row['answers']['text'][0])

# print(ground_truth_answers)

print(calculate_rouge_score(generated_answers, ground_truth_answers))