from evaluate import load
import os
import json
# from ..utils.load_huggingface_dataset import get_context_merged_datset

def calculate_rouge_score(generated_answer: list, ground_truth_answer: list):
    rogue = evaluate.load('rouge')
    
    return rouge.compute(predictions=generated_answer, references=ground_truth_answer)

def load_answers(file_name: str):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return None

def evaluate(dataset, answers_file):

    # answers_file = 'output/query_results.json'

    answers = load_answers(answers_file)

    generated_answers = []
    for answer in answers:
        generated_answers.append(answer['global_answer'])
        
    # squad_df = get_context_merged_datset(dataset_name="rajpurkar/squad", split='train')

    ground_truth_answers = []
    for _, row in dataset.iterrows():
        for answer in row['answer']:
            ground_truth_answer.append(answer['text'][0])

    return calculate_rouge_score(generate_answers, ground_truth_answers)