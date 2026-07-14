import json
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/utils')))

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from load_huggingface_dataset import get_context_merged_datset
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Normalize text function
def normalize_text(text):
    """Normalize text by converting to lowercase, removing special characters, and stripping spaces."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.lower().strip()

# Load the merged dataset using your custom function
print("Loading merged dataset...")
merged_df = get_context_merged_datset()
questions = merged_df['question'].tolist()  # Convert the 'question' column to a list
print(f"Loaded {len(questions)} questions from the merged dataset.\n")

# Load the JSON file containing queries and answers
file_name = 'query_results8.json'
print(f"Loading queries and answers from {file_name}...")
with open(file_name, 'r', encoding='utf-8') as file:
    query_data = json.load(file)
print(f"Loaded {len(query_data)} queries.\n")

# Format queries and answers directly
print("Formatting queries and answers...")
generated_dict = {normalize_text(item['query']): item['global_answer'] for item in query_data}
print(f"Formatted {len(generated_dict)} queries and answers.")
print("Top 5 formatted queries and answers:")
for i, (query, answer) in enumerate(generated_dict.items()):
    print(f"Q: {query}")
    print(f"A: {answer}\n")
    if i == 4:  # Print only the top 5
        break

# Match questions with reference answers from merged dataset
print("\nMatching questions with reference answers from the merged dataset...")
flattened_questions = {}
for index, row in merged_df.iterrows():
    question_list = row['question'] if isinstance(row['question'], list) else [row['question']]
    if isinstance(row['answers'], list) and row['answers']:
        # Extract the first answer's text
        first_answer = row['answers'][0].get('text', [])
        answer = first_answer[0] if len(first_answer) > 0 else ""
    else:
        answer = ""

    # Flatten the question list and normalize each question
    for question in question_list:
        normalized_question = normalize_text(question)
        flattened_questions[normalized_question] = answer

# Normalize the questions from query_results4.json
normalized_generated_dict = {normalize_text(q): a for q, a in generated_dict.items()}

# Match normalized questions
matched_answers = {}
for question, answer in normalized_generated_dict.items():
    if question in flattened_questions:
        matched_answers[question] = flattened_questions[question]

print(f"Matched {len(matched_answers)} questions with reference answers.")
print("Top 5 matched questions and reference answers:")
for i, (question, answer) in enumerate(matched_answers.items()):
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    if i == 4:  # Print only the top 5
        break

# Align generated answers with matched reference answers
print("\nAligning generated answers with matched reference answers...")
references = []
candidates = []
answered_questions = 0
correct_answers = 0

for question, ref_answer in matched_answers.items():
    if question in normalized_generated_dict:  # Ensure the normalized key is used
        references.append(ref_answer)  # Reference as plain strings
        candidate = normalized_generated_dict[question]
        candidates.append(candidate)  # Candidate as plain strings
        answered_questions += 1
        if normalize_text(ref_answer) == normalize_text(candidate):
            correct_answers += 1
    else:
        print(f"Warning: No generated answer for question: {question}")

print(f"Aligned {len(references)} references and candidates.")
print("Top 5 aligned references and candidates:")
for i, (ref, cand) in enumerate(zip(references, candidates)):
    print(f"Reference: {ref}")
    print(f"Candidate: {cand}\n")
    if i == 4:  # Print only the top 5
        break


# Calculate BLEU scores
print("\nCalculating BLEU scores...")
chencherry = SmoothingFunction()

# Calculate BLEU-1 (1-gram precision) for each sentence
sentence_bleu_scores = []
for ref, cand in zip(references, candidates):
    score = sentence_bleu(ref, cand, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    sentence_bleu_scores.append(score)

# Calculate BLEU-1 for the corpus
corpus_bleu_score = corpus_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)

# Calculate BERTScore
print("\nCalculating BERTScore...")
bert_precision, bert_recall, bert_f1 = bert_score(candidates, references, lang='en', rescale_with_baseline=True, batch_size=8)

# Calculate ROUGE
print("\nCalculating ROUGE scores...")
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [rouge_scorer_obj.score(ref, cand) for ref, cand in zip(references, candidates)]

# Average ROUGE scores
avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
avg_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)

# Save results to a text file
output_file_combined = 'questions_answers.txt'
with open(output_file_combined, 'w', encoding='utf-8') as file:
    for question, answer in generated_dict.items():
        file.write(f"Q: {question}\n")
        file.write(f"A: {answer}\n\n")

with open('metrics_results.txt', 'w') as result_file:
    result_file.write("Overall Statistics:\n")
    result_file.write(f"Total Questions: {len(normalized_generated_dict)}\n")
    result_file.write(f"Total Answered Questions: {answered_questions}\n")
    result_file.write(f"Correctly Answered Questions: {correct_answers}\n")
    result_file.write(f"Accuracy: {correct_answers / answered_questions:.4f}\n\n")

    result_file.write("BLEU Scores:\n")
    result_file.write(f"Corpus-level BLEU-1 score: {corpus_bleu_score:.4f}\n\n")

    result_file.write("BERTScore:\n")
    result_file.write(f"Precision: {sum(bert_precision) / len(bert_precision):.4f}\n")
    result_file.write(f"Recall: {sum(bert_recall) / len(bert_recall):.4f}\n")
    result_file.write(f"F1: {sum(bert_f1) / len(bert_f1):.4f}\n\n")

    result_file.write("ROUGE Scores:\n")
    result_file.write(f"ROUGE-1 F1: {avg_rouge1:.4f}\n")
    result_file.write(f"ROUGE-2 F1: {avg_rouge2:.4f}\n")
    result_file.write(f"ROUGE-L F1: {avg_rougeL:.4f}\n")

print(f"Results have been saved to 'metrics_results.txt' and '{output_file_combined}'.")