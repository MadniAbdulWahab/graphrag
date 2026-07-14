import faiss
import numpy as np
import re
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from test import py_dict  # Import the graph data from the external file

# Set the default device to trun the models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Process Provided Graph Data
def process_graph(py_dict):
    community_summaries = []
    for community_id, community_data in py_dict.items():
        summary = community_data['summary']
        if community_data['relationships'] != []:
            community_summaries.append({"community_id": community_id, "summary": summary})
    return community_summaries

# Step 2: Embedding and Indexing
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

def index_summaries(community_summaries):
    summaries = [item["summary"] for item in community_summaries]
    summary_embeddings = embedding_model.encode(summaries)
    index.add(np.array(summary_embeddings))
    return summaries

# Step 3: Retrieval
def retrieve_relevant_communities(query, summaries, community_summaries, top_k=2):
    #print("[DEBUG] Retrieving relevant communities...")
    query_embedding = embedding_model.encode([query])
    if len(summaries) < top_k:
        top_k = len(summaries)
    _, indices = index.search(query_embedding, top_k)
    retrieved = [{"community_id": community_summaries[idx]["community_id"], "summary": summaries[idx]} for idx in indices[0]]
    #print(f"[DEBUG] Retrieved Communities: {retrieved}")
    return retrieved

# Step 4: Answer Generation
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)

def score_answer(query, answer):
    input_text = (
        f"Query: {query}\n"
        f"Answer: {answer}\n"
        "Rate the relevance of this answer to the query on a scale of 0 to 100. Respond with only the score."
    )
    try:
        score_output = generator(input_text, max_new_tokens=5, num_return_sequences=1, truncation=True)
        score_text = score_output[0]["generated_text"].strip()
        match = re.search(r"\\b\\d+\\b", score_text)
        score = int(match.group()) if match else 0
    except Exception as e:
        #print(f"[ERROR] Scoring failed: {e}")
        score = 0  # Default score
    return score

def extract_relevant_answer(generated_text, query):
    # Find the part where the query is repeated
    query_start = generated_text.find(f"Answer the query based on the context: {query}")
    if query_start != -1:
        # Extract the text after the repeated query
        relevant_part = generated_text[query_start + len(f"Answer the query based on the context: {query}"):]
        return relevant_part.strip()  # Remove leading/trailing whitespace
    return generated_text.strip()  # Fallback to the entire text if no match

def extract_answer_from_text(text, question):
    # Case-insensitive regex to find all instances of "Answer:"
    pattern = re.compile(r"the final answer is:\s*", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    if matches:
        # Find the position after the last occurrence
        last_match = matches[-1]
        start_pos = last_match.end()
        
        # Extract text after the last "Answer:"
        remaining_text = text[start_pos:]
        
        # Find the first sentence using regex
        first_sentence = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', remaining_text, 1)[0]
        
        # Remove "The final answer is:" if present (case-insensitive)
        # clean_sentence = re.sub(r"the final answer is:\s*", "", first_sentence, flags=re.IGNORECASE)
        clean_sentence = re.sub(question, "", first_sentence, flags=re.IGNORECASE)
        
        
        return clean_sentence.strip()
    else:
        return "No 'Answer:' phrase found."

def generate_answer(query, retrieved_communities):
    print("[DEBUG] Generating answers...")
    
    
    answers = []
    for community in retrieved_communities:
        print(f'[COMMUNITY ID] {community["community_id"]}')
        # Construct the input prompt
        QUESTION_ANSWER_PROMPT = """
            You are an expert AI informaition retrieving system. Your goal is to answer the user's question by analyzing the provided context, which includes entities, their features, and relationships. Your task is to carefully interpret the context and use it to generate an accurate answer to the question.

            ======================================================================
            EXAMPLE SECTION: The following section includes example outputs. These examples **must be excluded from your answer**.

            EXAMPLE 1  
            Question: Where is the statue of Saint George located?  
            Context:  
            Entities in this community:  
            The city park features a grand statue of Saint George, a heroic figure in local folklore. This statue stands near the central fountain and overlooks the main entrance to the art museum. The surrounding area includes several benches and a small garden.  
            Relationships:  
            City park and Saint George are related by The city park contains the statue of Saint George (strength: 10).  
            City park and fountain are related by The city park contains a central fountain (strength: 9).  
            Answer: city park, near the central fountain, overlooking the main entrance to the art museum
            END OF EXAMPLE 1  

            EXAMPLE 2  
            Question: What is the significance of the Eiffel Tower?  
            Context:  
            Entities in this community:  
            The Eiffel Tower, located in Paris, is a renowned landmark, originally constructed for the 1889 World's Fair. The tower stands as a symbol of French art and engineering and is a major tourist attraction. Visitors often climb to the top to enjoy panoramic views of the city.  
            Relationships:  
            Eiffel Tower and France are related by The Eiffel Tower represents France's cultural identity (strength: 10).  
            Eiffel Tower and World's Fair are related by The Eiffel Tower was constructed for the 1889 World's Fair (strength: 8).  
            Answer: The Eiffel Tower is a symbol of French art and engineering, constructed for the 1889 World's Fair. 
            END OF EXAMPLE 2  

            EXAMPLE 3  
            Question: What buildings feature statues of Christ?  
            Context:  
            Entities in this community:  
            Several buildings in the historical district feature statues of Christ, including the cathedral at the heart of the city. The statue at the cathedral is located at the altar and is known for its intricate design. Additionally, the chapel near the western edge of the district contains a smaller but equally significant statue of Christ near the entrance.  
            Relationships:  
            Cathedral and Christ are related by The cathedral features a statue of Christ at the altar (strength: 10).  
            Chapel and Christ are related by The chapel contains a statue of Christ near the entrance (strength: 9).  
            ANSWER: cathedral at the heart of the city, chapel near the western edge of the district  
            END OF EXAMPLE 3  

            ======================================================================
            REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate the answer only.

            Question: {question}  
            Context: {context}  
            Answer:         
        """.format(
            question=query, context=community['summary']
        )
        
        inputs = tokenizer(QUESTION_ANSWER_PROMPT, return_tensors="pt").to(device=1)
        outputs = model.generate(
            inputs['input_ids'], 
            max_new_tokens=2000, 
            do_sample=True, 
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Generate text
        # generated_text = generator(QUESTION_ANSWER_PROMPT, max_new_tokens=500, num_return_sequences=1, truncation=True)[0]["generated_text"].strip()
        # print(f'[GENERATED TEXT]: {response_text}')        
        
        # Extract answer from LLM response
        extracted_answer = extract_answer_from_text(response_text, query)
        # print(f'[EXTRACTED ANSWER]: {extracted_answer}')        
        
        # print(f"[DEBUG] Generated Answer for Community {community['community_id']}: {relevant_answer}")
    return extracted_answer

# Step 5: Global Answer Refinement
def reduce_to_global_answer(answers, token_limit=200):
    # #print("[DEBUG] Reducing to global answer...")
    answers = sorted(answers, key=lambda x: x["score"], reverse=True)
    global_answer, current_token_count = "", 0
    for answer in answers:
        tokens = len(answer["answer"].split())
        if current_token_count + tokens > token_limit:
            break
        global_answer += answer["answer"] + "\n\n"
        current_token_count += tokens
    if not global_answer.strip():
        global_answer = "No relevant answer could be generated."
    #print(f"[DEBUG] Global Answer: {global_answer.strip()}")
    return global_answer.strip()

def filter_empty_communities(retrieved_communities):
    return [community for community in retrieved_communities if community["summary"].strip()]



# Step 6: Full Pipeline Execution
# def main():
community_summaries = process_graph(py_dict)
# print(f'[COMMUNITY SUMMARIES] {community_summaries}')
# print(f'[LENGTH COMMUNITY SUMMARIES] {len(community_summaries)}')

summaries = index_summaries(community_summaries)
# print(f'[SUMMARIES] {summaries}')
# print(f'[LENGTH SUMMARIES] {len(summaries)}')

query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
retrieved_communities = retrieve_relevant_communities(query, summaries, community_summaries, top_k=2)
# print(f'[RETRIEVED COMMUNITIES] {retrieved_communities}')
# print(f'[LENGTH RETRIEVED COMMUNITIES] {len(retrieved_communities)}')

answer = generate_answer(query, retrieved_communities)
print(answer)

