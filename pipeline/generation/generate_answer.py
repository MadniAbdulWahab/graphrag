from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login


login(token=os.environ["HUGGINGFACE_TOKEN_LLAMA32"])

# Load the Model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)

# Import Graph Data
# from test import py_dict 

# Validate Context
def validate_context(context):
    """Check if the context has enough content to process."""
    return len(context.split()) > 10  # Ensure at least 10 words in the context


# Generate Concise Answers with Improved Prompt
def generate_concise_answer(query, context):
    """Generate an answer based strictly on the context."""
    input_text = (
        f"You are a historian tasked with answering questions. "
        f"Answer the question strictly using only the provided context. "
        f"If the context does not contain relevant information, respond: 'No valid context provided 1.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    result = generator(input_text, max_new_tokens=50, num_return_sequences=1, truncation=True)
    return result[0]["generated_text"]

# Clean Generated Output
def clean_generated_output(output):
    """Clean the generated output and extract the final answer."""
    if "Answer:" in output:
        output = output.split("Answer:")[-1]
    return output.strip().split("\n")[0]  # Take the first line only

# BERTScore Evaluation
def evaluate_with_bertscore(predicted_answer, reference_answer):
    """Evaluate the predicted answer against a reference using BERTScore."""
    P, R, F1 = bert_score([predicted_answer], [reference_answer], lang="en", rescale_with_baseline=True)
    return F1[0].item()

# Main Execution
def generate_answer(cluster, query, reference_answer):
    # query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    # reference_answer = "The Virgin Mary allegedly appeared to Saint Bernadette Soubirous in 1858 in Lourdes, France."

    for community_id, community_data in cluster.items():
        # Retrieve context
        context = community_data.get("summary", "") or "No context provided."

        if not validate_context(context):
            print(f"[FINAL ANSWER for Community {community_id}]: No valid context provided.\n")
            continue

        # Generate and clean answer
        raw_output = generate_concise_answer(query, context)
        cleaned_output = clean_generated_output(raw_output)
        final_answer = cleaned_output

        # BERTScore evaluation
        score = evaluate_with_bertscore(final_answer, reference_answer)

        # print(f"[FINAL ANSWER for Community {community_id}]:\n{final_answer}")
        # print(f"[BERTScore F1 for Community {community_id}]: {score:.4f}\n")
        
        return (final_answer, score)