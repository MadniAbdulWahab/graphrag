from transformers import AutoModelForCausalLM, AutoTokenizer

context = """The BBC, a renowned UK-based broadcaster, played a pivotal role in reconstructing the missing episodes of The Invasion, collaborating with various production companies to bring these episodes to life. The corporation's efforts not only preserved the history of Doctor Who but also made these episodes accessible to fans through DVD and MP3 CD-ROM releases. The reconstruction of these episodes highlights the BBC's commitment to preserving the show's history and legacy. Furthermore, the BBC's archives, which date back to 1964, contain a wealth of information about Doctor Who's past, including its video tape and film libraries. The preservation of these archives in 1972 marked a significant milestone in the BBC's efforts to safeguard the show's history. The decision to stop wiping tapes and destroying spare film copies in 1978 ensured the long-term preservation of the show's episodes, providing a unique glimpse into the world of Doctor Who for future generations. Through its efforts, the BBC has not only preserved the show's history but also made it accessible to fans worldwide."""
query = "When was the Doctor Who series released on DVD?"
instructions = """Instructions:
1. If the context contains sufficient information to answer the query, provide a detailed answer.
2. If the context does not contain the exact information, infer the most likely answer based on the given context.
3. If the context does not contain sufficient information to answer the query, respond with:
'Answer: Not found in the provided context.'
4. Assign a score out of 100 based on the relevance and completeness of the context.
Provide your response in the following format:
Answer: [Your answer here]\\nScore: [0-100]"""

# Combine all into a prompt
prompt = f"You are an expert assistant tasked with answering the query strictly based on the given context. Ignore your own knowledge or assumptions.\n\nContext: {context}\n\nQuery: {query}\n\n{instructions}"

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate the response
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    pad_token_id=tokenizer.eos_token_id,
    max_length=500,  # Adjust as needed for longer outputs
    temperature=0.4,  # Control randomness
    top_p=0.9,  # Nucleus sampling
    num_beams=8,
    repetition_penalty=1.1,
    early_stopping=True,
    num_return_sequences=1
)

# Decode the output
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the model's answer
print(response)

# Optional: Clean up and extract the formatted response
response_start = response.find("Answer:")  # Look for "Answer:" in the response
if response_start != -1:
    print(response[response_start:].strip())
else:
    print("Response format not found. Check the model output.")

# from transformers import AutoConfig, AutoModelForCausalLM

# # Load the model configuration
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# config = AutoConfig.from_pretrained(model_name)

# # Inspect all default parameters
# print(config)
