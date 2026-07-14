from transformers import AutoTokenizer, AutoModelForCausalLM
from test import py_dict  # Import the graph data from the external file

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'], 
        max_new_tokens=2000, 
        do_sample=True, 
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print(f'RESPONSE FROM ENTITY TYPES: {response_text}')
    return response_text

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# print(len(py_dict))

for i in range(1, len(py_dict)+1):
    # print(py_dict[i])
    if py_dict[i]['relationships'] != []:
        cluster_summary = py_dict[i]['summary']
        
        prompt = """
            Answer the following query based on the provided context. Provide only the answer with no additional text. Ensure the answer is precise and directly supported by the context provided.

            ######################
            **Examples**
            ######################

            Example 1:
            Query: Who discovered penicillin?
            Context: 
            Entities in this community:
            - Alexander Fleming, a scientist, made the groundbreaking discovery of penicillin in 1928 when he observed a mold called Penicillium notatum inhibiting bacterial growth in his laboratory. 
            - This discovery revolutionized medicine and led to the development of antibiotics, saving countless lives.

            Relationships:
            - Penicillin and Alexander Fleming are related by Discovery (strength: 10).
            - Alexander Fleming and Medicine are related by Contribution (strength: 9).

            Answer: Alexander Fleming
            ######################

            Example 2:
            Query: What is the capital of France?
            Context: 
            Entities in this community:
            - Paris, the capital city of France, is renowned for its cultural landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. 
            - It is a global hub for art, fashion, and gastronomy.
            - France is a country in Western Europe, bordered by countries like Germany and Spain, and known for its rich history and diverse culture.

            Relationships:
            - France and Paris are related by Capital-City (strength: 10).
            - Paris and Eiffel Tower are related by Landmark-Location (strength: 9).

            Answer: Paris
            ######################

            Example 3:
            Query: What is the tallest mountain in the world?
            Context: 
            Entities in this community:
            - Mount Everest, part of the Himalayan mountain range, is the tallest mountain in the world, standing at 8,848.86 meters above sea level.
            - It is located on the border between Nepal and Tibet and has been a historic destination for climbers.
            - The mountain holds significant cultural and spiritual importance for the local Sherpa community and is a major source of tourism for the region.

            Relationships:
            - Mount Everest and Height are related by Measurement (strength: 10).
            - Mount Everest and Nepal are related by Location (strength: 8).
            - Mount Everest and Sherpa are related by Cultural-Significance (strength: 7).

            Answer: Mount Everest
            ######################

            **Real Query**

            Query: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes, France?
            Context: 
            {cluster_summary}

            Answer:
            ######################

        """.format(cluster_summary=cluster_summary)
        
        print(generate_answer(model, tokenizer, prompt))