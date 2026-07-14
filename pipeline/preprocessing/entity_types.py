import re
import json

def prompt_llm_entity_types(model, tokenizer, device, prompt):

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_new_tokens=2000, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #! print(f'RESPONSE FROM ENTITY TYPES: {response_text}')
    return response_text

def extract_json_from_text(text):
    # Regular expression to capture JSON object for "entity_types"
    json_pattern = r'{"entity_types":\s*\[.*?\]}'

    # Search for the JSON pattern in the text
    match = re.search(json_pattern, text)
    
    if match:
        # Extract matched JSON string
        json_str = match.group(0)
        
        try:
            # Parse the JSON string to a Python dictionary
            json_data = json.loads(json_str)
            # If entity_types is empty, use the default
            if not json_data.get("entity_types"):
                return None
            return json_data
        except json.JSONDecodeError as e:
            #! print(f"Error decoding JSON: {e}")
            return None
    else:
        #! print("No JSON object found in the text, using default entity types.")
        return None

def get_entity_types(model, tokenizer, device, prompt):
    count = 0
    default_entities = {"entity_types": ["PERSON", "COUNTRY", "CITY" "ORGANIZATION", "DATE", "EVENT", "BUILDING", "CULTUE", "HISTORICAL EVENT"]}

    while count <= 3:
        response_text = prompt_llm_entity_types(model, tokenizer, device, prompt)
        extracted_entities = extract_json_from_text(response_text)

        if extracted_entities is None:
            count += 1
        else: 
            break

    if count > 3:
        extracted_entities = default_entities
        

    #! print (f'RESPOPNSE: {response_text}')
    return extracted_entities
