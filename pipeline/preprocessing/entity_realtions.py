# from .prompts import ENTITY_RELATIONSHIPS_GENERATION_PROMPT

import re
import json
from typing import List, Dict, Any
import re
import ast

def prompt_llm_kg(model, tokenizer, device, prompt):

    # prompt = ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(entity_types=entity_types, language='english', input_text=chunk_text)

    #! print(f'String from kg method: {entity_types}')
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=2000, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text


def extract_relations_list(long_string):
    # Regular expression to match JSON lists
    json_list_pattern = r'\[\s*\{.*?\}\s*\]'
    
    # Find all matches of JSON lists in the string
    json_lists = re.findall(json_list_pattern, long_string, re.DOTALL)
    # Check if the 4th list exists
    if len(json_lists) >= 4:
        fourth_list_str = json_lists[3]  # Get the 4th list (0-based index)
        try:
            # Parse the string as JSON to validate it
            return json.loads(fourth_list_str)
        except json.JSONDecodeError as e:
            #! print("Error decoding JSON:", e)
            return None
    else:
        #! print("Less than 4 lists found in the input.")
        return None

def get_relations(model, tokenizer, device, prompt):
    got_entites = False

    while (not got_entites):
        response_text = prompt_llm_kg(model, tokenizer, device, prompt)
        entity_relations_list = extract_relations_list(response_text)
        
        if entity_relations_list is not None:
            got_entites = True

    return entity_relations_list

