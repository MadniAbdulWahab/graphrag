from ..utils.prompts import ENTITY_SUMMARIZATION_PROMPT

def create_discription_list(relationships, entity):
    description_list=[entity['description']]
    for relation in relationships:
        if relation['source']==entity['name'] or relation['target']==entity['name']:
            description_list.append(relation['relationship'])
    return description_list

def prompt_llm_enetity_summaries(model, tokenizer, device, prompt):
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


def extract_summary(text):
    last_occurrence_index = text.rfind("Output:")
    if last_occurrence_index != -1:
        # Extract the substring starting after "Output:"
        extracted_substring = text[last_occurrence_index + len("Output:"):].strip()
        
        paragraphs = extracted_substring.split("\n\n")
        first_paragraph = paragraphs[0].strip() if paragraphs else ""
 
        return first_paragraph
    else:
        return None


def get_entity_summaries(entity_relations ,persona, model, tokenizer, device):
    if entity_relations is not None:
        entities = [item for item in entity_relations if 'name' in item and 'type' in item and 'description' in item]
        relationships = [item for item in entity_relations if 'source' in item and 'target' in item and 'relationship' in item and 'relationship_strength' in item]
        entity_names = [item['name'] for item in entities]

    #! print(f'ENTITIES: {entities}')
    #! print(f'RELATIONSHIPS: {relationships}')


    
    entity_summaries = []
    for entity in entities:
        description_list = create_discription_list(relationships, entity)
        #! print(f'DESCRIPTION LIST: {description_list}') 
        entity_summarization_prompt = ENTITY_SUMMARIZATION_PROMPT.format(
            persona = persona,
            entity_name=entity['name'],
            descriptions="\n - ".join(description_list)
        )
        
        count = 0
                
        while count <= 3:
            response_text = prompt_llm_enetity_summaries(model, tokenizer, device, entity_summarization_prompt)
            entity_summary = extract_summary(response_text)

            if entity_summary is None:
                count += 1
            else:
                break
        
        summary_item = {
            'Entity': entity['name'],
            'Type': entity['type'],
            'Summary': entity_summary
        }
        entity_summaries.append(summary_item)    
        
    return entity_summaries, entities, relationships