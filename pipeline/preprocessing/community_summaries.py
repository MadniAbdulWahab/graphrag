from ..utils.prompts import COMMUNITY_SUMMARY_PROMPT

def generate_community_summaries(global_graph, affected_communities, model, tokenizer, device):
    """
    Generate summaries for affected graph communities using an LLM.
    """
    community_summaries = {}

    for community_id in affected_communities:
        # Extract nodes in the community
        nodes_in_community = [
            node for node, data in global_graph.nodes(data=True) 
            if data.get('community_id') == community_id
        ]

        # Prepare entities and text summaries for the prompt
        entities = []
        text_summaries = []

        for node in nodes_in_community:
            entities.append(node)
            if 'description' in global_graph.nodes[node]:
                text_summaries.append(global_graph.nodes[node]['description'])

        # Construct the prompt
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            nodes=entities, 
            node_summaries=text_summaries
        )

        # Retry up to 3 times if the summary generation fails
        for attempt in range(3):
            response = generate_summary_with_llm(model, tokenizer, device, prompt)
            summary = extract_summary(response)
            if summary:
                break
        else:
            summary = "Summary generation failed after multiple attempts."

        # Store the generated summary
        community_summaries[community_id] = summary

    return community_summaries



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
    
def generate_summary_with_llm(model, tokenizer, device, prompt):
    """
    Generate a summary using the LLM.
    """
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

    #! print(f'RESPONSE FROM COMMUNITY SUMMARY: {response_text}')
    return response_text