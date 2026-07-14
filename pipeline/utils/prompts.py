DEFAULT_TASK = """
Identify the relations and structure of the community of interest, specifically within the {domain} domain.
"""

ENTITY_TYPE_GENERATION_PROMPT = """
  The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
  The user's task is to {task}.
  As part of the analysis, you want to identify the entity types present in the following text.
  The entity types must be relevant to the user task.
  Avoid general entity types such as "other" or "unknown".
  This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
  Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
  Return the entity types in JSON format with "entities" as the key and the entity types as an array of strings.
  =====================================================================
  EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

  EXAMPLE 1
  Task: Determine the connections and organizational hierarchy within the specified community.
  Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
  JSON RESPONSE:
  {{"entity_types": [organization, person] }}
  END OF EXAMPLE 1

  EXAMPLE 2
  Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought, and trace the historical or ideological influences they have on each other.
  Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge. Key concepts within this school include the emphasis on the deductive method of reasoning.
  JSON RESPONSE:
  {{"entity_types": [concept, person, school of thought] }}
  END OF EXAMPLE 2

  EXAMPLE 3
  Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
  Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector. They are investing heavily in research and development and are exploring new technologies to gain a competitive edge.
  JSON RESPONSE:
  {{"entity_types": [organization, technology, sectors, investment strategies] }}
  END OF EXAMPLE 3
  ======================================================================

  ======================================================================
  REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
  Task: {task}
  Text: {input_text}
  JSON response:
  {{"entity_types": [<entity_types>] }}
"""

ENTITY_RELATIONSHIPS_GENERATION_PROMPT = """
  -Goal-
  Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

  -Steps-
  1. Identify all entities. For each identified entity, extract the following information:
  - entity_name: Name of the entity, capitalized
  - entity_type: One of the following types: [{entity_types}]
  - entity_description: Comprehensive description of the entity's attributes and activities

  Format each entity output as a JSON entry with the following format:

  {{"name": "<entity name>", "type": "<type>", "description": "<entity description>"}}

  2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
  For each pair of related entities, extract the following information:
  - source_entity: name of the source entity, as identified in step 1
  - target_entity: name of the target entity, as identified in step 1
  - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
  - relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity

  Format each relationship as a JSON entry with the following format:

  {{"source": "<source_entity>", "target": "<target_entity>", "relationship": "<relationship_description>", "relationship_strength": "<relationship_strength>"}}

  3. Return output in {language} as a single list of all JSON entities and relationships identified in steps 1 and 2.

  4. If you have to translate into {language}, just translate the descriptions, nothing else!

  ######################
  -Examples-
  ######################
  Example 1:
  Text:
  The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
  ######################
  Output:
  [
    {{"name": "CENTRAL INSTITUTION", "type": "ORGANIZATION", "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday"}},
    {{"name": "MARTIN SMITH", "type": "PERSON", "description": "Martin Smith is the chair of the Central Institution"}},
    {{"name": "MARKET STRATEGY COMMITTEE", "type": "ORGANIZATION", "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply"}},
    {{"source": "MARTIN SMITH", "target": "CENTRAL INSTITUTION", "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference", "relationship_strength": 9}}
  ]

  ######################
  Example 2:
  Text:
  TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

  TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
  ######################
  Output:
  [
    {{"name": "TECHGLOBAL", "type": "ORGANIZATION", "description": "TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones"}},
    {{"name": "VISION HOLDINGS", "type": "ORGANIZATION", "description": "Vision Holdings is a firm that previously owned TechGlobal"}},
    {{"source": "TECHGLOBAL", "target": "VISION HOLDINGS", "relationship": "Vision Holdings formerly owned TechGlobal from 2014 until present", "relationship_strength": 5}}
  ]

  ######################
  Example 3:
  Text:
  Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

  The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

  The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

  They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

  The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
  ######################
  Output:
  [
    {{"name": "FIRUZABAD", "type": "GEO", "description": "Firuzabad held Aurelians as hostages"}},
    {{"name": "AURELIA", "type": "GEO", "description": "Country seeking to release hostages"}},
    {{"name": "QUINTARA", "type": "GEO", "description": "Country that negotiated a swap of money in exchange for hostages"}},
    {{"name": "TIRUZIA", "type": "GEO", "description": "Capital of Firuzabad where the Aurelians were being held"}},
    {{"name": "KROHAARA", "type": "GEO", "description": "Capital city in Quintara"}},
    {{"name": "CASHION", "type": "GEO", "description": "Capital city in Aurelia"}},
    {{"name": "SAMUEL NAMARA", "type": "PERSON", "description": "Aurelian who spent time in Tiruzia's Alhamia Prison"}},
    {{"name": "ALHAMIA PRISON", "type": "GEO", "description": "Prison in Tiruzia"}},
    {{"name": "DURKE BATAGLANI", "type": "PERSON", "description": "Aurelian journalist who was held hostage"}},
    {{"name": "MEGGIE TAZBAH", "type": "PERSON", "description": "Bratinas national and environmentalist who was held hostage"}},
    {{"source": "FIRUZABAD", "target": "AURELIA", "relationship": "Firuzabad negotiated a hostage exchange with Aurelia", "relationship_strength": 2}},
    {{"source": "QUINTARA", "target": "AURELIA", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
    {{"source": "QUINTARA", "target": "FIRUZABAD", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "ALHAMIA PRISON", "relationship": "Samuel Namara was a prisoner at Alhamia prison", "relationship_strength": 8}},
    {{"source": "SAMUEL NAMARA", "target": "MEGGIE TAZBAH", "relationship": "Samuel Namara and Meggie Tazbah were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "DURKE BATAGLANI", "relationship": "Samuel Namara and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "MEGGIE TAZBAH", "target": "DURKE BATAGLANI", "relationship": "Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "FIRUZABAD", "relationship": "Samuel Namara was a hostage in Firuzabad", "relationship_strength": 2}},
    {{"source": "MEGGIE TAZBAH", "target": "FIRUZABAD", "relationship": "Meggie Tazbah was a hostage in Firuzabad", "relationship_strength": 2}},
    {{"source": "DURKE BATAGLANI", "target": "FIRUZABAD", "relationship": "Durke Bataglani was a hostage in Firuzabad", "relationship_strength": 2}}
]

  -Real Data-
  ######################
  entity_types: {entity_types}
  text: {input_text}
  ######################
  output:
"""

GENERATE_PERSONA_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a specific type of task and sample text, help the user by generating a 3 to 4 sentence description of an expert who could help solve the problem.
Use a format similar to the following:
You are an expert {{role}}. You are skilled at {{relevant skills}}. You are adept at helping people with {{specific task}}.

task: {sample_task}
persona description:"""

ENTITY_SUMMARIZATION_PROMPT = """
You are {persona}.
Using your expertise, you're tasked with generating a comprehensive summary of the data provided below.
The data includes one or more entities and a set of descriptions, all of which are related to the same entity or group of entities.

Here is what you need to do:
1. **Understand and Integrate Information:**
   - Combine all the descriptions provided into a cohesive and unified summary.
   - Ensure all critical details are included, even if they are implied rather than explicitly stated.

2. **Resolve Contradictions:**
   - If any descriptions conflict with one another, resolve the contradictions based on the most plausible and coherent interpretation.
   - Do not leave unresolved or ambiguous statements in the summary.

3. **Provide Explicit Context:**
   - Write the summary in **third person**, explicitly naming the entities wherever appropriate.
   - Include sufficient context to make the summary independently meaningful, even if someone reading it doesn't have access to the original data.

4. **Enrich the Summary:**
   - Use insights from nearby text (if provided) to enrich the summary with relevant details.
   - Where possible, connect entities to related concepts, events, or broader contexts to provide a more thorough understanding.
   
5. **Maintain Clarity and Brevity:**
   - Keep the summary concise but ensure it covers all critical points.
   - Aim for a balance between richness of detail and readability.

######################
-Examples-
######################

Example 1:
Entities: Great Barrier Reef
Descriptions:
- 'The Great Barrier Reef is the world’s largest coral reef system.'
- 'It is located off the coast of Queensland, Australia, and is a UNESCO World Heritage site.'
- 'The reef is home to diverse marine species and is a popular tourist attraction.'
######################
Output:
The Great Barrier Reef, located off the coast of Queensland, Australia, is the world’s largest coral reef system and a UNESCO World Heritage site. Renowned for its biodiversity, the reef is home to a vast array of marine species and serves as a significant tourist destination.

######################
Example 2:
Entities: Tesla, Inc.
Descriptions:
- 'Tesla, Inc. is an American electric vehicle manufacturer founded in 2003.'
- 'Elon Musk, the company’s CEO, is known for his role in revolutionizing the automotive industry.'
- 'Tesla specializes in electric vehicles, energy storage solutions, and solar energy products.'
######################
Output:
Tesla, Inc., founded in 2003, is an American company specializing in electric vehicles, energy storage solutions, and solar energy products. Under the leadership of CEO Elon Musk, the company has become a global leader in innovative automotive and energy technologies.

######################
Example 3:
Entities: Mount Everest
Descriptions:
- 'Mount Everest is the tallest mountain in the world, standing at 8,848 meters.'
- 'It is located in the Himalayan mountain range on the border between Nepal and Tibet.'
- 'The mountain attracts climbers from all over the world, despite its challenging conditions and high risk.'
######################
Output:
Mount Everest, the tallest mountain in the world at 8,848 meters, is situated in the Himalayan range along the border between Nepal and Tibet. Renowned for its breathtaking beauty, it is a sought-after destination for climbers worldwide, though it poses extreme challenges and significant risks.

######################
-Real Data-
Entities: {entity_name}
Descriptions:
{descriptions}
######################
Output: 
"""

COMMUNITY_SUMMARY_PROMPT = """
  -Goal-
  Given a list of nodes representing entities in a community and their detailed summaries, generate a comprehensive community summary that captures all relationships, key details, and the significance of the community as a whole.

  -Steps-
  1. Review the provided list of nodes and their summaries.
    - Each node has the following attributes:
    - title: The name of the entity in the node.
    - summary: A detailed explanation of the entity and its significance, including its relationships to other nodes within the community.
       
  2. Synthesize the provided information to produce a cohesive summary of the entire community.
    - Include all relevant details about the entities.
    - Highlight relationships and interactions between nodes.
    - Convey the overall importance of the community based on the provided details.

  3. Ensure the summary is well-organized and concise, capturing the essence of the community in a single paragraph.

  ######################
  -Examples-
  ######################
  Example 1:
  Nodes: ['AUSTRALIAN CAPITAL TERRITORY', 'COMMONWEALTH PARK', 'RECONCILIATION PLACE']
  Node Summaries: [
  {{'Entity': 'AUSTRALIAN CAPITAL TERRITORY', 'Type': 'GEO', 'Summary': "The event took place in the Australian Capital Territory, a region in the heart of Australia, in Canberra's central areas. It is within this region that Agnes Shea offered a message stick to Chinese officials on behalf of the Aboriginal people, symbolizing a connection between the territory and the broader Australian community. This event highlights the importance of the Australian Capital Territory as a hub for cultural exchange and understanding."}},
  {{'Entity': 'COMMONWEALTH PARK', 'Type': 'GEO', 'Summary': 'The Commonwealth Park, a popular recreational area in Canberra, served as the venue for a significant event. The event took place in the park, utilizing its facilities and amenities to host the gathering.'}},
  {{'Entity': 'RECONCILIATION PLACE', 'Type': 'GEO', 'Summary': 'The RECONCILIATION PLACE, a venue in Canberra, served as the site for the event. Located in this place, the gathering took place.'}},
  ]
  ######################
  Output:
  The Australian Capital Territory is a culturally significant region at the heart of Australia, hosting major events that symbolize unity and cultural exchange. Within this territory, Commonwealth Park and Reconciliation Place served as central venues for a key gathering. Commonwealth Park, known for its recreational appeal, provided the necessary amenities for the event, while Reconciliation Place symbolized the event’s deeper connection to reconciliation and cultural significance. A pivotal moment of the event was Agnes Shea’s presentation of a message stick to Chinese officials, reinforcing the territory’s role in fostering international understanding and cultural diplomacy.

  ######################
  Example 2:
  Nodes: ['TECHGLOBAL', 'VISION HOLDINGS']
  Node Summaries: [
  {{'Entity': 'TECHGLOBAL', 'Type': 'ORGANIZATION', 'Summary': 'TechGlobal is a technology company that powers 85% of premium smartphones and recently returned to public markets on the Global Exchange after being privately held by Vision Holdings since 2014.'}},
  {{'Entity': 'VISION HOLDINGS', 'Type': 'ORGANIZATION', 'Summary': 'Vision Holdings is an investment firm that owned TechGlobal from 2014 until its recent public relisting. The firm specializes in fostering growth in the technology sector.'}},
  ]
  ######################
  Output:
  TechGlobal, a leading technology company powering 85% of premium smartphones, recently returned to public markets, marking a significant milestone in its history. Formerly held by Vision Holdings, an investment firm known for its focus on technological innovation, TechGlobal’s relisting highlights its growth and enduring impact on the technology industry. The relationship between Vision Holdings and TechGlobal underscores the firm's role in shaping TechGlobal’s success during its private ownership.

  ######################
  Real Data:
  ######################
  
  Nodes: {nodes}
  Node Summaries: {node_summaries}
  
  ######################
  Output:
"""
