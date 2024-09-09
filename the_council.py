import openai
from utils import load_config, logger
import markdown
import os
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
import time
from tqdm import tqdm

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_context(context, max_tokens=3000):
    tokens = count_tokens(context)
    if tokens <= max_tokens:
        return context
    
    lines = context.split('\n')
    while tokens > max_tokens and len(lines) > 1:
        lines.pop(0)
        context = '\n'.join(lines)
        tokens = count_tokens(context)
    
    return context

def query_openai_model(api_key, model, prompt):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI API call: {str(e)}")
        return None

def query_model(api_key, model, prompt):
    return query_openai_model(api_key, model, prompt)

def auto_review(api_key, model, result):
    review_prompt = f"Review the following response. Highlight strengths and suggest improvements:\n\n{result}"
    return query_model(api_key, model, review_prompt)

def stage_completion_check(api_key, model, context):
    check_prompt = f"Determine if the following stage is complete and ready to move on to the next stage. Provide a yes or no answer and justify your decision:\n\n{context}"
    return query_model(api_key, model, check_prompt)

def summarize_content(api_key, model, content):
    summary_prompt = f"Summarize the following content while retaining key information:\n\n{content}"
    return query_model(api_key, model, summary_prompt)

def hierarchical_summarization(api_key, model, document):
    sections = document.split("\n\n## ")
    summarized_sections = []

    for section in sections:
        if section.strip():
            summarized_section = summarize_content(api_key, model, section)
            summarized_sections.append(summarized_section)

    return "\n\n".join(summarized_sections)

def generate_embeddings(text):
    return embedding_model.encode(text, convert_to_tensor=True)

def find_similar_sections(embeddings, threshold=0.8):
    similar_sections = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j])
            if similarity > threshold:
                similar_sections.append((i, j))
    return similar_sections

def merge_similar_sections(sections, similar_sections):
    merged_sections = sections[:]
    for i, j in similar_sections:
        merged_sections[i] = f"{sections[i]}\n\n{sections[j]}"
        merged_sections[j] = ""
    return [section for section in merged_sections if section]

def generate_step_by_step_thinking(api_key, model, prompt):
    step_prompt = f"Break down your thinking process step-by-step to address the following task:\n\n{prompt}"
    return query_model(api_key, model, step_prompt)

def generate_multiple_responses(api_key, model, prompt, num_responses=3):
    responses = []
    for _ in range(num_responses):
        response = query_model(api_key, model, prompt)
        responses.append(response)
    return responses

def select_best_response(api_key, model, responses):
    selection_prompt = f"Review the following responses and select the best one. Explain your choice:\n\n"
    for i, response in enumerate(responses, 1):
        selection_prompt += f"Response {i}:\n{response}\n\n"
    return query_model(api_key, model, selection_prompt)

def reflect_on_response(api_key, model, response):
    reflection_prompt = f"Reflect on the following response. What are its strengths and weaknesses? What assumptions were made? What could be improved?\n\n{response}"
    return query_model(api_key, model, reflection_prompt)

def acknowledge_limitations(api_key, model, response):
    limitation_prompt = f"Identify potential limitations or biases in the following response. What aspects might be incorrect or need further verification?\n\n{response}"
    return query_model(api_key, model, limitation_prompt)

def chain_of_thought(api_key, model, prompt, steps=3):
    thoughts = []
    for i in range(steps):
        if i == 0:
            thought_prompt = f"Initial thought on the task: {prompt}"
        else:
            thought_prompt = f"Considering the previous thoughts: {thoughts}, provide the next step in reasoning about: {prompt}"
        thought = query_model(api_key, model, thought_prompt)
        thoughts.append(thought)
    return thoughts

def debate_topic(api_key, model, topic, rounds=3):
    debate = []
    for i in range(rounds):
        if i % 2 == 0:
            prompt = f"Argue for the following topic: {topic}. Consider previous arguments: {debate}"
        else:
            prompt = f"Argue against the following topic: {topic}. Consider previous arguments: {debate}"
        argument = query_model(api_key, model, prompt)
        debate.append(argument)
    return debate

def peer_review(config, model_result, reviewer_count=2):
    reviews = []
    for i in range(reviewer_count):
        reviewer_config = config['models'][i % len(config['models'])]
        review_prompt = f"As a peer reviewer, critically evaluate this response: {model_result['best_response']}"
        review = query_model(reviewer_config['api_key'], reviewer_config['model'], review_prompt)
        reviews.append(review)
    return reviews

def meta_analysis(api_key, model, results):
    meta_prompt = "Perform a meta-analysis of the following results from different models. Identify common themes, contradictions, and unique insights:\n\n"
    for model_name, result in results.items():
        meta_prompt += f"{model_name}:\n{result['best_response']}\n\n"
    return query_model(api_key, model, meta_prompt)

def iterative_improvement(api_key, model, response, iterations=3):
    improved_response = response
    for _ in range(iterations):
        improvement_prompt = f"Improve the following response by addressing any weaknesses, adding more detail, or considering alternative perspectives:\n\n{improved_response}"
        improved_response = query_model(api_key, model, improvement_prompt)
    return improved_response

def process_model(model_config, context, config):
    prompt = model_config['prompt'].format(truncate_context(context))
    
    structured_response = {
        'overview': query_model(model_config['api_key'], model_config['model'], f"{prompt}\n\nProvide a brief overview."),
        'detailed_design': query_model(model_config['api_key'], model_config['model'], f"{prompt}\n\nProvide a detailed design."),
        'implementation_considerations': query_model(model_config['api_key'], model_config['model'], f"{prompt}\n\nDiscuss implementation considerations."),
        'potential_challenges': query_model(model_config['api_key'], model_config['model'], f"{prompt}\n\nIdentify potential challenges."),
        'integration_points': query_model(model_config['api_key'], model_config['model'], f"{prompt}\n\nDescribe integration points with other game elements.")
    }
    
    return structured_response

def final_review(api_key, model, document):
    review_prompt = "Review the following game design document. Identify any inconsistencies, missing information, or areas that need further development. Provide suggestions for improvement:\n\n" + document
    return query_model(api_key, model, review_prompt)

def run_council(idea, config):
    results = {}
    context = idea
    document = f"""# Game Design Document

## 1. Executive Summary
{idea}

## 2. Game Overview
### 2.1 Concept
### 2.2 Genre
### 2.3 Target Audience
### 2.4 Game Flow Summary
### 2.5 Look and Feel

## 3. Gameplay and Mechanics
### 3.1 Gameplay
#### 3.1.1 Game Progression
#### 3.1.2 Mission/challenge Structure
#### 3.1.3 Puzzle Structure
### 3.2 Mechanics
### 3.3 Screen Flow

## 4. Story, Setting and Character
### 4.1 Story and Narrative
### 4.2 Game World
### 4.3 Characters

## 5. Levels
### 5.1 Level Design
### 5.2 Training Level

## 6. Interface
### 6.1 Visual System
### 6.2 Control System
### 6.3 Audio, music, sound effects

## 7. Artificial Intelligence
### 7.1 Opposition AI
### 7.2 Friend AI
### 7.3 Support AI

## 8. Technical
### 8.1 Target Hardware
### 8.2 Development hardware and software
### 8.3 Network requirements

## 9. Game Art
### 9.1 Concept Art
### 9.2 Style Guides

"""
    
    start_time = time.time()
    total_models = len(config['models'])
    
    print(f"Starting the council process with {total_models} models...")
    
    for i, model_config in tqdm(enumerate(config['models'], 1), total=total_models, desc="Processing models"):
        print(f"\nProcessing {model_config['name']} ({i}/{total_models})...")
        
        model_start_time = time.time()
        model_result = process_model(model_config, context, config)
        model_end_time = time.time()
        
        model_duration = model_end_time - model_start_time
        print(f"{model_config['name']} completed in {model_duration:.2f} seconds")
        
        for section in model_config['sections']:
            section_content = f"## {section}\n\n"
            section_content += f"### Overview\n{model_result['overview']}\n\n"
            section_content += f"### Detailed Design\n{model_result['detailed_design']}\n\n"
            section_content += f"### Implementation Considerations\n{model_result['implementation_considerations']}\n\n"
            section_content += f"### Potential Challenges\n{model_result['potential_challenges']}\n\n"
            section_content += f"### Integration Points\n{model_result['integration_points']}\n\n"
            
            document = document.replace(f"## {section}", section_content)
        
        print("Conducting peer review...")
        peer_reviews = peer_review(config, model_result)
        document += f"### Peer Reviews\n{peer_reviews}\n\n"
        
        results[model_config['name']] = model_result
        context += f"\n\n{model_config['name']}'s Contribution:\n{model_result['improved_response']}"
        
        print("Summarizing context...")
        context = summarize_content(model_config['api_key'], "gpt-4o-mini", context)
        
        print(f"Progress: {i}/{total_models} models processed")
    
    print("\nPerforming meta-analysis...")
    meta_result = meta_analysis(config['models'][-1]['api_key'], config['models'][-1]['model'], results)
    document += f"## Meta Analysis\n{meta_result}\n\n"
    
    print("\nPerforming final review...")
    final_review_result = final_review(config['models'][-1]['api_key'], config['models'][-1]['model'], document)
    document += f"\n\n## Final Review and Recommendations\n{final_review_result}\n"
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nCouncil process completed in {total_duration:.2f} seconds")
    
    return results, document

def produce_product(results):
    product = ""
    for model, result in results.items():
        product += f"## {model}\n"
        product += f"### Best Response\n{result['best_response']}\n\n"
        product += f"### Reflection\n{result['reflection']}\n\n"
        product += f"### Limitations\n{result['limitations']}\n\n"
    return product

def save_document(document, filename="game_design_document.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(document)
    
    html = markdown.markdown(document)
    with open(filename.replace(".md", ".html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    try:
        config = load_config('config.json')
        idea = input("Enter your idea for the incremental game: ")
        
        print("\nStarting the council process...")
        start_time = time.time()
        
        results, document = run_council(idea, config)
        
        print("\nGenerating final product...")
        final_product = produce_product(results)
        document += f"# Final Product\n\n{final_product}"
        
        print("Saving document...")
        save_document(document)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\nThe Council's Product:\n" + "="*40)
        print(final_product)
        print("="*40)
        print(f"\nFull document saved as 'game_design_document.md' and 'game_design_document.html'")
        print(f"\nTotal process completed in {total_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()