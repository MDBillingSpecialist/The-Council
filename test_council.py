import time
from utils import load_config, logger
from tqdm import tqdm
import json
import openai

def query_model(api_key, model, prompt, max_tokens=150):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error querying OpenAI model: {str(e)}")
        return f"Error: {str(e)}"

def chain_of_thought(api_key, model, prompt, steps=3):
    thoughts = []
    for i in range(steps):
        thought_prompt = f"Step {i+1} of {steps}: Provide a detailed thought about: {prompt}"
        thought = query_model(api_key, model, thought_prompt)
        thoughts.append(thought)
    return thoughts

def debate_topic(api_key, model, topic, rounds=2):
    debate = []
    for i in range(rounds):
        if i % 2 == 0:
            prompt = f"Argue for the following aspect of the game: {topic}. Consider previous arguments: {debate}"
        else:
            prompt = f"Argue against the following aspect of the game: {topic}. Consider previous arguments: {debate}"
        argument = query_model(api_key, model, prompt)
        debate.append(argument)
    return debate

def peer_review(config, model_result, reviewer_count=2):
    reviews = []
    for i in range(reviewer_count):
        reviewer_config = config['models'][i % len(config['models'])]
        review_prompt = f"Critically review this game design element. Highlight strengths and suggest improvements:\n\n{model_result['overview']}"
        review = query_model(reviewer_config['api_key'], reviewer_config['model'], review_prompt)
        reviews.append(review)
    return reviews

def meta_analysis(api_key, model, results):
    meta_prompt = "Perform a comprehensive analysis of these game design elements. Identify common themes, contradictions, and unique insights:\n"
    for model_name, result in results.items():
        meta_prompt += f"{model_name}: {result['overview']}\n"
    return query_model(api_key, model, meta_prompt, max_tokens=300)

def iterative_improvement(api_key, model, response, iterations=2):
    improved_response = response
    for i in range(iterations):
        improvement_prompt = f"Iteration {i+1}: Improve the following game design element by addressing weaknesses, adding detail, or considering alternative perspectives:\n\n{improved_response}"
        improved_response = query_model(api_key, model, improvement_prompt)
    return improved_response

def process_model(model_config, context):
    prompt = model_config['prompt'].format(context[:150])  # Truncate context for faster processing
    overview = query_model(model_config['api_key'], model_config['model'], f"{prompt}\nProvide a comprehensive overview.")
    chain_of_thought_result = chain_of_thought(model_config['api_key'], model_config['model'], prompt)
    debate_result = debate_topic(model_config['api_key'], model_config['model'], overview)
    improved_overview = iterative_improvement(model_config['api_key'], model_config['model'], overview)
    return {
        'overview': overview,
        'improved_overview': improved_overview,
        'chain_of_thought': chain_of_thought_result,
        'debate': debate_result
    }

def run_test_council(idea, config):
    results = {}
    document = f"# Enhanced Test Game Design Document\n\n## Initial Idea\n{idea}\n\n"
    
    start_time = time.time()
    total_models = len(config['models'])
    
    print(f"Starting the enhanced test council process with {total_models} models...")
    
    for i, model_config in tqdm(enumerate(config['models'], 1), total=total_models, desc="Processing models"):
        print(f"\nProcessing {model_config['name']} ({i}/{total_models})...")
        
        try:
            model_start_time = time.time()
            model_result = process_model(model_config, idea)
            model_end_time = time.time()
            
            model_duration = model_end_time - model_start_time
            print(f"{model_config['name']} completed in {model_duration:.2f} seconds")
            
            document += f"## {model_config['name']}'s Contribution\n"
            document += f"### Initial Overview\n{model_result['overview']}\n\n"
            document += f"### Improved Overview\n{model_result['improved_overview']}\n\n"
            document += f"### Chain of Thought\n" + "\n".join(model_result['chain_of_thought']) + "\n\n"
            document += f"### Debate\n" + "\n".join(model_result['debate']) + "\n\n"
            
            peer_reviews = peer_review(config, model_result)
            document += f"### Peer Reviews\n" + "\n".join(peer_reviews) + "\n\n"
            
            results[model_config['name']] = model_result
        except Exception as e:
            logger.error(f"Error processing {model_config['name']}: {str(e)}")
    
    meta_result = meta_analysis(config['models'][-1]['api_key'], config['models'][-1]['model'], results)
    document += f"## Meta Analysis\n{meta_result}\n\n"
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nEnhanced test council process completed in {total_duration:.2f} seconds")
    
    return results, document

def compare_runs(run1, run2):
    comparison = "Detailed Comparison of Two Runs:\n\n"
    for model in run1.keys():
        if model in run2:
            comparison += f"## {model}\n"
            comparison += f"Run 1 initial overview: {run1[model]['overview'][:100]}...\n"
            comparison += f"Run 1 improved overview: {run1[model]['improved_overview'][:100]}...\n"
            comparison += f"Run 2 initial overview: {run2[model]['overview'][:100]}...\n"
            comparison += f"Run 2 improved overview: {run2[model]['improved_overview'][:100]}...\n\n"
            comparison += "Chain of Thought Comparison:\n"
            for i, (thought1, thought2) in enumerate(zip(run1[model]['chain_of_thought'], run2[model]['chain_of_thought'])):
                comparison += f"Step {i+1}:\nRun 1: {thought1[:50]}...\nRun 2: {thought2[:50]}...\n"
            comparison += "\n"
    return comparison

def main():
    try:
        config = load_config('test_config.json')
        idea = "A 2D incremental game about building and managing a space colony"
        
        print("\nStarting the enhanced test council process...")
        start_time = time.time()
        
        # Run the council twice to compare outputs
        results1, document1 = run_test_council(idea, config)
        results2, document2 = run_test_council(idea, config)
        
        print("\nEnhanced Test Document 1 Preview:")
        print(document1[:1000] + "...\n")
        
        print("\nEnhanced Test Document 2 Preview:")
        print(document2[:1000] + "...\n")
        
        comparison = compare_runs(results1, results2)
        print("\nDetailed Comparison of Two Runs:")
        print(comparison)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"Total enhanced test process completed in {total_duration:.2f} seconds")
        
        # Save results for further analysis
        with open('enhanced_test_results.json', 'w') as f:
            json.dump({'run1': results1, 'run2': results2}, f, indent=2)
        
        # Save full documents
        with open('enhanced_test_document1.md', 'w') as f:
            f.write(document1)
        with open('enhanced_test_document2.md', 'w') as f:
            f.write(document2)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()