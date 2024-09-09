import streamlit as st
import time
from utils import load_config, logger
import json
import openai

st.set_page_config(page_title="The Council - Game Design Document Generator", layout="wide")

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

def process_model(model_config, context):
    prompt = model_config['prompt'].format(context[:150])  # Truncate context for faster processing
    overview = query_model(model_config['api_key'], model_config['model'], f"{prompt}\nProvide a comprehensive overview.")
    return {
        'overview': overview,
    }

def run_streamlit_council(idea, config):
    results = {}
    document = f"# Game Design Document\n\n## Initial Idea\n{idea}\n\n"
    
    total_models = len(config['models'])
    
    st.write(f"Starting the council process with {total_models} models...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_config in enumerate(config['models'], 1):
        status_text.text(f"Processing {model_config['name']} ({i}/{total_models})...")
        
        try:
            model_result = process_model(model_config, idea)
            
            st.write(f"## {model_config['name']}'s Contribution")
            st.write("### Overview")
            st.write(model_result['overview'])
            
            results[model_config['name']] = model_result
        except Exception as e:
            st.error(f"Error processing {model_config['name']}: {str(e)}")
        
        progress_bar.progress((i / total_models))
    
    status_text.text("Process completed!")
    progress_bar.progress(1.0)
    
    return results, document

def main():
    st.title("The Council - Game Design Document Generator")

    config = load_config('test_config.json')
    idea = st.text_input("Enter your idea for the incremental game:", "A 2D incremental game about building and managing a space colony")

    if st.button("Generate Game Design Document"):
        start_time = time.time()
        
        results, document = run_streamlit_council(idea, config)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        st.write(f"Total process completed in {total_duration:.2f} seconds")
        
        st.download_button(
            label="Download Game Design Document",
            data=document,
            file_name="game_design_document.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()