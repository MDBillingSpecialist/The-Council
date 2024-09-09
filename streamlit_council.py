import streamlit as st
import time
from utils import load_config, logger
import json
import openai
from test_council import query_model, chain_of_thought, debate_topic, peer_review, meta_analysis, iterative_improvement, process_model

st.set_page_config(page_title="The Council - Game Design Document Generator", layout="wide")

def run_streamlit_council(idea, config):
    results = {}
    document = f"# Enhanced Test Game Design Document\n\n## Initial Idea\n{idea}\n\n"
    
    total_models = len(config['models'])
    
    st.write(f"Starting the enhanced test council process with {total_models} models...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_config in enumerate(config['models'], 1):
        status_text.text(f"Processing {model_config['name']} ({i}/{total_models})...")
        
        try:
            model_result = process_model(model_config, idea)
            
            st.write(f"## {model_config['name']}'s Contribution")
            st.write("### Initial Overview")
            st.write(model_result['overview'])
            st.write("### Improved Overview")
            st.write(model_result['improved_overview'])
            st.write("### Chain of Thought")
            for thought in model_result['chain_of_thought']:
                st.write(thought)
            st.write("### Debate")
            for argument in model_result['debate']:
                st.write(argument)
            
            peer_reviews = peer_review(config, model_result)
            st.write("### Peer Reviews")
            for review in peer_reviews:
                st.write(review)
            
            results[model_config['name']] = model_result
        except Exception as e:
            st.error(f"Error processing {model_config['name']}: {str(e)}")
        
        progress_bar.progress((i / total_models))
    
    status_text.text("Performing meta-analysis...")
    meta_result = meta_analysis(config['models'][-1]['api_key'], config['models'][-1]['model'], results)
    st.write("## Meta Analysis")
    st.write(meta_result)
    
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