import streamlit as st
import time
from utils import load_config, logger
import json
import openai

st.set_page_config(page_title="The Council - Game Design Document Generator", layout="wide")

# Define specialist roles and their corresponding sections
SPECIALISTS = {
    "Game Designer": ["Game Overview", "Gameplay and Mechanics"],
    "Story Writer": ["Story, Setting and Character"],
    "Level Designer": ["Levels"],
    "UI/UX Designer": ["Interface"],
    "AI Designer": ["Artificial Intelligence"],
    "Technical Designer": ["Technical"],
    "Art Director": ["Game Art"]
}

@st.cache_data
def query_model(api_key, model, prompt, max_tokens=4000):
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

def generate_section_content(api_key, model, specialist, section, context):
    prompt = f"As a {specialist}, create detailed content for the '{section}' section of a game design document. Consider the following context:\n\n{context}\n\nProvide a comprehensive and detailed description for this section."
    return query_model(api_key, model, prompt)

def expand_content(api_key, model, content):
    prompt = f"Expand on the following content, adding more details, examples, and considerations:\n\n{content}"
    return query_model(api_key, model, prompt)

def summarize_content(api_key, model, content):
    prompt = f"Summarize the key points of the following content:\n\n{content}"
    return query_model(api_key, model, prompt, max_tokens=500)

def run_streamlit_council(idea, config):
    document = f"# Game Design Document\n\n## Executive Summary\n{idea}\n\n"
    context = idea
    
    total_sections = sum(len(sections) for sections in SPECIALISTS.values())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create tabs for each specialist
    specialist_tabs = st.tabs(SPECIALISTS.keys())

    section_counter = 0
    for specialist, sections in SPECIALISTS.items():
        with specialist_tabs[list(SPECIALISTS.keys()).index(specialist)]:
            st.subheader(f"{specialist}'s Contributions")
            for section in sections:
                section_counter += 1
                status_text.text(f"Processing {section} with {specialist} ({section_counter}/{total_sections})...")
                
                try:
                    # Generate initial content
                    with st.spinner(f"Generating content for {section}..."):
                        content = generate_section_content(config['api_key'], config['model'], specialist, section, context)
                    
                    # Expand content
                    with st.spinner(f"Expanding content for {section}..."):
                        expanded_content = expand_content(config['api_key'], config['model'], content)
                    
                    # Add to document
                    document += f"## {section}\n\n{expanded_content}\n\n"
                    
                    # Summarize for context
                    with st.spinner(f"Summarizing {section}..."):
                        summary = summarize_content(config['api_key'], config['model'], expanded_content)
                    context += f"\n{section} summary: {summary}"
                    
                    # Display content in an expander
                    with st.expander(f"{section} - Click to expand"):
                        st.write(expanded_content)
                        st.write("---")
                        st.write("Summary:")
                        st.write(summary)
                    
                except Exception as e:
                    st.error(f"Error processing {section} with {specialist}: {str(e)}")
                
                progress_bar.progress(section_counter / total_sections)
    
    status_text.text("Process completed!")
    progress_bar.progress(1.0)
    
    return document

def main():
    st.title("The Council - Game Design Document Generator")

    config = load_config('config.json')
    
    # Sidebar for input and controls
    with st.sidebar:
        st.header("Game Idea Input")
        idea = st.text_area("Enter your idea for the game:", "A 2D incremental game about building and managing a space colony")
        generate_button = st.button("Generate Game Design Document")

    if generate_button:
        start_time = time.time()
        
        # Main content area
        main_content = st.empty()
        with main_content.container():
            st.header("Generating Game Design Document")
            document = run_streamlit_council(idea, config)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Display results
        main_content.empty()
        with main_content.container():
            st.header("Game Design Document Generated")
            st.success(f"Total process completed in {total_duration:.2f} seconds")
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download as Markdown",
                    data=document,
                    file_name="game_design_document.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    label="Download as Text",
                    data=document,
                    file_name="game_design_document.txt",
                    mime="text/plain"
                )
            
            # Display document preview
            with st.expander("Document Preview", expanded=True):
                st.markdown(document)

if __name__ == "__main__":
    main()