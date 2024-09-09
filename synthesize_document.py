import openai
from utils import load_config, logger
import markdown
import os
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def read_document(filename="game_design_document.md"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Document file not found: {filename}")
        raise

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_into_chunks(text, max_tokens=4000, model="gpt-4o"):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, model)
        if current_tokens + line_tokens > max_tokens:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def query_openai_model(api_key, model, prompt):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000  # Increased token limit for synthesis
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error querying OpenAI model: {str(e)}"

def query_model(api_key, model, prompt):
    return query_openai_model(api_key, model, prompt)

def synthesize_chunk(config, chunk):
    overmind_config = next((m for m in config['models'] if m['name'] == "Overmind"), None)
    if not overmind_config:
        logger.error("Overmind configuration not found in config.json")
        return None

    prompt = f"As the Overmind, synthesize the following chunk of the game design document into a cohesive and logical format. Ensure all parts are logically connected and highlight any inconsistencies or areas that need further development:\n\n{chunk}"
    result = query_model(overmind_config['api_key'], overmind_config['model'], prompt)
    return result

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

def synthesize_document(config, document_content):
    chunks = split_into_chunks(document_content)
    synthesized_chunks = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            futures = [executor.submit(synthesize_chunk, config, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    synthesized_chunks.append(result)
                else:
                    logger.warning("A chunk failed to synthesize")
        except Exception as e:
            logger.error(f"Error during document synthesis: {str(e)}")

    # Generate embeddings for synthesized chunks
    embeddings = [generate_embeddings(chunk) for chunk in synthesized_chunks]
    similar_sections = find_similar_sections(embeddings)

    # Merge similar sections
    merged_chunks = merge_similar_sections(synthesized_chunks, similar_sections)

    # Combine all synthesized chunks into a final document
    combined_content = "\n\n".join(merged_chunks)
    final_synthesis = synthesize_chunk(config, combined_content)
    return final_synthesis

def save_synthesized_document(content, filename="synthesized_game_design_document.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    html = markdown.markdown(content)
    with open(filename.replace(".md", ".html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    try:
        config = load_config('config.json')
        document_content = read_document()
        synthesized_content = synthesize_document(config, document_content)
        
        if synthesized_content:
            save_synthesized_document(synthesized_content)
            print("\nSynthesized Game Design Document saved as 'synthesized_game_design_document.md' and 'synthesized_game_design_document.html'")
        else:
            print("Failed to synthesize the document.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()