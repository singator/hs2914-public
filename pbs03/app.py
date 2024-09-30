import streamlit as st
from transformers import BertTokenizer
from nltk.util import pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import gutenberg
import nltk

# Ensure NLTK and transformers packages are properly loaded
# Cache loading of BERT tokenizer
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-cased")

# Cache loading of NLTK data and models for bigram text generation
@st.cache_resource
def get_all_bigram_models():
    nltk.download('gutenberg')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    text_mapping = {
        'shakespeare-caesar': 'shakespeare-caesar.txt',
        'carroll-alice': 'carroll-alice.txt',
        'bible-kjv': 'bible-kjv.txt'
    }
    mdl_mapping = {}
    for kk, vv in text_mapping.items():
        train_data, vocab = padded_everygram_pipeline(2, gutenberg.sents(vv))
        model = MLE(2)
        model.fit(train_data, vocab)
        mdl_mapping[kk] = model
    return mdl_mapping

# Function to tokenize text using BERT tokenizer
def tokenize_text(text, tokenizer):
    return tokenizer.tokenize(text)

# Function to format tokens in lines with a specified number of tokens per line
def format_tokens(tokens, tokens_per_line=15):
    lines = [' '.join(tokens[i:i + tokens_per_line]) for i in range(0, len(tokens), tokens_per_line)]
    return '\n'.join(lines)

# Function to generate multiple strings from the bigram model
def generate_multiple_strings(model, seed_text, max_length, num_strings=10):
    generated_strings = []
    for _ in range(num_strings):
        generated_words = model.generate(max_length, text_seed=seed_text)
        generated_strings.append(' '.join(generated_words))
    return generated_strings

# Load the tokenizer and bigram models
tokenizer = load_tokenizer()
bigram_models = get_all_bigram_models()

# Tabs for the two functionalities
tab1, tab2 = st.tabs(["BERT Tokenizer", "Bigram Text Generator"])

# BERT Tokenizer tab
with tab1:
    st.title("BERT Tokenizer")
    
    # Input from the user
    sequence = st.text_area("Enter a sentence or string of text:", 
                            value="But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.")
    
    # Display the tokenized version when the button is clicked
    if st.button("Tokenize", key="tokenize_button"):
        tokens = tokenize_text(sequence, tokenizer)
        pretty_tokens = format_tokens(tokens, tokens_per_line=15)
        st.write("**Tokenized Text:**")
        st.text(pretty_tokens)

# Bigram Text Generator tab
with tab2:
    st.title("Bigram Text Generator")

    # Step 1: Text selection
    text_choice = st.selectbox("Choose a text:", 
                               ['shakespeare-caesar', 'carroll-alice', 'bible-kjv'])

    # Step 2: User inputs for prompt and string length
    start_prompt = st.text_input("Enter the starting prompt (comma-separated):", value="I am")
    max_length = st.number_input("Enter the maximum length of the generated string:", 
                                 min_value=5, max_value=100, value=20)

    # Convert the input prompt to a list (seed for generation)
    seed_text = wordpunct_tokenize(start_prompt)

    # Step 3: Generate and display the text
    if st.button("Generate Text", key="generate_button"):
        # Access the cached model based on the selected text
        bigram_model = bigram_models.get(text_choice, None)
        
        if bigram_model:
            # Generate 10 strings
            generated_strings = generate_multiple_strings(bigram_model, seed_text, max_length)
            
            # Display the prompt and generated strings
            st.write(f"**Prompt:** {' '.join(seed_text)}")
            st.write("**Generated Strings:**")
            
            # Use Streamlit's markdown to display the strings as a bullet list
            for gen_str in generated_strings:
                st.markdown(f"- {gen_str}")
        else:
            st.error(f"No model found for {text_choice}")
