import streamlit as st
import numpy as np
import pandas as pd

import pickle
import requests
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.parsing.preprocessing import *
import gensim.downloader as api
import nltk
from nltk.stem import WordNetLemmatizer

CUSTOM_FILTER = [lambda x: x.lower(), strip_punctuation, 
                 strip_multiple_whitespaces, strip_numeric, 
                 remove_stopwords, strip_short]

# Title of the Streamlit App
st.title('Job Description Data Retrieval and Similarity Search')

st.markdown('''
## Job descriptions corpus:

The data in this activity comes from two sources. First is from a dataset on data science-related jobs in Europe. The relevant URLs are:

1. https://methodmatters.github.io/data-jobs-europe/
2. https://github.com/methodmatters/data-jobs-europe (contains a notebook for studying jobs.)
3. https://www.r-bloggers.com/2022/04/text-analysis-of-job-descriptions-for-data-scientists-data-engineers-machine-learning-engineers-and-data-analysts/

These are the columns in the dataset:

| **Attribute**                    | **Description**                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------|
| employer_id                      | Unique identifier for the employer                                                                   |
| company_name                     | Name of the company posting the job                                                                   |
| job_id                           | Unique identifier for the job description                                                            |
| job_title                        | Job title as input by the employer                                                                   |
| job_function                     | Harmonized job title managed by job listing site (values: data scientist, data engineer, data analyst, machine learning engineer) |
| job_description_text             | Text of the job advertisement (not all in English)                                                   |
| job_skills                       | List of employer-determined skill keywords (in English)                                              |
| education_desired                | List of employer-requested educational attainments                                                   |
| job_location                     | City where the job is located                                                                        |
| company_hq_location              | City and country of the company's headquarters                                                       |
| company_sector_name              | Sector in which the company is active                                                                |
| company_industry                 | Industry in which the company is active                                                              |
| company_type                     | Type of company (e.g., government, private company, etc.)                                            |
| company_size                     | Number of employees that the company has                                                             |
| company_revenue                  | Annual company revenue in USD                                                                        |
| company_year_founded             | Year the company was founded                                                                         |
| company_website                  | Company website                                                                                      |
| rating_global                    | Site users' overall rating of company                                                                |
| rating_comp_ben                  | Site users' rating of compensation & benefits (pay, bonus, etc.)                                     |
| rating_culture_values            | Site users' rating of culture and values                                                             |
| rating_career_opportunities      | Site users' rating of career opportunities                                                           |
| rating_w_life_balance            | Site users' rating of work-life balance                                                              |
| rating_sr_mgt                    | Site users' rating of senior management                                                              |
| query_country                    | Country used in the query to scrape the job ad                                                       |
| date_job_posted                  | Date the job advertisement was posted                                                                |
| date_job_expires                 | Date the job advertisement expires                                                                   |
| age_job_posting_days             | Age of the job ad (in days) on the date that it was scraped                                          |
| scraping_date                    | Date the job was scraped                                                                             |
| language                         | Language of the job description text (determined via the langid package in Python)                   |

The second dataset comes from [hugging face](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions). The columns in this dataset are:

* `company_name`
* `job_description`
* `position_title`
* `description_length`
* `model_response`, and
* `job_description_text`

For this activity, we have extracted the `job_description_text` for English
jobs from the first dataset, and combined it with the `job_description` and
`model_response` columns from the second dataset. There are a total of 3869 +
853 = 4722 job descriptions.''')

# Load Data at Startup
@st.cache_resource
def get_all_data():
    url = 'https://github.com/methodmatters/data-jobs-europe/raw/master/Data/omnibus_jobs_df.pickle'
    response = requests.get(url)
    omnibus_jobs_df = pickle.loads(response.content)
    en_jobs_df = omnibus_jobs_df[omnibus_jobs_df.language == 'en']

    ds = load_dataset("jacob-hugging-face/job-descriptions", split='train')
    hf_jobs = ds.to_pandas()
    hf_jobs['job_description_text'] = hf_jobs.apply(lambda x: ' '.join([x.job_description, x.model_response[2:-1]]), axis=1)
    return pd.concat([en_jobs_df.job_description_text, hf_jobs.job_description_text], ignore_index=True)

# Combine all job descriptions
all_jobs = get_all_data()

@st.cache_resource
def get_tfidf_model():
    nltk.download('wordnet')
    wn = WordNetLemmatizer()

    all_job_strings = all_jobs.values
    #all_job_strings[:3]
    all_jobs_tokenized = [preprocess_string(x, CUSTOM_FILTER) for x in all_job_strings]
    dct = gensim.corpora.Dictionary(all_jobs_tokenized)
    bow_corpus = [dct.doc2bow(text) for text in all_jobs_tokenized]
    tfidf = gensim.models.TfidfModel(dictionary=dct)

    index = gensim.similarities.Similarity(None, corpus=tfidf[bow_corpus], num_features=len(dct))
    return index, dct, tfidf

index,dct,tfidf = get_tfidf_model()   

@st.cache_resource
def d2v_model():
    all_job_strings = all_jobs.values
    all_jobs_tokenized = [preprocess_string(x, CUSTOM_FILTER) for x in all_job_strings]
    train_corpus = [gensim.models.doc2vec.TaggedDocument(x, [i]) for i,x in enumerate(all_jobs_tokenized)]
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    return model

model = d2v_model()

# Random Number Generator initialized once
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng(2914)

# Random Sampling of Job Descriptions
st.header('Random Job Description Sample')

with st.form("random_sample"):
    sample_size = st.slider('Select number of job descriptions to sample:', min_value=1, max_value=20, value=5)
    generate_sample = st.form_submit_button('Generate')
    clear_sample = st.form_submit_button('Clear')

if generate_sample:
    out_sample = st.session_state.rng.choice(all_jobs, size=sample_size)

    #out_sample = rng.choice(all_jobs, size=sample_size)
    out_string = ''
    for xx in out_sample:
        out_string += "-------\n"
        out_string += f"{xx} \n\n"
    st.session_state.generated_text = out_string
    #st.session_state.generated_text = "\n".join([f"-------\n{desc}" for desc in out_sample])
elif clear_sample:
    st.session_state.generated_text = ""  # Clear the stored text

# Use Expander to show/hide the generated job descriptions
with st.expander("Show/Hide Job Descriptions", expanded=True):
    if "generated_text" in st.session_state and st.session_state.generated_text:
        st.write(st.session_state.generated_text)
    else:
        st.write("No job descriptions to display. Please generate some.")

st.header('Query Job Descriptions with Tfidf')

with st.form("query_form"):
    query = st.text_input("Type query:", value='')
    return_no = st.number_input("No. to return:", min_value=1, max_value=20, value=10)
    submit_query = st.form_submit_button('Submit')

if submit_query:
    # Preprocess the query
    wn = WordNetLemmatizer()
    q1 = [wn.lemmatize(x) for x in preprocess_string(query, CUSTOM_FILTER)]
    
    # Calculate similarities
    sims = index[tfidf[dct.doc2bow(q1)]]
    flipped_sims = np.flip(np.sort(sims))

    # Get top results
    q1_results = np.argsort(-sims)[:return_no]
    out_string = ''
    for i, qq in enumerate(q1_results):
        out_string += "---\n"
        out_string += f"Rank: {i+1}, Similarity with query: {flipped_sims[i]:.3f}\n"
        out_string += f"{all_jobs.values[qq]} \n\n"

    # Display the results in an expander
    with st.expander("Show/Hide Retrieved Descriptions", expanded=True):
        st.text_area("Retrieved descriptions:", value=out_string, height=300)

st.header('Query Job Descriptions with Doc2Vec')

with st.form("query_form_w2v"):
    query_w2v = st.text_input("Type query:", value='', key='doc2vec_query')
    return_no_w2v = st.number_input("No. to return:", min_value=1, max_value=20, value=10, key='doc2vec_return_no')
    submit_query_w2v = st.form_submit_button('Submit')

if submit_query_w2v:
    # Infer the vector for the query using the Doc2Vec model
    query_vec = model.infer_vector(query_w2v.split())
    most_similar = model.dv.most_similar([query_vec], topn=return_no_w2v)
    
    # Prepare the output string
    out_string_w2v = ''
    for i, z in enumerate(most_similar):
        out_string_w2v += "---\n"
        out_string_w2v += f"Rank: {i+1}, Similarity with query: {z[1]:.3f}\n"
        out_string_w2v += f"{all_jobs.values[z[0]]} \n\n"

    # Display the results in an expander
    with st.expander("Show/Hide Retrieved Descriptions (Doc2Vec)", expanded=True):
        st.text_area("Retrieved descriptions:", value=out_string_w2v, height=300)


