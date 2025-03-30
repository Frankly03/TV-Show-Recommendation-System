import pandas as pd 
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import gradio as gr

load_dotenv()

# Loading Dataset
shows = pd.read_csv('shows_with_ratings.csv')

# adding a column to the original url path
BASE_URL = "https://image.tmdb.org/t/p/original"
shows['poster_url'] = BASE_URL + shows['poster_path']

shows['poster_url'] = np.where(
    shows['poster_url'].isna(),
    "cover-not-found.jpg", 
    shows['poster_url'],
)


raw_documents = TextLoader("tagged_overview.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db_shows = Chroma.from_documents(
    documents,
    embedding=hf_embeddings
)



def retrieve_semantic_recommendations(
        query: str,
        genre: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
)-> pd.DataFrame:
           
    recs = db_shows.similarity_search(query, k=initial_top_k)

    shows_list = []
    for rec in recs:
        first_part = rec.page_content.strip('"').split()[0]

        if first_part.isdigit():
            shows_list.append(int(first_part))
        else:
            print(f"skipping invalid show ID : {first_part}")
    
    shows_recs = shows[shows['id'].isin(shows_list)]

    
    # Apply genre filtering
    if genre != "All":
        shows_recs = shows_recs[shows_recs['primary_genre'] == genre]

    # Apply tone filtering
    if tone == "Happy":
        shows_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == "Surprising":
        shows_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == "Angry":
        shows_recs.sort_values(by='angry', ascending=False, inplace=True)
    elif tone == "Suspenseful":
        shows_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == "Sad":
        shows_recs.sort_values(by='sadness', ascending=False, inplace=True)

    # **Sort by weighted popularity score**
    shows_recs = shows_recs.sort_values(by='weighted_score', ascending=False)

    return shows_recs.head(final_top_k)


def recommend_shows(
        query: str,
        genre: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, genre, tone)
    results = []

    for _, row in recommendations.iterrows():
        overview = row['overview']
        truncated_desc_split = overview.split()
        truncated_overview = " ".join(truncated_desc_split[:30]) + "...."

        caption = f'{row['title_and_tagline']} : {truncated_overview}'
        results.append((row['poster_url'], caption))

    return results

genre = ["All"] + sorted(shows['primary_genre'].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Shows recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a show:",
                                placeholder='e.g., A show about mystery')
        genre_dropdown = gr.Dropdown(choices=genre, label="Select a genre:", value ="All")
        tone_dropdown = gr.Dropdown(choices=tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label="Recommendations", columns=8, rows=2)

    submit_button.click(
        fn=recommend_shows,
        inputs = [user_query, genre_dropdown, tone_dropdown],
        outputs = output
    )
    
    

if __name__ == "__main__":
    dashboard.launch(share=True)
