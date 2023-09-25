import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import requests
import urllib
from PIL import Image
import json

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from scipy.spatial import distance

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Create a Streamlit app
st.set_page_config(
    page_title="Recommender System",
    layout="wide",
)

# Center-align subheading and image using HTML <div> tags
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <h2>Google Books Recommender System</h2>

    </div>
    """,
    unsafe_allow_html=True
)
st.image("rec1.jpg")


# Add an introductory paragraph
st.markdown("""
This Application is a book recommender system uses a content-based approach, leveraging book features to recommend similar books to users based on features of the books or characteristics of the items and the user's preferences.
""")

# Load the data into a DataFrame
df = pd.read_csv("recommender_books_with_url.csv")

# Create a checkbox to toggle the visibility of the DataFrame
show_data = st.checkbox("Preview Data")

# Display the DataFrame if the checkbox is checked
if show_data:
    st.dataframe(df)
    
# Copy df to df1
df1 = df.copy()

# text preprocessing function
def preprocess_text(text):
    # Handle NaN or float values
    if isinstance(text, float) or text is None or pd.isnull(text):
        return ''

    # Lowercase the text
    text = text.lower()

    # Remove URLs, hashtags, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers/digits
    text = re.sub(r'\b[0-9]+\b\s*', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)

# Features to preprocess
features_to_preprocess = ['title', 'description', 'author', 'genre', 'publisher', 'language', 'rating', 'page_count']

# Data preprocessing for each feature
for feature in features_to_preprocess:
    # Apply text preprocessing to each entry in the feature
    df1[feature] = df1[feature].apply(preprocess_text)

# Concatenate the preprocessed features into a single text feature
df1['combined_features'] = df1[features_to_preprocess].apply(lambda x: ' '.join(x), axis=1)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF Vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df1['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display the TF-IDF matrix shape
# st.write("TF-IDF Matrix shape:", tfidf_matrix.shape)

# Display the cosine similarity matrix shape and content
# st.write("Cosine Similarity Matrix shape:", cosine_sim.shape)
# st.write("Cosine Similarity Matrix:")
# st.write(cosine_sim)

# All Books heading
st.markdown(
    """
    <div style="margin-top: 20px; padding: 10px; background-color: #FF2525; border-radius: 10px; text-align:center; align-items:center;">
        <h2>All Books</h2>
    </div>
    """,
    unsafe_allow_html=True
)
    
# Pagination function
def display_books_with_pagination(book_data, items_per_page=5):
    total_books = len(book_data)
    num_pages = (total_books + items_per_page - 1) // items_per_page

    page = st.sidebar.slider("Section", 1, num_pages, 1)

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.write(f"Section {page}/{num_pages}:")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .book-card {
            border: 1px solid #ccc;
            padding: 12px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .book-card img {
            max-width: 50%; /* Adjust the image width */
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    for i in range(start_idx, min(end_idx, total_books)):
        book = book_data[i]
        st.markdown(
            f"""
            <div class="book-card">
                <h5>{book['title']}</h5>
                <p><strong>Author:</strong> {book['author']}</p>
                <p><strong>Genre:</strong> {book['genre']}</p>
                <p><strong>Description:</strong> {book['description']}</p>
                <img src="{book['cover_url']}" alt="Book Cover">
            </div>
            """,
            unsafe_allow_html=True
        )
book_data = df.to_dict('records')

# sidebar title
st.sidebar.title("Drag the Slider to display all books")
display_books_with_pagination(book_data)

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     /* Add custom styles here */
#     .custom-section-heading {
#         font-size: 24px;
#         font-weight: bold;
#         margin-bottom: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display a customized heading for Search for Books section
# st.markdown(
#     """
#     <div class="custom-section-heading">
#         Search for Books
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # Function to fetch book titles
# def fetch_book_titles(search_term):
#     # Filter titles that contain the search term
#     matching_titles = df[df['title'].str.contains(search_term, case=False, na=False)]
#     return matching_titles['title'].tolist()

# # Sidebar heading for search functionality
# st.sidebar.markdown("## Search for Books")

# # Search for books using autocomplete
# search_term = st.sidebar.text_input("Start typing to search for books")
# matching_titles = fetch_book_titles(search_term)

# # Display autocomplete suggestions
# selected_titles = st.sidebar.multiselect("Matching Books", matching_titles)

# # Display the selected titles
# st.write("Selected Books:", selected_titles)

# SEARCH FOR BOOKS
book_titles = df['title']
book_titles_list = df['title'].tolist()

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Add custom styles here */
    .custom-section-heading {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display a customized heading for a new section
st.markdown(
    """
    <div class="custom-section-heading">
        Search for Books
    </div>
    """,
    unsafe_allow_html=True
)

# # Input for the user to search for books
# search_query = st.text_input("Start typing to search for a book")

# # Create a placeholder for the suggestions
# suggestions_placeholder = st.empty()

# # JavaScript to dynamically update suggestions
# js_code = f"""
# <script>
# function dynamicSuggestions() {{
#     var query = document.getElementById('search_query').value.toLowerCase();
#     var suggestions = {json.dumps(book_titles_list)};
#     var matchingBooks = suggestions.filter(function(book) {{
#         return book.toLowerCase().includes(query);
#     }});
#     var suggestionsElement = document.getElementById('suggestions');
#     suggestionsElement.innerHTML = '';
#     matchingBooks.forEach(function(book) {{
#         var suggestionDiv = document.createElement('div');
#         suggestionDiv.style.padding = '5px';
#         suggestionDiv.innerHTML = book;
#         suggestionsElement.appendChild(suggestionDiv);
#     }});
# }}
# document.getElementById('search_query').addEventListener('input', dynamicSuggestions);
# </script>
# """

# # Display the JavaScript code
# st.markdown(js_code, unsafe_allow_html=True)

# # Display the suggestions in the placeholder
# if search_query:
#     matching_books = [book for book in book_titles_list if search_query.lower() in book.lower()]
#     for book in matching_books:
#         st.write(book)
# else:
#     suggestions_placeholder.write('suggestions')
    

class RecommenderSystem:
    def __init__(self, data, tfidf_matrix, cosine_sim):
        self.data = data
        self.tfidf_matrix = tfidf_matrix
        self.cosine_sim = cosine_sim

        # Build the index using Faiss
        self.vectors = self.tfidf_matrix.toarray()
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors.astype('float32'))

    def recommend_books(self, book_title, top_n=5):
        # Get the index of the book using its title
        book_index = self.data[self.data['title'] == book_title].index

        # If the book title is not found, return an empty list
        if len(book_index) == 0:
            print("Book not found.")
            return []

        book_index = book_index[0]

        # Get the nearest neighbors using Faiss
        _, nearest_neighbors = self.index.search(self.vectors[book_index].reshape(1, -1).astype('float32'), top_n+1)

       # Get the indices and similarity scores of the recommended books
        recommended_books_info = []
        for index in nearest_neighbors[0]:
            if index != book_index:
                title = self.data['title'].iloc[index]
                author = self.data['author'].iloc[index]
                genre = self.data['genre'].iloc[index]
                publisher = self.data['publisher'].iloc[index]
                similarity_score = self.cosine_sim[book_index][index]
                image_url = self.data['cover_url'].iloc[index]

                recommended_books_info.append({
                    'title': title,
                    'author': author,
                    'genre': genre,
                    'publisher': publisher,
                    'similarity_score': similarity_score,
                    'image_url': image_url
                })

        return recommended_books_info
    
# Create a recommender system
recommender_system = RecommenderSystem(df, tfidf_matrix, cosine_sim)

st.markdown(
    """
    <style>
    .book-card {
        border: 1px solid #ccc;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .book-card img {
        max-width: 100%;
        border-radius: 5px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .book-info {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
def recommend_books_with_images(book_title, top_n=5):
    recommended_books_info = recommender_system.recommend_books(book_title, top_n)
    return recommended_books_info

def display_book_card(book_info):
    # Create a styled card for displaying book information
    st.write(
        f"""
        <div class="book-card">
            <img src="{book_info['image_url']}" alt="Book Cover">
            <div class="book-details">
                <h3>{book_info['title']}</h3>
                <p><strong>Author:</strong> {book_info['author']}</p>
                <p><strong>Genre:</strong> {book_info['genre']}</p>
                <p><strong>Publisher:</strong> {book_info['publisher']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

book_title = st.text_input("Enter a book title:")

if st.button("Search"):
    if book_title:
        st.write(f"Searching for books similar to '{book_title}'...")
        recommended_books_info_ = recommend_books_with_images(book_title, top_n=5)

        # Display details of the searched book
        if recommended_books_info_:
            st.write("Details of the searched book:")
            display_book_card(recommended_books_info_[0])

            # Ask if the user wants to see other related books
            show_related_books = st.radio("Would you like to see other books related to what you searched?", ("Yes", "No"))

            if show_related_books == "Yes":
                st.write("Recommended books:")
                for book_info in recommended_books_info_[1:]:
                    display_book_card(book_info)
                else:
                    st.write("No recommendations to display.")



