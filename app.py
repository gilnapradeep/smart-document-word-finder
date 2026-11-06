
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from document_reader import extract_text

from preprocessing import (
    preprocess_document,
    preprocess_text
)

from similarity import (
    create_chunks,
    find_similar_words
)


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Smart Document Word Finder",
    page_icon="📚",
    layout="wide"
)


# --------------------------------
# 2. Application Title
# --------------------------------

st.title("📚 Smart Document Word Finder")

st.write(
    "Upload a large document and discover similar "
    "and related words using NLP techniques."
)


# --------------------------------
# 3. Sidebar
# --------------------------------

st.sidebar.header("Project Information")

st.sidebar.write("Technologies Used:")

st.sidebar.write("- Python")
st.sidebar.write("- NLTK")
st.sidebar.write("- Pandas")
st.sidebar.write("- Scikit-learn")
st.sidebar.write("- Gensim")
st.sidebar.write("- Streamlit")

st.sidebar.divider()

st.sidebar.write("NLP Techniques:")

st.sidebar.write("- Text Preprocessing")
st.sidebar.write("- Tokenization")
st.sidebar.write("- Stopword Removal")
st.sidebar.write("- Lemmatization")
st.sidebar.write("- TF-IDF")
st.sidebar.write("- Word2Vec")
st.sidebar.write("- Cosine Similarity")


# --------------------------------
# 4. Upload Document
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF or TXT document",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    try:

        # --------------------------------
        # 5. Extract Text
        # --------------------------------

        raw_text = extract_text(uploaded_file)

        if not raw_text.strip():

            st.error("No readable text found in this document.")
            st.stop()

        st.success("Document uploaded successfully!")

        # --------------------------------
        # 6. Document Information
        # --------------------------------

        st.subheader("1. Document Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Characters",
            f"{len(raw_text):,}"
        )

        col2.metric(
            "Words",
            f"{len(raw_text.split()):,}"
        )

        col3.metric(
            "Sentences",
            f"{len(preprocess_document(raw_text)):,}"
        )

        # Original document preview

        with st.expander("View Original Document"):

            st.text(raw_text[:5000])

            if len(raw_text) > 5000:

                st.caption(
                    "Showing first 5,000 characters."
                )

        # --------------------------------
        # 7. Text Preprocessing
        # --------------------------------

        st.subheader("2. Text Preprocessing")

        sentences = preprocess_document(raw_text)

        if len(sentences) < 2:

            st.error(
                "Not enough usable text. "
                "Please upload a larger document."
            )

            st.stop()

        # Convert tokens into readable text

        cleaned_text = " ".join(
            [" ".join(sentence) for sentence in sentences]
        )

        st.success("Text preprocessing completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.write("Original Text")

            st.text(raw_text[:1000])

        with col2:

            st.write("Cleaned Text")

            st.text(cleaned_text[:1000])

        # --------------------------------
        # 8. Word Statistics
        # --------------------------------

        all_words = [
            word
            for sentence in sentences
            for word in sentence
        ]

        unique_words = set(all_words)

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Processed Words",
            f"{len(all_words):,}"
        )

        col2.metric(
            "Unique Words",
            f"{len(unique_words):,}"
        )

        # --------------------------------
        # 9. Create Document Chunks
        # --------------------------------

        chunks = create_chunks(
            sentences,
            chunk_size=5
        )

        st.write(
            f"Document chunks created: {len(chunks):,}"
        )

        if len(chunks) < 2:

            st.warning(
                "Very few document chunks were created. "
                "TF-IDF context similarity may be limited."
            )

        # --------------------------------
        # 10. Search Similar Words
        # --------------------------------

        st.subheader("3. Find Similar Words")

        search_word = st.text_input(
            "Enter a word to find similar words",
            placeholder="Example: machine"
        )

        top_n = st.slider(
            "Number of similar words",
            min_value=5,
            max_value=20,
            value=10
        )

        search_button = st.button(
            "🔍 Find Similar Words",
            type="primary"
        )

        if search_button:

            if not search_word.strip():

                st.warning("Please enter a search word.")

            else:

                # Preprocess the search word

                processed_search = preprocess_text(
                    search_word
                )

                if not processed_search:

                    st.error("Invalid search word.")

                else:

                    # Correctly extract the first token

                    query_word = processed_search[0]

                    # Check whether word exists in document

                    if query_word not in unique_words:

                        st.warning(
                            f"'{query_word}' was not found "
                            "in the document after preprocessing."
                        )

                    else:

                        # --------------------------------
                        # 11. Calculate Similarity
                        # --------------------------------

                        with st.spinner(
                            "Calculating similar words..."
                        ):

                            results, model = find_similar_words(
                                sentences,
                                chunks,
                                query_word,
                                top_n
                            )

                        st.subheader(
                            f"4. Results for '{query_word}'"
                        )

                        if results.empty:

                            st.warning(
                                "No similar words were found."
                            )

                        else:

                            st.success(
                                "Similar words found successfully!"
                            )

                            # --------------------------------
                            # 12. Display Results
                            # --------------------------------

                            display_results = results.copy()

                            score_columns = [
                                'Word2Vec Score',
                                'TFIDF Score',
                                'Combined Score'
                            ]

                            for column in score_columns:

                                display_results[column] = (
                                    display_results[column].round(4)
                                )

                            st.dataframe(
                                display_results,
                                use_container_width=True,
                                hide_index=True
                            )

                            # --------------------------------
                            # 13. Visualization
                            # --------------------------------

                            st.subheader(
                                "5. Similarity Visualization"
                            )

                            fig, ax = plt.subplots(
                                figsize=(10, 5)
                            )

                            ax.barh(
                                results['Word'],
                                results['Combined Score']
                            )

                            ax.set_xlabel(
                                "Combined Similarity Score"
                            )

                            ax.set_ylabel("Similar Words")

                            ax.set_title(
                                f"Words Similar to '{query_word}'"
                            )

                            ax.invert_yaxis()

                            plt.tight_layout()

                            st.pyplot(fig)

                            plt.close(fig)

                            # --------------------------------
                            # 14. Download Results
                            # --------------------------------

                            csv = results.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="⬇️ Download Results as CSV",
                                data=csv,
                                file_name="similar_words.csv",
                                mime="text/csv"
                            )

                            # --------------------------------
                            # 15. Result Insights
                            # --------------------------------

                            st.subheader("6. Result Insights")

                            st.write(
                                f"Search word: **{query_word}**"
                            )

                            st.write(
                                f"Number of related words: **{len(results)}**"
                            )

                            st.write(
                                "**Word2Vec:** Identifies word relationships "
                                "based on surrounding words."
                            )

                            st.write(
                                "**TF-IDF:** Identifies words that appear "
                                "in similar document chunks."
                            )

                            st.write(
                                "**Cosine Similarity:** Measures the "
                                "similarity between numerical vectors."
                            )

                            st.write(
                                "**Combined Score:** Ranks the results "
                                "using Word2Vec and TF-IDF scores."
                            )

    except Exception as e:

        st.error(
            f"An error occurred: {str(e)}"
        )


# --------------------------------
# 16. Initial Instructions
# --------------------------------

else:

    st.info(
        "Please upload a PDF or TXT document to begin."
    )

    st.markdown(
        """
        ### How to use this application

        1. Upload a PDF or TXT document.
        2. Wait for text extraction.
        3. Review the preprocessed text.
        4. Enter a word from the document.
        5. Click Find Similar Words.
        6. View similarity scores and charts.
        7. Download the results as CSV.
        """
    )
