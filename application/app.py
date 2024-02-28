import pickle
import string
import codecs
import string
import re
from nltk.corpus import stopwords
import json
import numpy as np
import fasttext
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

nep_stopwrods = stopwords.words("nepali")


def remove_emojis_english_and_numbers(data):
    """
    Removes emojis, non-nepali texts and numbers from the given text
    """
    # Removes emoji from given data
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    res = re.sub(emoj, "", data)
    res = re.sub("[0-9]+", "", res)
    return re.sub("[a-zA-Z]", "", res)


def preprocess_text(data):
    """
    Cleans the given textual data
    Removes special characters, english texts, numbers, and stopwords
    """
    if type(data) == float:
        return data
    data = (
        data.replace("-", " ")
        .replace("—", " ")
        .replace("‘", " ")
        .replace("’", " ")
        .replace("।", " ")
        .replace("–", " ")
        .replace("“", " ")
        .replace("”", " ")
        .replace("\n", " ")
        .replace("–", " ")
        .replace(" : ", ": ")
    )
    no_extra_spaces = " ".join(data.split())
    no_emoji_english_numbers = remove_emojis_english_and_numbers(no_extra_spaces)
    no_punc = "".join(
        [
            char
            for char in no_emoji_english_numbers
            if char not in list(string.punctuation)
        ]
    )
    extra = " ".join(no_punc.split())
    no_num = "".join([char for char in extra if char not in "०१२३४५६७८९"])
    no_stopwords = [
        word.strip() for word in no_num.split() if word.strip() not in nep_stopwrods
    ]
    return " ".join(no_stopwords)


def avg_word_vector(text, model, preprocess=False):
    """
    Takes in a text and returns the average out word vector over all the words
    """
    if preprocess:
        text = preprocess_text(text)
    word_vectors = []
    words = text.split()
    for word in words:
        word_vectors.append(model.get_word_vector(word))
    return np.mean(np.array(word_vectors), axis=0)


def sentence_vectors(sentences, model):
    """
    Computes the sentence vector for each of the input sentences
    """
    sentence_vectors = []
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        sentence_vector = avg_word_vector(sentence, model)
        sentence_vectors.append(sentence_vector)
    return sentence_vectors


def get_top_n_sentences(sentences, model, top_n=5):
    """
    Generates the top-n most prominent sentences from the given list of sentences
    """
    sen_vecs = sentence_vectors(sentences, model)
    similarity_matrix = cosine_similarity(sen_vecs)
    G = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(G)
    sentences_scores = [[scores[i], sentences[i]] for i in range(len(sentences))]
    sentences_rank = sorted(sentences_scores, reverse=True)
    summary = []
    for i in range(top_n):
        summary.append(sentences_rank[i][1])
    return " ".join(" । ".join(summary).split())


def summarize(input_news, input_news_path, model):
    """
    Utility function that handles all the steps related to news summarization
    Returns true is the text is successfully summarized, else returns false
    """
    while True:
        try:
            top_n = int(input("Enter no. of senetnces in summarized news: "))
            break
        except:
            print("Please enter a valid numeric value for top-n")
    print("\n")

    # Check if the text file is empty
    if len(input_news.strip()) == 0:
        print("The file is empty. Please insert some news in the file and try again!!!")
        return False

    # If the text file is not empty
    else:
        # Remove the new line escape sequence
        news = input_news.replace("\n", " ")

        # Split the news into sentences
        sentences = news.split("।")

        # Remove unnecessary whitespaces from each sentences
        sentences = [sentence for sentence in sentences if sentence.strip() != ""]

        # Check if the no of setences in output given by user exceeds the no of sentences in the input news
        if len(sentences) < int(top_n):
            print(
                "The provided no. of sentences in output is more than sentences in input news!!!"
            )
            return False
        else:
            # Generate summary
            res = get_top_n_sentences(sentences, model, top_n=int(top_n))

            # Create a file "summary_topn.txt" into the input news directory and insert the summary into the file

            # Get the path of directory of input file
            file_dir = input_news_path.split("/")[:-1]

            output_file_path = "/".join(file_dir) + f"/summarized_{top_n}.txt"
            with codecs.open(output_file_path, "w", encoding="utf-8") as f:
                f.write(res)
            print(f"Summarized news saved in location: {output_file_path}")
            return True


def make_array(text):
    """
    Takes in a string and converts it into array.
    Specifically used to convert stringified news embeddings to array
    """
    text = text.replace("\n", "").replace("[", "").replace("]", "")
    return np.fromstring(text, sep=" ")


def recommend(model, input_news, input_news_path, df_news_embeddings):
    """
    Utility function that handles all the steps related to news recommendation
    Returns true is recommendation is successfull, else returns false
    """
    while True:
        try:
            top_n = int(input("Enter no. of news to be recommended(1-10): "))
            if top_n >= 1 and top_n <= 10:
                break
        except:
            print("Please enter a valid numeric value for top-n")
    print("\n")

    # Check if the text file is empty
    if len(input_news.strip()) == 0:
        print("The file is empty. Please insert some news in the file and try again!!!")
        return False

    # If the text file is not empty
    else:
        print("Recommending.....")
        # Remove the new line escape sequence
        news = input_news.replace("\n", " ")

        # Remove unnecessary whitespaces from the news
        news = " ".join(news.split())

        # Clean the input news
        cleaned_news = preprocess_text(news)

        # We will only consider first N words from the news for recommendation,
        # Any news with more than N words will be trimmed to have at most N words

        # The value of N is stored in the CONSTANTS.json file
        with codecs.open(
            "../News Recommender/CONSTANTS.json", "r", encoding="utf-8"
        ) as f:
            constants = json.load(f)

        # Trim the cleaned input news if it has more than N words
        if len(cleaned_news.split()) > constants["max_news_length"]:
            cleaned_news = " ".join(
                cleaned_news.split()[: constants["max_news_length"]]
            )

        # Compute average word vector for the news
        input_news_embeddings = avg_word_vector(cleaned_news, model, preprocess=False)

        # Find the cosine similarity between the input news embeddings and all the other news embeddings in our storage
        cosine_similarities = [
            (
                category,
                title,
                news,
                cosine_similarity([embedding], [input_news_embeddings])[0][0],
            )
            for category, title, news, embedding in zip(
                df_news_embeddings["category"],
                df_news_embeddings["title"],
                df_news_embeddings["news"],
                df_news_embeddings["news_embeddings"],
            )
        ]

        # Sort the cosine similarities in descending order
        cosine_similarities.sort(reverse=True, key=lambda k: k[len(k) - 1])

        # Select the top-n news articles with highest cosine similarity
        recs = cosine_similarities[:top_n]

        # Get the path of directory of input file
        file_dir = input_news_path.split("/")[:-1]

        # For each recommendation
        for index, rec in enumerate(recs):
            # Create a file "recommendation_n.txt" into the input news directory and save the nth recommendation in it
            output_file_path = "/".join(file_dir) + f"/recommendation_{index + 1}.txt"
            store = f"Catgeory:\n{rec[0]}\n\nTitle:\n{rec[1]}\n\nNews:\n{rec[2]}"
            with codecs.open(output_file_path, "w", encoding="utf-8") as f:
                f.write(store)
            print(f"Recommended news saved in location: {output_file_path}")
        return True


def main():
    print("Welcome To Nepali News Summarization and Recommendation Tool")
    print("\n")
    print("You need to provide the path(seperated by /) of input news as a text file")
    print("The output(s) will be stored in new file(s) within the input news directory")
    print("\n")

    print("Loading embedding model....")

    # Load the fasttext model
    embedding_model = fasttext.load_model(
        "../fasttext model/fasttext_summarizer_embeddings.bin"
    )
    print("Embedding model loaded.")
    print("\n")

    print("Loading news embeddings...")
    df_news_embeddings = pd.read_csv("../news_embeddings/news_raw_with_embeddings.csv")
    df_news_embeddings["news_embeddings"] = df_news_embeddings["news_embeddings"].apply(
        make_array
    )
    print("News embeddings loaded.")
    print("\n")

    start = "y"
    while start == "y":
        # Get user inputs: file_path and no. of sentences to be in the output summary
        while True:
            input_news_path = input("Enter News(.txt file) Path: ")
            try:
                with codecs.open(input_news_path, "r", encoding="utf-8") as f:
                    input_news = f.read()
                break
            except:
                print(
                    "Invalid file path. Please enter a valid file path and try again!!!"
                )
        print("\n")
        task = "task"

        # Until a task is determined, keep asking user
        while task not in ["s", "r"]:
            task = input("Do you want summarization(s) or recommendation(r): ")

        if task == "s":
            res = summarize(embedding_model, input_news, input_news_path)
        else:
            res = recommend(
                embedding_model, input_news, input_news_path, df_news_embeddings
            )
        print("\n")
        start = "start"
        while start not in ["y", "n"]:
            # Check if user wants to cotinue
            start = input("Do you want to continue?(y/n) ")


if __name__ == "__main__":
    main()
