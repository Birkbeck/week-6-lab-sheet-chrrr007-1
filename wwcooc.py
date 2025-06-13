import nltk
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def build_cooccurrence_matrix(text):
    # Initialize co-occurrence dictionary
    cooccurrence = defaultdict(lambda: defaultdict(int))

    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    for sentence in sentences:
        # Tokenize sentence into words and normalize
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in string.punctuation]

        unique_words = set(words)

        for word in unique_words:
            for co_word in unique_words:
                if word != co_word:
                    cooccurrence[word][co_word] += 1

    return cooccurrence

# Example usage
if __name__ == "__main__":
    with open('Dorian_Gray-Wilde-1890.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        matrix = build_cooccurrence_matrix(text)
    print(matrix["coat"])                   #prints the cooccuring words with "coat"
    # print(matrix["coat"]["flower"])           #prints the nonumber of times both words cooccur together
    exit()
    # Display co-occurrence counts
    for word, co_words in matrix.items():
        print(f"{word}: {dict(co_words)}")
