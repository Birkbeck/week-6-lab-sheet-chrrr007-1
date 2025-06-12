import matplotlib.pyplot as plt
import os

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

print(os.getcwd())
exit()

with open(r'data/sentences.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# 2. Calculate embeddings by calling model.encode()
s_embeddings = model.encode(sentences)



# Reduce dimensionality to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(s_embeddings)

# Plot it
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(reduced):
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, sentences[i], fontsize=9)

plt.title("2D Sentence Embedding Visualization (PCA)")
plt.show()