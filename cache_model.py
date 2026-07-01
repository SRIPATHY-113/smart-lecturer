from sentence_transformers import SentenceTransformer
print("Downloading and caching model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
test = model.encode(["test sentence"])
print("Model cached and working! Shape:", test.shape)
print("Done! You can now run api.py")
