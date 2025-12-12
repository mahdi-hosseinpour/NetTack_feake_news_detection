# download_nltk.py

import nltk

print("Downloading averaged_perceptron_tagger_eng ...")
nltk.download('averaged_perceptron_tagger_eng')
print("Download completed!")

# For completeness, download these as well (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

print("All required NLTK datasets have been downloaded!")