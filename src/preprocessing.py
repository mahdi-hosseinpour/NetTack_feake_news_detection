# src/preprocessing.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
    return tag_dict.get(tag, 'n')


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def categorize_venue(text):
    text = text.lower()
    if any(k in text for k in
           ['confer', 'press', 'speech', 'interview', 'debate', 'broadcast', 'meet', 'opinion', 'statement', 'letter',
            'ralli']):
        return 'interview'
    if any(k in text for k in ['campaign', 'ad', 'flier', 'commerci', 'mailer', 'panel', 'billboard']):
        return 'ad'
    if any(k in text for k in
           ['facebook', 'imag', 'media', 'meme', 'tweet', 'email', 'e-email', 'forum', 'blog', 'twitter']):
        return 'social media'
    if any(k in text for k in
           ['abc', 'articl', 'news', 'cnn', 'msnbc', 'book', 'journal', 'hbo', 'fox', 'column', 'newslett']):
        return 'news'
    if any(k in text for k in ['websit', 'web']):
        return 'website'
    if 'show' in text:
        return 'show'
    if 'unknown' in text or text == '':
        return 'unknown'
    return 'other'


def categorize_job(text):
    text = text.lower()
    if 'repres' in text:
        return 'U.S. representative'
    if any(k in text for k in ['governor', 'state', 'congressman', 'congresswoman', 'deleg', 'mayor', 'senat']):
        return 'state representative'
    if 'presid' in text:
        return 'president'
    if 'director' in text:
        return 'office director'
    if any(k in text for k in ['group', 'chairman', 'program']):
        return 'company'
    if any(k in text for k in ['counti', 'attorney', 'govern']):
        return 'government'
    if any(k in text for k in ['media', 'blog', 'show', 'host', 'radio', 'tv']):
        return 'media'
    if 'unknown' in text or text == '':
        return 'unknown'
    return 'other'


def preprocess_liar(train_path, test_path, valid_path):
    columns = [
        "id", "label", "statement", "subject(s)", "speaker", "speaker's job title",
        "state info", "party affiliation", "barely true counts", "false counts",
        "half true counts", "mostly true counts", "pants on fire counts", "venue"
    ]

    train = pd.read_csv(train_path, sep='\t', header=None, names=columns, on_bad_lines='skip')
    test = pd.read_csv(test_path, sep='\t', header=None, names=columns, on_bad_lines='skip')
    valid = pd.read_csv(valid_path, sep='\t', header=None, names=columns, on_bad_lines='skip')

    df = pd.concat([train, test, valid], ignore_index=True)
    # Drop the '[ID].json' column at the beginning of preprocessing (to prevent future errors)
    if '[ID].json' in df.columns:
        df = df.drop(columns=['[ID].json'])
        print("Column '[ID].json' successfully dropped")
    else:
        print("Column '[ID].json' not found — it may have a different name")
        print("Available columns:", df.columns.tolist())

    df = df.fillna({
        'speaker': 'unknown', 'subject(s)': 'unknown', 'venue': 'unknown',
        'speaker\'s job title': 'unknown', 'party affiliation': 'unknown',
        'state info': 'unknown', 'label': 'unknown'
    })

    df['statement'] = df['statement'].apply(clean_text)
    df['subject(s)'] = df['subject(s)'].str.replace(',', ' ').apply(clean_text)
    df['venue'] = df['venue'].apply(clean_text)
    df["speaker's job title"] = df["speaker's job title"].apply(clean_text)

    party_map = {
        'none': 'Unknown', 'activist': 'Other', 'organization': 'Other',
        'libertarian': 'Other', 'journalist': 'Other', 'columnist': 'Other',
        'state-official': 'Other', 'business-leader': 'Other', 'talk-show-host': 'Other',
        'government-body': 'Other', 'newsmaker': 'Other', 'county-commissioner': 'Other',
        'constitution-party': 'Other', 'labor-leader': 'Other', 'education-official': 'Other',
        'tea-party-member': 'Other', 'green': 'Other', 'liberal-party-canada': 'Other',
        'Moderate': 'Other', 'democratic-farmer-labor': 'Other',
        'ocean-state-tea-party-action': 'Other', 'independent': 'Other'
    }
    df["party affiliation"] = df["party affiliation"].replace(party_map)

    df['venue_category'] = df['venue'].apply(categorize_venue)
    df['job_category'] = df["speaker's job title"].apply(categorize_job)

    categorical_cols = ['speaker', 'label', 'state info', 'party affiliation', 'venue_category', 'job_category']
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes

    df = df[df['label'] != 255].reset_index(drop=True)
    print(df.columns.tolist())

    # Drop any column related to ID regardless of its exact name (e.g., id, ID, [ID].json, etc.)
    id_columns = [col for col in df.columns if 'id' in col.lower() or '[id]' in col.lower()]
    if id_columns:
        df = df.drop(columns=id_columns)
        print(f"ID columns dropped: {id_columns}")
    else:
        print("No ID column found")
        print("Available columns:", df.columns.tolist())
    return df
