import pickle
import re

import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def track_para_id(sentence, para_id):
    if sentence.strip()[0].isdigit():
        match = re.match(r'\d+\.', sentence.strip())
        if match is not None:
            number = int(sentence.strip().split('.')[0])
            if number == para_id+1:
                para_id = number
    elif para_id==None:
        para_id = 0

    return para_id


def track_speaker(sentence,speaker):
    new_judge = False
    names = None
    if sentence.startswith("LORD") or sentence.startswith("LADY") or sentence.startswith("BARONESS"):
        match = re.search(r'with whom (.+?) agree', sentence)
        if match:
            judges = match.group(1)
            names = re.findall(r'(?:Lord|Lady|Baroness) (\w+)', judges)
        speaker = ' '.join(sentence.split()[:2]).lower() # Extract speaker name from sentence
        speaker = speaker.replace('lord', '').strip()
        speaker = speaker.replace(':', '').strip()
        new_judge = True
    elif speaker == None:
        speaker = 'None'
    return speaker, new_judge, names

def new_case(filename, train=False):
    nlp = spacy.load("en_core_web_sm")
    speaker = None
    para_id = None
    case = filename.split(".")[0]
    with open('data/UKHL_txt/'+filename, "r", encoding="utf-8") as file:
        text = file.read()
    doc = nlp(text)
    old_sentences = [sent.text for sent in doc.sents]
    sentences = []
    for sentence in old_sentences:
        split_sentences = sentence.split('\n')
        sentences.extend(split_sentences)

    sentences = [sent.strip() for sent in sentences if sent.strip()]

    '''sentences = re.split(r'[\n\.]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]'''
    max_line = len(sentences)
    vectorizer = TfidfVectorizer()
    data = pd.read_csv("AI.csv")
    X = vectorizer.fit_transform(data['body'])
    if train:
        y_to = data['to']
        print('start split')
        X_train, X_test, y_to_train, y_to_test = train_test_split(X, y_to, test_size=0.2, random_state=42)

        #model_to = LogisticRegression()
        model_to = RandomForestClassifier()
        model_to.fit(X_train, y_to_train)

        y_to_pred = model_to.predict(X_test)
        accuracy_to = accuracy_score(y_to_test, y_to_pred)

        print("Accuracy for 'to':", accuracy_to)
        with open('RF_to.pkl', 'wb') as f:
            pickle.dump(model_to, f)
    else:
        with open('RF_to.pkl', 'rb') as f:
            model_to = pickle.load(f)

    line_num =0
    results = []
    for sentence in sentences:
        speaker, new_judge, names = track_speaker(sentence,speaker)
        #print(speaker, new_judge, name)
        para_id = track_para_id(sentence,para_id)
        new_X = vectorizer.transform([sentence])
        #predicted_to = mlb.inverse_transform(classifier.predict(new_X))
        predicted_to = model_to.predict(new_X)
        #print(predicted_to)
        pos = round(line_num/max_line, 1)
        if names is not None:
            for name in names:
                add_sentence = '------------- NEW JUDGE --------------- '
                para_id = track_para_id(add_sentence, para_id)
                results.append({'case': case, 'line': line_num, 'para_id': para_id, 'body': add_sentence, 'from': 'None',
                                'to': 'None', 'relation': 'NAN', 'pos': pos, 'mj': 'NAN'})
                line_num += 1
                results.append({'case': case, 'line': line_num, 'para_id': para_id, 'body': 'LORD ' + name.upper(), 'from': name,
                                'to': 'None', 'relation': 'NAN', 'pos': pos, 'mj': 'NAN'})
                line_num += 1
                AS = 'For the reasons given by ' +speaker+ ' in his opinion I would also make the order which he proposes'
                results.append(
                    {'case': case, 'line': line_num, 'para_id': para_id, 'body': AS, 'from': name,
                     'to': speaker, 'relation': 'fullagr', 'pos': pos, 'mj': 'NAN'})
                line_num += 1
        if new_judge:
            add_sentence = '------------- NEW JUDGE --------------- '
            para_id = track_para_id(add_sentence, para_id)
            results.append({'case': case, 'line': line_num, 'para_id': para_id, 'body': add_sentence, 'from': 'None',
                            'to': 'None', 'relation': 'NAN', 'pos': pos, 'mj': 'NAN'})
            line_num += 1
        results.append({'case': case ,'line': line_num ,'para_id': para_id,'body': sentence, 'from': speaker, 'to': predicted_to[0], 'relation':'NAN', 'pos': pos, 'mj':'NAN'})
        line_num += 1

    df = pd.DataFrame(results)
    df.to_csv('data/UKHL_csv/'+case + '.csv', index=False)
    return df

def rewrite_rel(predicted, filename):
    case = filename.split(".")[0]
    original_data = pd.read_csv('data/UKHL_csv/' + case + '.csv')

    for index, row in predicted.iterrows():
        original_data.loc[(original_data['line'] == row['line']), 'relation'] = row['relation']

    original_data.to_csv('data/UKHL_csv/' + case + '.csv', index=False)

def rewrite_mj(mj, filename):
    case = filename.split(".")[0]
    original_data = pd.read_csv('data/UKHL_csv/' + case + '.csv')
    list = []
    for index, row in mj.iterrows():
        list.append(row['mj'])
    original_data["mj"] = original_data["mj"].replace(original_data["mj"].unique(), list)
    original_data.to_csv('data/UKHL_csv/' + case + '.csv', index=False)

def rewrite_to(judge_map, filename):
    case = filename.split(".")[0]
    original_data = pd.read_csv('data/UKHL_csv/' + case + '.csv')
    case_mapping = judge_map.get(case, {})
    judges = list(dict(case_mapping.items()).keys())
    for index, row in original_data[original_data['relation'] != 'NAN'].iterrows():
        from_judge = row['from']
        to_judge = None
        if from_judge in judges:
            to_set = case_mapping.get(from_judge)
            if None not in to_set:
                to_judge = '+'.join(to_set)
            else:
                to_judge = 'self'
        original_data.at[index, 'to'] = to_judge

    original_data.to_csv('data/UKHL_csv/' + case + '.csv', index=False)

#new_case('UKHL20012-old.txt',False)