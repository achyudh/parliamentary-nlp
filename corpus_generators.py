import os, json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora


english_stopwords = set(stopwords.words('english'))
common_words = ['said', 'would', 'issue', 'raised', 'several', 'members', 'this', 'also', 'learnt', 'need']
for x in common_words:
    english_stopwords.add(x)


def filter_word_ls(x):
    if x.isupper() or x.isdigit() or len(x) < 4:
        return False
    if x in english_stopwords:
        return False
    if 'xx' in x or 'iii' in x:
        return False
    if x.isalnum():
        return True


def generate_ls_corpus(folder_name):
    session_enum = dict()
    session_docs = list()
    for file_name in os.listdir(folder_name):
        current_doc = list()
        with open(folder_name+file_name, 'r', encoding="utf8") as txt_file:
            for line in txt_file:
                line = [x.lower() for x in word_tokenize(str(line)) if filter_word_ls(x)]
                if len(line) > 3:
                    current_doc.extend(line)
        if len(current_doc) == 0:
            continue
        month = int(file_name[3:5])
        session_name = file_name[6:10]
        if 2 <= month <= 5:
            session_name += "-Budget"
        elif 7 <= month <= 9:
            session_name += "-Monsoon"
        else:
            session_name += "-Winter"
        session_no = session_enum.get(session_name, None)
        if session_no is None:
            session_no = len(session_docs)
            session_enum[session_name] = session_no
            session_docs.append(current_doc)
            # session_docs.append([" ".join(current_doc)])
        else:
            session_docs[session_no].extend(current_doc)
            # session_docs[session_no].append(" ".join(current_doc))

    with open("data/ls_session_enum_corpus.json", 'w') as json_file:
        json.dump(session_enum, json_file)

    for x1 in range(len(session_docs)):
        session_docs[x1] = set(session_docs[x1])

    ls_dictionary = corpora.Dictionary(session_docs)
    ls_dictionary.save('data/ls_debates.dict')

    corpus = [ls_dictionary.doc2bow(text) for text in session_docs]
    corpora.MmCorpus.serialize('data/ls_debates.mm', corpus)


def filter_word_rs(x):
    if x.isupper() or x.isdigit() or len(x) < 4:
        return False
    if x in english_stopwords:
        return False
    if 'xx' in x or 'iii' in x:
        return False
    if x.isalnum():
        return True


def generate_rs_corpus(folder_name):
    session_enum = dict()
    session_docs = list()
    for file_name in os.listdir(folder_name):
        current_doc = list()
        with open(folder_name+file_name, 'r', encoding="utf8") as txt_file:
            for line in txt_file:
                line = [x.lower() for x in word_tokenize(str(line)) if filter_word_ls(x)]
                if len(line) > 3:
                    current_doc.extend(line)
        if len(current_doc) == 0:
            continue

        if file_name[1] == 'S':
            month = int(file_name[5:7])
            session_name = '20' + file_name[8:10]
        elif file_name[0] == 'S':
            month = int(file_name[4:6])
            session_name = '20' + file_name[7:9]
        else:
            month = int(file_name[3:5])
            session_name = '20' + file_name[6:8]

        if 2 <= month <= 5:
            session_name += "-Budget"
        elif 7 <= month <= 9:
            session_name += "-Monsoon"
        else:
            session_name += "-Winter"
        session_no = session_enum.get(session_name, None)
        if session_no is None:
            session_no = len(session_docs)
            session_enum[session_name] = session_no
            session_docs.append(current_doc)
            # session_docs.append([" ".join(current_doc)])
        else:
            session_docs[session_no].extend(current_doc)
            # session_docs[session_no].append(" ".join(current_doc))

    with open("data/rs_session_enum_corpus.json", 'w') as json_file:
        json.dump(session_enum, json_file)

    for x1 in range(len(session_docs)):
        session_docs[x1] = set(session_docs[x1])

    ls_dictionary = corpora.Dictionary(session_docs)
    ls_dictionary.save('data/rs_debates.dict')

    corpus = [ls_dictionary.doc2bow(text) for text in session_docs]
    corpora.MmCorpus.serialize('data/rs_debates.mm', corpus)


generate_ls_corpus(folder_name="data/ls_debates/")
generate_rs_corpus(folder_name="data/rs_debates/")
