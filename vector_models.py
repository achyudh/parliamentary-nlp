from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, json, os


def generate_tfidf_ls():
    corpus_filename = 'data/ls_debates.mm'
    p_corpus = corpora.MmCorpus(corpus_filename)
    tfidf = models.TfidfModel(p_corpus)
    pfile = open('data/ls_tfidf.pkl', 'w')
    pickle.dump(tfidf[p_corpus], pfile)
    pfile.close()
    return tfidf[p_corpus]


def generate_tfidf_rs():
    corpus_filename = 'data/rs_debates.mm'
    p_corpus = corpora.MmCorpus(corpus_filename)
    tfidf = models.TfidfModel(p_corpus)
    pfile = open('data/rs_tfidf.pkl', 'w')
    pickle.dump(tfidf[p_corpus], pfile)
    pfile.close()
    return tfidf[p_corpus]


def generate_tfidf_ngram_ls(folder_name="data/ls_debates/"):
    session_enum = dict()
    session_docs = list()
    for file_name in os.listdir(folder_name):
        with open(folder_name + file_name, 'r', encoding="utf8") as txt_file:
            current_doc = txt_file.read().replace('\n', ' ')
            txt_file.close()
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
            session_docs[session_no] += ' ' + current_doc
            # session_docs[session_no].append(" ".join(current_doc))

    with open("data/ls_session_enum_tfidf.json", 'w') as json_file:
        json.dump(session_enum, json_file)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.1, stop_words='english',
                                       use_idf=True, ngram_range=(1, 5))

    tfidf_matrix = tfidf_vectorizer.fit_transform(session_docs)  # fit the vectorizer to synopses
    # pfile = open('data/ls_tfidf_ngram.pkl', 'w')
    # pickle.dump(tfidf_matrix, pfile)
    # pfile.close()
    print(tfidf_matrix.shape)
    return tfidf_matrix.A


def generate_tfidf_ngram_rs(folder_name="data/rs_debates/"):
    session_enum = dict()
    session_docs = list()
    for file_name in os.listdir(folder_name):
        with open(folder_name + file_name, 'r', encoding="utf8") as txt_file:
            current_doc = txt_file.read().replace('\n', ' ')
            txt_file.close()
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
        else:
            session_docs[session_no] += " " + current_doc

    with open("data/combined_session_enum_tfidf.json", 'w') as json_file:
        json.dump(session_enum, json_file)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.1, stop_words='english',
                                       use_idf=True, ngram_range=(1, 5))

    tfidf_matrix = tfidf_vectorizer.fit_transform(session_docs)  # fit the vectorizer to synopses
    print(tfidf_matrix.shape)
    return tfidf_matrix.A


def generate_tfidf_ngram_combined(folder_name="data/"):
    session_enum = dict()
    session_docs = list()
    for file_name in os.listdir(folder_name + "rs_debates/"):
        with open(folder_name + "rs_debates/" + file_name, 'r', encoding="utf8") as txt_file:
            current_doc = txt_file.read().replace('\n', ' ')
            txt_file.close()
        if len(current_doc) == 0:
            continue
        if file_name[1] == 'S':
            month = int(file_name[5:7])
            session_name = 'RS-20' + file_name[8:10]
        elif file_name[0] == 'S':
            month = int(file_name[4:6])
            session_name = 'RS-20' + file_name[7:9]
        else:
            month = int(file_name[3:5])
            session_name = 'RS-20' + file_name[6:8]
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
        else:
            session_docs[session_no] += " " + current_doc

    for file_name in os.listdir(folder_name + "ls_debates/"):
        with open(folder_name + "ls_debates/" + file_name, 'r', encoding="utf8") as txt_file:
            current_doc = txt_file.read().replace('\n', ' ')
            txt_file.close()
        if len(current_doc) == 0:
            continue
        month = int(file_name[3:5])
        session_name = "LS-" + file_name[6:10]
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
        else:
            session_docs[session_no] += ' ' + current_doc

    with open("data/combined_session_enum_tfidf.json", 'w') as json_file:
        json.dump(session_enum, json_file)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.1, stop_words='english',
                                       use_idf=True, ngram_range=(1, 5))

    tfidf_matrix = tfidf_vectorizer.fit_transform(session_docs)  # fit the vectorizer to synopses
    print(tfidf_matrix.shape)
    return tfidf_matrix.A


def generate_lsi_topics_ls():
    dictionary_filename = 'data/rs_debates.dict'
    corpus_filename = 'data/rs_debates.mm'
    p_dictionary = corpora.Dictionary.load(dictionary_filename)
    corpus_tfidf = generate_tfidf_ls()
    lsi = models.LsiModel(corpus_tfidf, id2word=p_dictionary, num_topics=500)
    corpus_lsi = lsi[corpus_tfidf]
    for x in lsi.show_topics(num_topics=10, num_words=20):
        print(x)

