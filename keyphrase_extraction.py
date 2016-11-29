import RAKE
import os
from nltk.corpus import words


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def extract_keyphrases_ls(folder_name="data/ls_debates/"):

    session_enum = dict()
    session_docs = list()
    dict_words = set(words.words())
    for file_name in os.listdir(folder_name):
        with open(folder_name+file_name, 'r', encoding="utf8") as txt_file:
            txt_data = txt_file.read().replace('\n', ' ')
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
            session_docs.append(txt_data)
        else:
            session_docs[session_no] += " " + txt_data

    custom_stopwords = ['affirmation', ' jo ', ' ki ', ' tha', 'ho', 'shri', 'shrimati', 'laid', 'madam', 'synopsis', 'oath', 'obituary', ' hai', ' hain', 'har', 'bhi ', 'aur ', ' ka ', 'guru', '______', '**  **']
    print("LOK SABHA\n=========\n")
    for session_name, session_no in sorted(session_enum.items(), reverse=True):
        session_text = session_docs[session_no]
        rake = RAKE.Rake("util/smart_stoplist.txt")
        print("=================\n"+session_name+"\n=================\n")
        session_keyphrases = rake.run(session_text)
        for phrase, score in session_keyphrases:
            for x in custom_stopwords:
                break_flag = 0
                in_dict = 0
                if x in phrase:
                    break_flag = 1
                    break
            if score > 5:
                for word in phrase.split(' '):
                    if word in dict_words:
                        in_dict = 1
                        break
                if break_flag == 1 or in_dict == 0 or has_numbers(phrase) or len(phrase) > 200 or len(phrase) < 6:
                    continue
                print(phrase, score)
        print("\n")


def extract_keyphrases_rs(folder_name="data/rs_debates/"):

    session_enum = dict()
    session_docs = list()
    dict_words = set(words.words())
    for file_name in os.listdir(folder_name):
        with open(folder_name+file_name, 'r', encoding="utf8") as txt_file:
            txt_data = txt_file.read().replace('\n', ' ')
        if file_name[1] == 'S':
            month = int(file_name[5:7])
            session_name = '20' + file_name[8:10]
        elif file_name[0] == 'S':
            month = int(file_name[4:6])
            session_name = '20' + file_name[7:9]
        else:
            month = int(file_name[3:5])
            session_name = '20'+file_name[6:8]

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
            session_docs.append(txt_data)
        else:
            session_docs[session_no] += " " + txt_data
    custom_stopwords = ['affirmation', ' jo ', ' ki ', ' tha', 'ho', 'shri', 'shrimati', 'laid', 'madam', 'synopsis', 'oath', 'obituary', ' hai', ' hain', 'har', 'bhi ', 'aur ', ' ka ', 'guru', '______', '**  **']
    print("RAJYA SABHA\n===========\n")
    for session_name, session_no in sorted(session_enum.items(), reverse=True):
        session_text = session_docs[session_no]
        rake = RAKE.Rake("util/smart_stoplist.txt")
        print("=================\n"+session_name+"\n=================\n")
        session_keyphrases = rake.run(session_text)
        for phrase, score in session_keyphrases:
            for x in custom_stopwords:
                break_flag = 0
                in_dict = 0
                if x in phrase:
                    break_flag = 1
                    break
            if score > 5:
                for word in phrase.split(' '):
                    if word in dict_words:
                        in_dict = 1
                        break
                if break_flag == 1 or in_dict == 0 or has_numbers(phrase) or len(phrase) > 200 or len(phrase) < 6:
                    continue
                print(phrase, score)
        print("\n")


extract_keyphrases_rs()