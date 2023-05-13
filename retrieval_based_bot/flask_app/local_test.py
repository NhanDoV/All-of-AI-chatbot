print("XBot has been stared to launch ..... \nImporting libraries")

import random, logging
import pickle, os, time
t0 = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
print(100*'#')
from transformers import pipeline
from tensorflow.keras.optimizers import SGD # from tensorflow.keras.optimizers.legacy import SGD # 
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from tensorflow.python.keras.layers import Dense, Dropout # from keras.layers import Dense, Dropout
from tensorflow.python.keras.models import load_model # from keras.models import load_model
from tensorflow.python.keras.models import Sequential # from keras.models import Sequential
import numpy as np
import json, warnings
import nltk, re
from nltk.stem import WordNetLemmatizer
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")
lemmatizer = WordNetLemmatizer()

current_fpath = os.getcwd()
print("current file-path", current_fpath)
data_folder = '/'.join(current_fpath.split('/')[:-1]) # r"/workspaces/AI_chatbot_flask"
print(f"file in this directory: {os.listdir(data_folder)}")

print("Loading model parameters")
model = load_model(f"{data_folder}/models/chatbot_model.h5")
gpt_model = pipeline("text-generation", model = "gpt2")
trans_model = pipeline("text2text-generation", model = "t5-small")

print("Loading word_vocalbularies parameters")
intents = json.loads(open(f"{data_folder}/json_data/intents.json", encoding='utf-8').read())
words = pickle.load(open(f"{data_folder}/models/words.pkl", "rb"))
classes = pickle.load(open(f"{data_folder}/models/classes.pkl", "rb"))
data_file = open(f"{data_folder}/json_data/intents.json", encoding='utf-8').read()
intents = json.loads(data_file)
print("Everything has been almostly completed")

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(ViTokenizer.tokenize(sentence))
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def rounding_text_hour(text):

    mask1 = any(word in text for word in ['tiếng', 'phút', 'giờ', 'minute', 'minutes'])
    mask2 = any(word in text for word in ['giờ làm việc', 'giờ hành chính', 'giờ đi làm', 'giờ vào làm', 'mấy giờ', "giờ ngủ trưa",
                                          'giờ có mặt', 'giờ tan sở', 'giờ ra về', 'giờ giải lao', 'giờ nghỉ trưa', "giờ nghỉ ngơi"])
    if mask1 & ~mask2:
        text = text.replace('giờ', 'tiếng').replace('hour', 'tiếng').replace('minute', 'phút').replace('mins', 'phút')
        h = re.findall('[0-9]+tiếng', text.replace(" ", ""))
        m = re.findall('[0-9]+phút', text.replace(" ", ""))
        if ((len(h) > 0) & (len(m) > 0)):
            h = h[0].replace("tiếng", "")
            m = m[0].replace("phút", "")
            if (int(m) < 0) or (int(m) > 60):
                txt = "số giờ (hoặc phút) làm việc không hợp lệ"
            elif int(m) < 60:
                txt = f"dưới {int(h) + 1} tiếng nhưng đã trên {int(h)} giờ làm việc"
            else:
                txt = f"dưới {int(h) + (m // 60) + 1} tiếng nhưng đã trên {int(h)} giờ làm việc"
        elif (len(h) == 0):
            txt = "dưới 1 tiếng làm việc"
        elif (len(m) == 0):
            txt = f"dưới {re.findall('[0-9]', h[0])[0]} giờ làm việc"
        else: 
            txt = text
        return txt    
    else: 
        return text
    
print(f"XBot has been launched after {int(time.time() - t0)} seconds, thank you for your patience")
print(f"{100*'*'}\n This is XBot, nice to meet you and please tell me something to talk")

from datetime import date, datetime, timedelta
import time

while True:
    print(100*"=")
    init_sentence = input("You: ")
    t1 = time.time()    
    sentence = rounding_text_hour(init_sentence)
    if sentence == "quit":
        break
    elif ('translate' in sentence.split()[:3]) or ('Germany' in sentence.split()) or ('France' in sentence.split()):        
        print("BOT:")
        print(trans_model(sentence)['generated_text'])
    else:        
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.6
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        ints = return_list
        print(ints)

        if len(return_list) > 0:
            tag = ints[0]["intent"]
            list_of_intents = intents["intents"]

            for i in list_of_intents:
                if i["tag"] == tag:
                    print(f"Tag: {tag} \n {100*'.'}")
                    result = random.choice(i["response"])
                    print(f"BOT: Hi\n{result}")
                    if i["tag"] == "daytime_today":
                        today_date = (datetime.now()).strftime('%b-%d, %Y \t %H:%M:%S GMT+07')
                        print(f"\tToday is {today_date }")
        else: 
            res_sentence = gpt_model(
                                    init_sentence, 
                                    do_sample=True, top_k=3, 
                                    temperature=0.9, max_length=120,
                                    pad_token_id=gpt_model.tokenizer.eos_token_id
                                    )
            res_sentence = res_sentence[0]["generated_text"]
            final_responses = res_sentence.replace('\n\n','\n').split('\n')[1:] 
            if len(final_responses) > 0:
                print(f"[{int(time.time() - t1)} secs] {np.random.choice(final_responses)} \nNếu bạn chưa hài lòng với câu trả lời, xin giải thích rõ ràng hơn vì có thể bot không hiểu hoặc chưa được học!")
            else:
                print(f"[{int(time.time() - t1)} secs] Please use English instead, sorry for this inconvenience!!")
