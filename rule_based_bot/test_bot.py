import os, json
from nltk.chat.util import Chat
import datetime as dt

today = dt.datetime.now()
today = today.date().strftime("%b-%d, %Y \t %H:%M:%S")

contents = [
    [
        "when (.*) (class|first|second|finish) ?(.*)",
        ["You can look at your schedule at ...(insert link)..."]
    ],
    [
        "(.*)(who|mission) (.*)?",
        ["I am a rule-based chatbot generated by Sir Do Van Nhan"]
    ],
    [
        "today (date|what|time|is)(.*)",
        [f"today is {today}"]
    ],
    [
        "(.*)I (need help|have a question)(.*)",
        ["Can I help you something",]
    ],
    [
        "(.*)(danke|merci|thanks)(.*)",
        ["you are welcome", "bienvenue", "wie geht's"]
    ],
     [
        "(.*) your name ?",
        ["My name is DataScientist_NhanDV, but you can just call me robot and I'm a chatbot ."]
    ],
    [
        "(.*)how are you?",
        ["I'm doing very well, and you", "i am great ! and you",]
    ],
    [
        "sorry (.*)",
        ["Its alright","Its OK, never mind that",]
    ],
    [
        "i'm (.*) (good|well|okay|ok)",
        ["Nice to hear that","Alright, great !",]
    ],
    [
        "(hi|hey|hello|hola|holla|bonjour|bonsoir|guten tag|konichiwa)(.*)",
        ["Hello", "Hey there","need help?"]
    ],
    [
        "what (.*) want ?",
        ["Make me an offer I can't refuse",]
        
    ],
    [
        "(.*)created(.*)",
        ["Sr Do Van Nhan had created me by using Python's NLTK library ", "top secret ;)",]
    ],
    [
        "(.*) (location|city|located) ?",
        ['Sai Gon City, Vietnam, Asia, UTC+7',]
    ],
    [
        "(.*) address (.*) ?",
        ["You mean the current address? or location?"]
    ],
    [
        "how (.*) health (.*)",
        ["Health is very important, but I am a computer, so I don't need to worry about my health ",]
    ],
    [
        "(.*)(sports|football|soccer-team|favorite-team)(.*)",
        ["May be MU - MC - Liv? and you?",]
    ],
    [
        "quit",
        ["Bye for now. See you soon :) ","It was nice talking to you. See you soon :)"]
    ],
    [
        "(.*)",
        ['That is nice to hear but I can not reply']
    ],
]

chat = Chat(contents)
chat.converse()