# All-of-AI-chatbot
Rule-based / Retrieval based &amp; AI Chatbot
Trong phần này, tôi sẽ đề cập đến 3 kiểu chatbot cơ bản là
- [ ] Rule-based
- [ ] Retrieval-based
- [ ] AI Conversation (GPT for advanced)

# A. Motivation

# B. Chatbot-types

## 1. Rule-based
Rule-based chatbots are structured as a dialog tree and often use regular expressions to match a user’s input to human-like responses. 

The aim is to simulate the back-and-forth of a real-life conversation, often in a specific context, like 
- telling the user what the weather is like outside. 
- answering the users's question about the service

In chatbot design, rule-based chatbots are **closed-domain**, also called dialog agents, because they are limited to conversations on a specific subject.

![image](https://user-images.githubusercontent.com/60571509/233780873-939e104f-c40b-4617-b01a-fd6f05dbee9a.png)

### 1.1. Chatbot Intents
In chatbots design, an `intent` is the purpose or category of the user query. The user’s utterance gets matched to a chatbot intent. In rule-based chatbots, you can use regular expressions to match a user’s statement to a chatbot intent.

### 1.2. Chatbot Utterances
An `utterance` is a statement that the user makes to the chatbot. The chatbot attempts to match the utterance to an intent.

### 1.3. Chatbot Entities
An `entity` is a value that is parsed from a user utterance and passed for use within the user response.

## 2. Retrieval-based
Retrieval-based chatbots are used in closed-domain scenarios and rely on a collection of predefined responses to a user message. A retrieval-based bot completes three main tasks: intent classification, entity recognition, and response selection.

### 2.1. Intent Similarity for Retrieval-Based Chatbots
For retrieval-based chatbots, it is common to use bag-of-words or tf-idf to compute intent similarity.

### 2.2. Entity Recognition for Retrieval-Based Chatbots
For retrieval-based chatbots, entity recognition can be accomplished using part-of-speech (POS) tagging or word embeddings such as word2vec.

## 3. AI Chatbot


# Summary
