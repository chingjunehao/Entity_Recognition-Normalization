# https://www.kaggle.com/PromptCloudHQ/flipkart-products products name from Flipkart.com, a leading Indian eCommerce store.
# https://www.kaggle.com/carrie1/ecommerce-data/home products name from UK retailer

fk_data = pd.read_csv('data/flipkart_com-ecommerce_sample.csv')
uk_data = pd.read_csv('data/uk_retailer_data.csv')

fk_data = fk_data.drop_duplicates(subset=["product_name"], keep=False)
uk_data = uk_data.drop_duplicates(subset=["Description"], keep=False)

fk_data = fk_data[fk_data["product_name"].notna()]
uk_data = uk_data[uk_data["Description"].notna()]

fk_data = fk_data["product_name"].tolist()
uk_data = uk_data["Description"].tolist()

physical_goods = fk_data + uk_data

df = pd.read_csv('data/ner_dataset.csv', encoding = "ISO-8859-1")

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence_count = 47960 # count from the ner_dataset
sentence = []
words = []
pos = []
tags = []

for goods in physical_goods:
  sentence.append("Sentence: " + str(sentence_count))
  goods_tokens = nltk.word_tokenize(goods)
  for word, predicted_pos in nltk.pos_tag(goods_tokens):
    words.append(word)
    pos.append(predicted_pos)
    if predicted_pos == "NN" or predicted_pos == "NNS":
      tags.append("B-pg")
    else:
      tags.append("O")
    sentence.append(None)
  sentence = sentence[:-1]
  sentence_count += 1

generated_df = pd.DataFrame(
    {'Sentence #': sentence,
     'Word': words,
     'POS': pos,
     'Tag' : tags
    }
)

sentence = []
words = []
pos = []
tags = []

for i in range(len(serial_num)):
  sentence_count += 1
  sentence.append("Sentence: " + str(sentence_count))
  
  words.append(serial_num[i])
  pos.append(".")
  tags.append("B-sn")

serial_num_df = pd.DataFrame(
    {'Sentence #': sentence,
     'Word': words,
     'POS': pos,
     'Tag' : tags
    }
)

merged_df = pd.concat([df, generated_df, serial_num_df], ignore_index=True)

# Thinking of randomly drop the data that is too much, but it might take away the temporal features of the model
# Have to think about the other way
# import numpy as np
# n = 1000
# to_remove = np.random.choice(merged_df[merged_df['Tag']=="O"].index,size=n,replace=False)
# merged_df = merged_df.drop(to_remove)
merged_df.to_csv("training_set.csv", index=False)