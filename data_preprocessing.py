import pandas as pd
import random
import string

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import io
from string import punctuation

import numpy as np
# Entity: Serial Number ###############################################################################
# Generating synthetic airway bill number 
# Ticketing code https://en.wikipedia.org/wiki/Air_waybill#cite_note-5 no.5 on reference
# Formatting https://en.wikipedia.org/wiki/Air_waybill

# http://parcelsapp.com/en/awb-air-cargo Might contain "-" or "/"


df = pd.read_fwf('data/SerialNumber/airline-codes.txt')
df = df[df['Ticketing'].notna()]
df = df.iloc[2:-1]
ticketing_codes = df['Ticketing'].tolist()


serial_num = []
for code in ticketing_codes:
  for i in range(4):
    awb_serial_num = random.randint(4637289,9374616)
    check_digit = awb_serial_num % 7

    if i < 2:
      awb_num = code + str(awb_serial_num) + str(check_digit)
    elif i < 3:
      awb_num = code + "-" + str(awb_serial_num) + str(check_digit)
    else:
      awb_num = code + "/" + str(awb_serial_num) + str(check_digit)

    serial_num.append(awb_num)

# Generating synthetic serial number of bill of lading
# https://en.wikipedia.org/wiki/Bill_of_lading
# Range of number of alphabet (4-10) followed by 6-10 numbers and end with 0 or more alphabet 

def random_char(num):
  return ''.join(random.choice(string.ascii_letters) for x in range(num))

# Example of synthetic serial number will be generated
# 1. (4-6 alphabets)(6-10 digits)
# 2. (4-6 alphabets)(1-6 alphabets)(6-10 digits)
# 3. (4-6 alphabets)(6-10 digits)(1-7 alphabets)
# 4. (4-6 alphabets)/(6-10 digits)
# 5. (4-6 alphabets)/(6-10 digits)(1-7 alphabets)
# 6. (4-6 alphabets)-(6-10 digits)(1-7 alphabets)
# 7. (4-6 alphabets)(6-10 digits)/(1-7 alphabets)
# 8. (4-6 alphabets)-(6-10 digits)-(1-7 alphabets)

bol_serial_num = ""
random.seed(1)
for i in range(28000):
  number_of_alph = random.randint(4, 6)
  alphabet = random_char(number_of_alph).upper()

  middle_num = random.randint(837463, 3029382736)

  if i < 6000:
    bol_serial_num = alphabet + str(middle_num)
  elif i < 12000:
    number_of_alph = random.randint(1, 6)
    middle_alphabet = random_char(number_of_alph).upper()
    bol_serial_num = alphabet + middle_alphabet + str(middle_num)
  elif i < 14000:
    number_of_alph = random.randint(1, 7)
    ending_alphabet = random_char(number_of_alph).upper()
    bol_serial_num = alphabet + str(middle_num) + ending_alphabet
  else:

    number_of_alph = random.randint(1, 7)
    ending_alphabet = random_char(number_of_alph).upper()

    random_prob = random.random()
    if random_prob < 0.2:
      bol_serial_num = alphabet + "/" + str(middle_num)
    elif random_prob < 0.4:
      bol_serial_num = alphabet + "/" + str(middle_num)  + ending_alphabet
    elif random_prob < 0.6:
      bol_serial_num = alphabet + "-" + str(middle_num)  + ending_alphabet
    elif random_prob < 0.8:
      bol_serial_num = alphabet + str(middle_num)  +  "/" + ending_alphabet
    else:
      bol_serial_num = alphabet + "-" + str(middle_num)  +  "-" + ending_alphabet

  serial_num.append(bol_serial_num)


# Entity: Physical Goods ###############################################################################

# https://www.kaggle.com/PromptCloudHQ/flipkart-products products name from Flipkart.com, a leading Indian eCommerce store.
# https://www.kaggle.com/carrie1/ecommerce-data/home products name from UK retailer

fk_data = pd.read_csv('data/PhysicalGoods/flipkart_com-ecommerce_sample.csv')
uk_data = pd.read_csv('data/PhysicalGoods/uk_retailer_data.csv')

fk_data = fk_data.drop_duplicates(subset=["product_name"], keep=False)
uk_data = uk_data.drop_duplicates(subset=["Description"], keep=False)

fk_data = fk_data[fk_data["product_name"].notna()]
uk_data = uk_data[uk_data["Description"].notna()]

fk_data = fk_data["product_name"].tolist()
uk_data = uk_data["Description"].tolist()

physical_goods = fk_data + uk_data

pg_words = []
pg_tags = []

# Since physical goods most of the time will be noun, so it is extracted using NLTK part-of-speech
for goods in physical_goods:
  goods_tokens = nltk.word_tokenize(goods)
  for word, predicted_pos in nltk.pos_tag(goods_tokens):
    if predicted_pos == "NN" or predicted_pos == "NNS":
      pg_words.append(word)
      pg_tags.append("B-PG")

# Entity: Location, Company name, Physical Goods ###############################################################################
# CoNLL-2003 dataset: https://github.com/itaigat/pycharner 

conll_train = open('data/conll_train.txt', 'r') 
conll_valid = open('data/conll_valid.txt', 'r') 
conll_test = open('data/conll_test.txt', 'r') 

train = conll_train.readlines() 
valid = conll_valid.readlines() 
test = conll_test.readlines() 

data = {
    "text": [],
    "label": []
}

# Extract useful entities from CoNLL-2003 dataset
# If it has entity (not "O"), then we just extract
# If it doesn't has entity, then we check if it is noun, if yes then we extract as physical goods
for line in train[2:]:
  if bool(line.strip()):
    if line.split(" ")[-1] != "O\n":
      data["text"].append(line.split(" ")[0])
      data["label"].append(line.split(" ")[-1][:-1])
    else:
      if (line.split(" ")[1] == "NN" or line.split(" ")[1] == "NNS"):
        data["text"].append(line.split(" ")[0])
        data["label"].append("B-PG")

for line in valid[2:]:
  if bool(line.strip()):
    if line.split(" ")[-1] != "O\n":
      data["text"].append(line.split(" ")[0])
      data["label"].append(line.split(" ")[-1][:-1])
    else:
      if (line.split(" ")[1] == "NN" or line.split(" ")[1] == "NNS"):
        data["text"].append(line.split(" ")[0])
        data["label"].append("B-PG")

for line in test[2:]:
  if bool(line.strip()):
    if line.split(" ")[-1] != "O\n":
      data["text"].append(line.split(" ")[0])
      data["label"].append(line.split(" ")[-1][:-1])
    else:
      if (line.split(" ")[1] == "NN" or line.split(" ")[1] == "NNS"):
        data["text"].append(line.split(" ")[0])
        data["label"].append("B-PG")

for sn in serial_num:
  data["text"].append(sn)
  data["label"].append("B-SN")

# Seems like adding extra noise and downgrade the performance of the model, so not included
# data["text"].extend(pg_words)
# data["label"].extend(pg_tags)
# print(len(data["text"]), len(data["label"])) 


# Entity: Location ###############################################################################
# Countries and major cities around the world https://datahub.io/core/world-cities#resource-world-cities_zip
country_city = open('data/Location/world-cities.csv', 'r', encoding='utf-8') 
country_city = country_city.readlines()

country_city_array = []
for cc in country_city[1:]:
  country_city_array.append(cc.split(",")[0])
  country_city_array.append(cc.split(",")[1])
  country_city_array.append(cc.split(",")[2])
country_city_array = list(dict.fromkeys(country_city_array)) # remove duplicate

country_label = ["LOC"] * len(country_city_array)

# Entity: Company names ###############################################################################
# UK companies 2018: https://data.world/makeovermonday/2018w23-uk-gender-pay-gap/workspace/file?filename=Gender+Pay+Gap.xlsx
# Telecom operator companies: https://datahub.io/ismail.shahzad/telecom-operators-of-the-world
# S&P 500: https://datahub.io/core/s-and-p-500-companies
# Companies of NASDAQ: https://datahub.io/JohnSnowLabs/list-of-companies-in-nasdaq-exchanges

company_names = []

constituents = open('data/CompanyNames/constituents.csv', 'r') 
constituents = constituents.readlines()
for c in constituents[1:5]:
  company_names.append(c.split(",")[1])

nasdaq_companies = open('data/CompanyNames/list-of-companies-in-nasdaq-exchanges.csv', 'r') 
nasdaq_companies = nasdaq_companies.readlines()
for nc in nasdaq_companies[1:5]:
  company_names.append(nc.split(",")[1].replace('"', ''))

telecom_companies = open('data/CompanyNames/telecom-operators.csv', 'r') 
telecom_companies = telecom_companies.readlines()
for tc in telecom_companies[1:]:
  company_names.append(tc.split(",")[1].replace('"', ''))

uk_companies = io.open('data/CompanyNames/uk_comp.csv', 'r', encoding='windows-1252') 
uk_companies = uk_companies.readlines()
for uc in uk_companies:
  company_names.append(uc.lstrip(punctuation).replace(",", "").replace('"', '').replace("\n", ''))

company_names = list(dict.fromkeys(company_names)) 
company_name_label = ["ORG"] * len(company_names)

data["text"].extend(country_city_array)
data["text"].extend(company_names)

data["label"].extend(country_label)
data["label"].extend(company_name_label)
print(len(data["text"]), len(data["label"]))

full_dataset =pd.DataFrame(
    {'text': data["text"],
     'label': data["label"]
    })


full_dataset.replace('', np.nan, inplace=True)
full_dataset.dropna(subset = ["text", "label"], inplace=True)

full_dataset = full_dataset.apply(lambda x: x.astype(str).str.upper())

# Entities that we need for the project
full_dataset.loc[full_dataset['label'].str.contains('ORG', na=False), 'label'] = 0
full_dataset.loc[full_dataset['label'].str.contains('LOC', na=False), 'label'] = 1
full_dataset.loc[full_dataset['label'].str.contains('PER', na=False), 'label'] = 2
full_dataset.loc[full_dataset['label'].str.contains('PG', na=False), 'label'] = 3
full_dataset.loc[full_dataset['label'].str.contains('SN', na=False), 'label'] = 4

# Will be removed later, since only add noise to the model
full_dataset.loc[full_dataset['label'].str.contains('MISC', na=False), 'label'] = 5 

full_dataset = full_dataset[full_dataset.label != 5]

print(full_dataset["label"].value_counts())

# I put as full-dataset.csv
full_dataset.to_csv('data/your-dataset-name.csv', index=False) 