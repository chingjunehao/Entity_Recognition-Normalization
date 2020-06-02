import pandas as pd
import random
import string

import io
import string
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

def random_char(y):
    return ''.join(random.choice(string.ascii_letters) for x in range(y))

def sn_generator():
    number_of_alph = random.randint(3, 6)
    alphabet = random_char(number_of_alph).upper()

    middle_num = random.randint(837463, 3029382736)
    
    number_of_alph = random.randint(1, 6)
    middle_alphabet = random_char(number_of_alph).upper()

    number_of_alph = random.randint(1, 7)
    ending_alphabet = random_char(number_of_alph).upper()

    return alphabet, middle_num, middle_alphabet, ending_alphabet

for i in range(28000):
    a, mn, ma, ea = sn_generator()
    if i < 6000:
        bol_serial_num = a + str(mn)
    elif i < 12000:
        bol_serial_num = a + ma + str(mn)
    elif i < 14000:
        bol_serial_num = a + str(mn) + ea
    else:
        random_prob = random.random()
        if random_prob < 0.2:
            bol_serial_num = a + "/" + str(mn)
        elif random_prob < 0.4:
            bol_serial_num = a + "/" + str(mn)    + ea
        elif random_prob < 0.6:
            bol_serial_num = a + "-" + str(mn)    + ea
        elif random_prob < 0.8:
            bol_serial_num = a + str(mn)    +    "/" + ea
        else:
            bol_serial_num = a + "-" + str(mn)    +    "-" + ea

    serial_num.append(bol_serial_num)


# Entity: Location, Company name ###############################################################################
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
for line in train[2:]:
    if bool(line.strip()):
        if line.split(" ")[-1] != "O\n":
            data["text"].append(line.split(" ")[0])
            data["label"].append(line.split(" ")[-1][:-1])

for line in valid[2:]:
    if bool(line.strip()):
        if line.split(" ")[-1] != "O\n":
            data["text"].append(line.split(" ")[0])
            data["label"].append(line.split(" ")[-1][:-1])

for line in test[2:]:
    if bool(line.strip()):
        if line.split(" ")[-1] != "O\n":
            data["text"].append(line.split(" ")[0])
            data["label"].append(line.split(" ")[-1][:-1])

for sn in serial_num:
    data["text"].append(sn)
    data["label"].append("B-SN")

# Entity: Location ###############################################################################
# Countries and major cities around the world https://datahub.io/core/world-cities#resource-world-cities_zip
country_city = open('data/Location/world-cities.csv', 'r', encoding='utf-8') 
country_city = country_city.readlines()

countries = []
country_city_array = []
for cc in country_city[1:]:
    country_city_array.append(cc.split(",")[0])
    country_city_array.append(cc.split(",")[1])
    country_city_array.append(cc.split(",")[2])
    countries.append(cc.split(",")[1])

countries = list(dict.fromkeys(country_city_array))   
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
for c in constituents[1:]:
    company_names.append(c.split(",")[1])

nasdaq_companies = open('data/CompanyNames/list-of-companies-in-nasdaq-exchanges.csv', 'r') 
nasdaq_companies = nasdaq_companies.readlines()
for index, nc in enumerate(nasdaq_companies[1:]):
    company_names.append(nc.split(",")[1].replace('"', ''))
    # For company name + location
    company_names.append(nc.split(",")[1].replace('"', '') + " " + countries[index%len(countries)]) 

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
data["label"].extend(country_label)

data["text"].extend(company_names)
data["label"].extend(company_name_label)

print(len(data["text"]), len(data["label"]))

ner_dataset =pd.DataFrame(
        {'text': data["text"],
         'label': data["label"]
        })


ner_dataset.replace('', np.nan, inplace=True)
ner_dataset.dropna(subset = ["text", "label"], inplace=True)

ner_dataset = ner_dataset.apply(lambda x: x.astype(str).str.upper())

# Entities that we need for the project
ner_dataset.loc[ner_dataset['label'].str.contains('ORG', na=False), 'label'] = 0
ner_dataset.loc[ner_dataset['label'].str.contains('LOC', na=False), 'label'] = 1
ner_dataset.loc[ner_dataset['label'].str.contains('SN', na=False), 'label'] = 2
ner_dataset.loc[ner_dataset['label'].str.contains('PER', na=False), 'label'] = 3

# Will remove, since they only add noise to the model
ner_dataset.loc[ner_dataset['label'].str.contains('MISC', na=False), 'label'] = 4
ner_dataset = ner_dataset[ner_dataset.label != 4]
print(ner_dataset["label"].value_counts())
print(len(ner_dataset))
ner_dataset.to_csv('data/ner_dataset.csv', index=False) 

# construct dataset for serial number binary classifier
sn_dataset = pd.read_csv("data/ner_dataset.csv")
sn_dataset.loc[sn_dataset['label'] != 2, 'label'] = 0
sn_dataset.loc[sn_dataset['label'] == 2, 'label'] = 1
sn_dataset['text'] = sn_dataset['text'].str.replace(" ","")
sn_dataset.drop_duplicates(keep=False,inplace=True) 
sn_dataset.to_csv('data/sn_dataset.csv', index=False)


# Preprocess again for the 3 classes classification ##########################
ner_dataset = pd.read_csv("data/ner_dataset.csv")
ner_dataset = ner_dataset[ner_dataset.label != 3]
ner_dataset = ner_dataset[ner_dataset.label != 2]

ner_text = ner_dataset["text"].tolist()
ner_label = ner_dataset["label"].tolist()

goods1 = pd.read_csv("data/PhysicalGoods/Purchase_Order_Quantity_Price_detail_for_Commodity_Goods_procurements.csv")
goods1 = goods1["COMMODITY_DESCRIPTION"].tolist()

goods2 = pd.read_csv("data/PhysicalGoods/purchase-order-quantity-price-detail-for-commodity-goods-procurements-1.csv")
goods2 = goods2["COMMODITY_DESCRIPTION"].tolist()

total_goods = []

for g in goods1:
    if "," in g:
        total_goods.extend(g.split(","))
    else:
        total_goods.append(g)

for g in goods2:
    if "," in g:
        total_goods.extend(g.split(","))
    else:
        total_goods.append(g)

total_goods = list(dict.fromkeys(total_goods))
goods_label = [2] *len(total_goods)

country_city = open('data/world-cities.csv', 'r') 
country_city = country_city.readlines()

address = ""
address2 = ""
country_city_array = []
country = []
for cc in country_city[1:]:
    address = str(cc.split(",")[0]) + ", " + str(cc.split(",")[1])
    address2 = str(cc.split(",")[0]) + " " + str(cc.split(",")[1])
    country_city_array.append(address)
    country_city_array.append(address2)
    country.append(str(cc.split(",")[1]))

country = list(dict.fromkeys(country)) # remove duplicate
country_city_array = list(dict.fromkeys(country_city_array)) # remove duplicate
country_label = [1] * len(country_city_array)

ner_text.extend(total_goods)
ner_label.extend(goods_label)

ner_text.extend(country_city_array)
ner_label.extend(country_label)

ner3_dataset =pd.DataFrame(
    {'text': ner_text,
     'label': ner_label
    })
ner3_dataset.to_csv('data/ner3_dataset.csv', index=False)

