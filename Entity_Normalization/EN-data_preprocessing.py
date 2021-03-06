import io
from string import punctuation
import re


company_names = []

constituents = open('data/CompanyNames/constituents.csv', 'r') 
constituents = constituents.readlines()
for c in constituents[1:]:
    company_names.append(c.split(",")[1])

nasdaq_companies = open('data/CompanyNames/list-of-companies-in-nasdaq-exchanges.csv', 'r') 
nasdaq_companies = nasdaq_companies.readlines()
for nc in nasdaq_companies[1:]:
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

# Limited and LTD
entity_type = []
et_set = []
similar_name = ""
for name in company_names:
    name = name.upper()
    if "LIMITED" in name or "COMPANY" in name or "GROUP" in name or "SOCIETY" in name or "INC" in name or "CO." in name:
        entity_type.append(name.upper().strip())
        similar_name = name.upper().strip()
        similar_name = similar_name.replace("LIMITED", "")
        similar_name = similar_name.replace("COMPANY", "")
        similar_name = similar_name.replace("GROUP", "")
        similar_name = similar_name.replace("SOCIETY", "")
        similar_name = similar_name.replace("INC", "")
        similar_name = similar_name.replace("CO.", "")
        entity_type.append(similar_name)
        entity_type.append('y')
        et_set.append(entity_type)
        entity_type = []

    
    if "LIMITED" in name:
        entity_type.append(name.upper().strip())
        similar_name = name.upper().strip()
        similar_name = re.sub("LIMITED", "LTD", similar_name)
        entity_type.append(similar_name)
        entity_type.append('y')
        et_set.append(entity_type)
        entity_type = []

    
    if "&" in name:
        similar_name = name.upper().strip()
        if " & " in name:
            entity_type.append(name.upper().strip())
            similar_name = re.sub(" & ", " AND ", similar_name)
            entity_type.append(similar_name)
            entity_type.append('y')
            et_set.append(entity_type)
            entity_type = []
        else:
            entity_type.append(name.upper().strip())
            similar_name = re.sub("&", " AND ", similar_name)
            entity_type.append(similar_name)
            entity_type.append('y')
            et_set.append(entity_type)
            entity_type = []


negative_company = []
negative_company_set = []

for i in range(0, len(company_names)-1):
    negative_company.append(company_names[i])
    negative_company.append(company_names[i+1])
    negative_company.append('n')
    negative_company_set.append(negative_company)
    negative_company = []

company_set = et_set + negative_company_set

person_name = open('data/NameSimilarity/persons.match', 'r', encoding='utf-8') 
person_name = person_name.readlines()
person_name_y = []
for pn in person_name:
    pn = pn.split("\t")
    pn[-1] = pn[-1][:-1]
    if pn[-1] == "y":
        person_name_y.append(pn)

for i in range(len(person_name_y)):
    for j in range(len(person_name_y[i])):
        person_name_y[i][j] = re.sub('\d', '', person_name_y[i][j])
        person_name_y[i][j] = person_name_y[i][j].replace("%", "")
        person_name_y[i][j] = person_name_y[i][j].replace("disambiguation", "")
        person_name_y[i][j] = person_name_y[i][j].strip()

negative_sample = []
negative_sample_set = []
for i in range(0, len(person_name_y)-1):
    negative_sample.append(person_name_y[i][0])
    negative_sample.append(person_name_y[i+1][0])
    negative_sample.append('n')
    negative_sample_set.append(negative_sample)
    negative_sample = []

person_name_set = person_name_y + negative_sample_set

sim_dataset = company_set + person_name_set

import pandas as pd
sample = []
sample_set = []
label = []

for data in sim_dataset:
  sample.append(data[0])
  sample.append(data[1])
  sample_set.append(sample)
  label.append(data[2])
  sample = []

similarity_dataset =pd.DataFrame(
    {'Name': sample_set,
     'label': label
    })

similarity_dataset.loc[similarity_dataset['label'].str.contains('y', na=False), 'label'] = 0
similarity_dataset.loc[similarity_dataset['label'].str.contains('n', na=False), 'label'] = 1

similarity_dataset.to_csv('data/en_similarity_dataset.csv', index=False)