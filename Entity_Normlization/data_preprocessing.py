import io
from string import punctuation
import re


company_names = []

constituents = open('../vector/Named_Entity_Recognition/data/CompanyNames/constituents_csv.csv', 'r') 
constituents = constituents.readlines()
for c in constituents[1:5]:
    company_names.append(c.split(",")[1])

nasdaq_companies = open('../vector/Named_Entity_Recognition/data/CompanyNames/list-of-companies-in-nasdaq-exchanges-csv_csv.csv', 'r') 
nasdaq_companies = nasdaq_companies.readlines()
for nc in nasdaq_companies[1:5]:
    company_names.append(nc.split(",")[1].replace('"', ''))

telecom_companies = open('../vector/Named_Entity_Recognition/data/CompanyNames/telecom-operators_csv.csv', 'r') 
telecom_companies = telecom_companies.readlines()
for tc in telecom_companies[1:]:
    company_names.append(tc.split(",")[1].replace('"', ''))

uk_companies = io.open('../vector/Named_Entity_Recognition/data/CompanyNames/uk_comp.csv', 'r', encoding='windows-1252') 
uk_companies = uk_companies.readlines()
for uc in uk_companies:
    company_names.append(uc.lstrip(punctuation).replace(",", "").replace('"', '').replace("\n", ''))

company_names = list(dict.fromkeys(company_names)) 
company_name_label = ["ORG"] * len(company_names)

# Limited and LTD
LTD = []
LTD_set = []
for name in company_names:
  if "LIMITED" in name.upper() or "COMPANY" in name.upper() or "GROUP" in name.upper() or "SOCIETY" in name.upper():
    LTD.append(name.upper().strip())
    similar_name = name.upper().replace("LIMITED", "")
    similar_name = similar_name.upper().replace("COMPANY", "")
    similar_name = similar_name.upper().replace("GROUP", "")
    similar_name = similar_name.upper().replace("SOCIETY", "")
    similar_name = similar_name.strip()
    LTD.append(similar_name)
    LTD.append('y')
    LTD_set.append(LTD)
    LTD = []

  if "LIMITED" in name.upper():
    LTD.append(name.upper().strip())
    similar_name = re.sub("LIMITED", "LTD", name.upper())
    similar_name = similar_name.strip()
    LTD.append(similar_name)
    LTD.append('y')
    LTD_set.append(LTD)
    LTD = []

negative_company = []
negative_company_set = []

for i in range(0, len(company_names)-1):
    negative_company.append(company_names[i])
    negative_company.append(company_names[i+1])
    negative_company.append('n')
    negative_company_set.append(negative_company)
    negative_company = []

company_set = LTD_set + negative_company_set

person_name = open('/content/drive/My Drive/vector.ai/persons.match', 'r') 
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

print(similarity_dataset["label"].value_counts())
similarity_dataset.loc[similarity_dataset['label'].str.contains('y', na=False), 'label'] = 1
similarity_dataset.loc[similarity_dataset['label'].str.contains('n', na=False), 'label'] = 0

similarity_dataset.to_csv('/content/drive/My Drive/vector.ai/similarity_dataset.csv', index=False)