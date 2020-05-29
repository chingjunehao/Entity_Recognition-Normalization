# Generating synthetic airway bill number 
# Ticketing code https://en.wikipedia.org/wiki/Air_waybill#cite_note-5 no.5 on reference
# Formatting https://en.wikipedia.org/wiki/Air_waybill

# http://parcelsapp.com/en/awb-air-cargo Might contain "-" or "/"

import pandas as pd
import random
import string

df = pd.read_fwf('data/airline-codes.txt')
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

def random_char(y):
  return ''.join(random.choice(string.ascii_letters) for x in range(y))

bol_serial_num = ""
random.seed(1)
for i in range(12000):
  number_of_alph = random.randint(4, 6)
  alphabet = random_char(number_of_alph).upper()

  middle_num = random.randint(837463, 3029382736)

  if i < 4000:
    bol_serial_num = alphabet + str(middle_num)
  elif i < 6000:
    number_of_alph = random.randint(1, 6)
    middle_alphabet = random_char(number_of_alph).upper()
    bol_serial_num = alphabet + middle_alphabet + str(middle_num)
  elif i < 8000:
    number_of_alph = random.randint(1, 7)
    ending_alphabet = random_char(number_of_alph).upper()
    bol_serial_num = alphabet + str(middle_num) + ending_alphabet
  else:
    number_of_alph = random.randint(1, 6)
    middle_alphabet = random_char(number_of_alph).upper()

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
      bol_serial_num = alphabet + "/" + str(middle_num)  +  "/" + ending_alphabet
    else:
      bol_serial_num = alphabet + "-" + str(middle_num)  +  "-" + ending_alphabet

  serial_num.append(bol_serial_num)