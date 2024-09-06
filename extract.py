"""
Context:
Client has SQL databases across USA and Canada, which contains addresses that are verified using a Google Maps API.
However many addresses are flagged as incorrect, because entries often contain typographical errors or are entered in an inconsistent format.
The database also gets regular updates of ~5,000 entries every month.
Client has previously solved this by manually correcting each entry, and instead wants an automated way of standardising data.

AIM 1:
    - Extract address data from 2 columns of messy inputs. Entries include addresses split over 2 rows, company names, and blank entries.

Extraction uses PyTorch and pre-trained transformer BERT for Named Entity Recognition (NER).
This is used to identify if an entry is not a company, and then concatenates the address with zipcode.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# load dataset
file_path = 'sample data.csv'
df = pd.read_csv(file_path)
df = df.astype(str)

# Load pre-trained BERT model for NER
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to check if the entry is a company name using deep learning
def is_company_name(entry):
    ner_results = ner_pipeline(entry)
    for entity in ner_results:
        if entity['entity_group'] == "ORG":
            return True
    return False

# Function to concatenate address parts with zip code
def concatenate_address(row):
    address_parts = []

    # Check Street column
    if not is_company_name(row['Street']):
        address_parts.append(row['Street'])

    # Check Street2 column
    if row['Street2'] and not is_company_name(row['Street2']):
        address_parts.append(row['Street2'])

    # If Street is a company name and Street2 is an address, only use Street2
    if is_company_name(row['Street']) and row['Street2'] and not is_company_name(row['Street2']):
        address_parts = [row['Street2']]

    # Combine the address parts and add Zipcode
    if address_parts:
        return ''.join(address_parts) + ', ' + row['ZipCode']
    else:
        return None

# Apply the function to each row and create a new column
df['FullAddress'] = df.apply(concatenate_address, axis=1)

# clean df
df = df.replace('nan', '', regex=True)
df = df.drop('Unnamed: 3', axis=1)

# Save the processed DataFrame to a new CSV file
output_file_path = 'extracted.csv'
df.to_csv(output_file_path, index=False)
print(f"Successfully saved to {output_file_path}")
