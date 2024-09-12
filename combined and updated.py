import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
import re
import time

# Dictionary of common suffixes
suffix_mapping = {
    'st': 'Street', 'str': 'Street',
    'ave': 'Avenue',
    'blvd': 'Boulevard',
    'rd': 'Road',
    'ln': 'Lane',
    'dr': 'Drive',
    'ct': 'Court',
    'pkwy': 'Parkway',
    'hwy': 'Highway',
    'wy': 'Way',
    'bldg': 'Building',
    'ste': 'Suite',
    'fwy': 'Freeway',
    'pl': 'Place',
    'inc': 'Incorporated',
    'ne': 'North East', 'sw': 'South West', 'se': 'South East', 'nw': 'North West', 'e': 'East', 'n': 'North', 's': 'South', 'w': 'West',
   # Add more abbreviations as needed
}

start_time = time.time()


# (I have been told that input access method will be changed, so for now have loaded a local test dataset csv file)
# Load dataset
file_path = 'big sample.csv'
df = pd.read_csv(file_path)
df = df.astype(str)

# for easier use
COLUMN1 = 'Street'
COLUMN2 = 'Street2'
ZIPCODE = 'ZipCode'

# Load pre-trained BERT model for NER
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# Initialize NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# func for batch processing, can edit batch size
def batch_process_ner(entries, ner_pipeline, batch_size=32):
    ner_results = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        ner_results.extend(ner_pipeline(batch))
    return ner_results

# func for company check
def is_company_name(entries):
    ner_results = batch_process_ner(entries, ner_pipeline)
    return ['ORG' in [entity['entity_group'] for entity in result] for result in ner_results]


# func for cleaning up commas
def remove_double_commas(address):
    address_cleaned = re.sub(r',\s*,', ',', address)    # Remove ', ,'
    address_cleaned = re.sub(r'\s*,\s*', ', ', address_cleaned)  # Ensure one space after commas
    return address_cleaned.strip(', ')  # Remove leading/trailing commas or spaces

# func for adding zipcode at the end
def concatenate_zipcode(address_column, zipcode_column):
    concatenated_address = address_column + ', ' + zipcode_column
    return concatenated_address.apply(remove_double_commas)

# address extraction function
def concatenate_address(row):
    address_parts = []
    # Check Street column
    if not row['Is_Company_Street1']:
        address_parts.append(row[COLUMN1])
    # Check Street2 column
    if row[COLUMN2] and not row['Is_Company_Street2']:
        address_parts.append(row[COLUMN2])

    # If Street is a company name and Street2 is an address, only use Street2
    if row['Is_Company_Street1'] and row[COLUMN2] and not row['Is_Company_Street2']:
        address_parts = [row[COLUMN2]]

    if address_parts:
        return ', '.join(address_parts)
    else:
        return None

suffix_regex = re.compile(r'\d+(th|rd|st|nd)', re.IGNORECASE) # pre-compiling regex pattern (improve speed)

# Func for titlecase converter
def custom_titlecase(street_name):
    def preserve_suffixes(word):
        # Preserve 'th', 'rd', 'st', 'nd'
        if suffix_regex.match(word):
            return word.lower()
        return word.title()
    # Split the street name into words, apply preserve_suffixes, and join them back
    return ' '.join([preserve_suffixes(word) for word in street_name.split()])


# func for removing NaNs before passing through polyfuzz
def fill_empty_entries(df):
    df[COLUMN1].fillna('Empty', inplace=True)
    df[COLUMN2].fillna('Empty', inplace=True)
    df[COLUMN1] = df[COLUMN1].apply(lambda x: 'Empty' if pd.isnull(x) or x.strip() == '' else str(x))
    df[ZIPCODE] = df[ZIPCODE].apply(lambda x: 'Empty' if pd.isnull(x) or x.strip() == '' else str(x))
    return df


# func for suffix abbreviations (case insensitive)
def replace_suffixes(street_name):
    if isinstance(street_name, str):
        words = street_name.split()
        replaced_words = []
        for word in words:
            # Convert word to lowercase for lookup, strip punctuation for comparison
            lower_word = re.sub(r'[^\w]', '', word.lower())
            """
            I have included an if statement removing info after the word 'bldg', to avoid unnecessary info EG. floor, flat number. This can be removed if the extra info doesn't affect the address verification
            """
            if lower_word == 'bldg':
                replaced_words.append('Building')
                break
            # Replace suffix if found in dict
            elif lower_word in suffix_mapping:
                replaced_words.append(suffix_mapping[lower_word])
            else:
                replaced_words.append(word)
        # Join words back into a string
        return ' '.join(replaced_words)
    return street_name


# func for grouping similar streets
def standardise_street_names(df):
    tfidf = TFIDF(n_gram_range=(3, 3), min_similarity=0.5)
    results_list = []
    df['Original Index'] = df.index
    df = fill_empty_entries(df)
    grouped = df.groupby(ZIPCODE)[['Extracted Address', 'Original Index']].apply(lambda x: x.values.tolist()) # group addresses by zipcode before running polyfuzz

    # Run PolyFuzz (TFIDF)
    for postal_code, streets_info, in grouped.items():
        streets = [entry[0] for entry in streets_info] # extract street names
        indices = [entry[1] for entry in streets_info] # extract original indices
        streets = [entry if entry is not None else 'Empty' for entry in streets] # make sure all string before Polyfuzz

        if len(streets) > 1:
            model = PolyFuzz(tfidf).match(streets)
            model.group(tfidf, link_min_similarity=0.5, group_all_strings=True)
            matches = model.get_matches()
            matches['Original Index'] = indices # add original index
            results_list.append(matches)
        else:
            # If group only has one entry, keep original name
            single_entry = pd.DataFrame({
                'From': streets[0], 'To': streets[0], 'Group': streets[0], 'Similarity': 1.0, ZIPCODE: postal_code, 'Original Index': [indices[0]]},
                index=[0])
            results_list.append(single_entry)

    # Concatenate results list and reformat
    results_df = pd.concat(results_list, ignore_index=True)
    results_df['Group'] = results_df['Group'].fillna(results_df['From']) # use original name if no group is found
    results_df = results_df.sort_values(by='Original Index').reset_index(drop=True) # sort back to original order
    print("Fuzzymatch successful, length:", len(results_df))  # check length

    # Clean up using previous functions
    results_df['Group'] = results_df['Group'].apply(replace_suffixes).apply(custom_titlecase).apply(remove_double_commas)
    print("Suffix change successful, length:", len(df)) # check length

    # Merge with original df
    if len(df) != len(results_df):
        raise ValueError("The number of rows in df and results_df do not match.")
    df = pd.concat([df.reset_index(drop=True), results_df[['Group']].reset_index(drop=True)], axis=1)
    print("Merge successful, length:", len(df))

    # Reformat df
    df = df.astype(str)
    df.drop(columns=['Original Index', 'Unnamed: 3', 'Is_Company_Street1', 'Is_Company_Street2'], inplace=True, errors='ignore')
    df.rename(columns={'Group': 'Standardised Addresses'}, inplace=True)
    df.replace(['Nan','nan'], '', regex=True, inplace=True)
    df['Extracted Address'] = df['Extracted Address'].apply(remove_double_commas)
    return df


# Apply batch processing
df['Is_Company_Street1'] = is_company_name(df[COLUMN1].tolist())
df['Is_Company_Street2'] = is_company_name(df[COLUMN2].tolist())

# Apply extract function
df['Extracted Address'] = df.apply(concatenate_address, axis=1)
print("Extraction successful, length:", len(df))  # check length

# Apply standardisation function
cleaned_df = standardise_street_names(df)
cleaned_df['Standardised Addresses'] = concatenate_zipcode(cleaned_df['Standardised Addresses'], cleaned_df[ZIPCODE])
cleaned_df['Extracted Address'] = concatenate_zipcode(cleaned_df['Extracted Address'], cleaned_df[ZIPCODE])


# Save result to CSV file for now (have been told this will be changed)
output_file_path = 'big_standardised_extracted.csv'
cleaned_df.to_csv(output_file_path, index=False)
print(f"Results successfully saved to {output_file_path}")

# show time
end_time = time.time()
execution_time = (end_time - start_time) / 60  # Convert time to minutes
print(f"Total script execution time: {execution_time:.2f} minutes")
