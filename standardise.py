"""
AIM 2:
    - Standardise extracted address inputs before verification with Google Maps API.
    - Fix the suffixes as API check may not pick up the abbreviations.
    - Standardise the addresses that are meant to be the same location.

This code groups entries by zip code, then uses Polyfuzz TFIDF fuzzymatch to get a similarity score, and assign similar words to a group.
Group name is then edited using a dict of common suffix abbreviations to get a standardised full street address.

(Note: dictionary will have to be manually updated if new abbreviations appear in the new datasets)
Accuracy is satisfactory, manages to get around 70-80% of entries correctly extracted and standardised. Manual correction will be required for entries that weren't extracted correctly.
"""

import pandas as pd
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
import re


# Dictionary of common suffixes
suffix_mapping = {
    'st': 'Street', 'st.': 'Street', 'str': 'Street',
    'ave': 'Avenue', 'ave.': 'Avenue',
    'blvd': 'Boulevard', 'blvd.': 'Boulevard', 'blvd,': 'Boulevard,',
    'rd': 'Road','rd.': 'Road',
    'ln': 'Lane',
    'dr': 'Drive', 'dr.': 'Drive',
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

# Sample dataset
df = pd.read_csv('extracted.csv')
df = df.astype(str)
df['Original Index'] = df.index


# parse FullAddress column
def split_address(entry):
    last_comma_index = entry.rfind(',')     # Find position of last comma

    # If no comma
    if last_comma_index == -1:
        return entry, ''

    # Split string into 2
    address = entry[:last_comma_index].strip()
    zip_code = entry[last_comma_index + 1:].strip()

    return address, zip_code


def tidy_up_commas(street_name):
    return re.sub(r',\s*', ', ', street_name)


# Suffix abbreviations function (case insensitive)
def replace_suffixes(street_name):

    if isinstance(street_name, str):
        words = street_name.split()
        replaced_words = []
        for word in words:
            # Convert word to lowercase for lookup
            lower_word = word.lower()
            # Stop processing if bldg to avoid unnecessary info eg. floor, flat number
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


def standardise_street_names(df):
    results_list = []

    # Group streets by Zip code
    grouped = df.groupby('Postal Code')[['Street', 'Original Index']].apply(lambda x: x.values.tolist())

    # Run polyfuzz (TFIDF)
    for postal_code, streets_info, in grouped.items():
        streets = [entry[0] for entry in streets_info] # extract street name
        indices = [entry[1] for entry in streets_info] # extract original indices

        if len(streets) > 1:
            tfidf = TFIDF(n_gram_range=(3, 3), min_similarity=0.5)
            model = PolyFuzz(tfidf).match(streets)
            model.group(tfidf, link_min_similarity=0.5, group_all_strings=True)
            matches = model.get_matches()
            matches['Postal Code'] = postal_code
            matches['Original Index'] = indices # add original index
            results_list.append(matches)
        else:
            # If group only has one entry, keep original name
            single_entry = pd.DataFrame({
                'From': streets[0], 'To': streets[0], 'Group': streets[0], 'Similarity': 1.0, 'Postal Code': postal_code, 'Original Index': [indices[0]]},
                index=[0])
            results_list.append(single_entry)

    # Concatenate results list and reformat
    results_df = pd.concat(results_list, ignore_index=True)
    results_df['Group'] = results_df['Group'].fillna(results_df['From']) # use original name if no group is found
    results_df = results_df.sort_values(by='Original Index').reset_index(drop=True) # sort back to original order
    print("After fuzzymatch length:", len(results_df))  # check length

    # Clean up using prev functions
    results_df['Group'] = results_df['Group'].apply(replace_suffixes)
    results_df['Group'] = results_df['Group'].apply(tidy_up_commas)

    # check length
    print("After suffix change length:", len(df))

    # Merge with original df
    if len(df) != len(results_df):
        raise ValueError("The number of rows in df and results_df do not match.")

    df = pd.concat([df.reset_index(drop=True), results_df[['Group']].reset_index(drop=True)], axis=1)
    print("After merge length:", len(df))

    # Reformat df
    df['Group'] = df['Group'] + ', ' + df['Postal Code']    # add zipcode back to address name
    df.drop(columns=['Original Index', 'Postal Code'], inplace=True, errors='ignore')
    df.rename(columns={'FullAddress': 'Extracted Addresses', 'Group': 'Standardised Addresses'}, inplace=True)

    return df




# Apply parse function
df[['Street', 'Postal Code']] = df['FullAddress'].apply(lambda x: pd.Series(split_address(x)))

# Apply standardisation function
cleaned_df = standardise_street_names(df)

# Save to new csv file
output_file_path = 'standardised extracted.csv'
cleaned_df.to_csv(output_file_path, index=False)
print(f"results successfully saved to {output_file_path}")
