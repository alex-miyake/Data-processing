Context:
Client has SQL databases across USA and Canada, which contains addresses that are verified using a Google Maps API.
However many addresses are flagged as incorrect, because entries often contain typographical errors or are entered in an inconsistent format.
The database also gets regular updates of ~5,000 entries every month.
Client has previously solved this by manually correcting each entry, and instead wants an automated way of standardising data.

AIM 1 (extract.py): Extract address data from 2 columns of messy inputs. Entries include addresses split over 2 rows, company names, and blank entries.

This extraction uses PyTorch and pre-trained transformer BERT for Named Entity Recognition (NER).
This is used to identify if an entry is not a company, and then concatenates the address with zipcode.


AIM 2 (standardise.py): Standardise extracted address inputs before verification with Google Maps API. Fix the suffixes as API check may not pick up the abbreviations.
Standardise the addresses that are meant to be the same location.

This code groups entries by zip code, then uses Polyfuzz TFIDF fuzzymatch to get a similarity score, and assign similar words to a group.
Group name is then edited using a dict of common suffix abbreviations to get a standardised full street address.

(Note: dictionary will have to be manually updated if new abbreviations appear in the new datasets)
Accuracy is satisfactory, manages to get around 70-80% of entries correctly extracted and standardised. Manual correction will be required for entries that weren't extracted correctly.


NB: project is still a work in progress
