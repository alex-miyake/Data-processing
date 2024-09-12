Context:


Client has SQL databases across USA and Canada, which contains addresses that are verified using a Google Maps API.
However many addresses are flagged as incorrect, because entries often contain typographical errors or are entered in an inconsistent format.
The database also gets regular updates of ~5,000 entries every month.
Client has previously solved this by manually correcting each entry, and instead wants an automated way of standardising data.

AIM 1 (extract.py): To extract address data from 2 columns of messy inputs. (Entries include addresses split over 2 rows, company names, and blank entries).

AIM 2 (standardise.py): To standardise similar extracted address inputs before verification with Google Maps API. To fix the suffixes as API check may not pick up the abbreviations.

Results: Sample dataset of 868 entries takes 5 minutes to process. Accuracy is satisfactory, the script manages to get around 80% of entries correctly extracted and standardised. Manual correction will be required for entries that weren't extracted correctly.


NB: project is still a work in progress
