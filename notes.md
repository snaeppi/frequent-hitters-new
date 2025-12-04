PubChem data processing step-by-step:

`assay_etl`:

1. Get list of assays with more than 10k substances screened from https://www.ncbi.nlm.nih.gov/pcassay?term=10000%3A10000000%5BTotal%20Sid%20Count%5D
2. For each assay, download the corresponding data table from https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi using the query format found in the download menu at https://pubchem.ncbi.nlm.nih.gov/bioassay/1#section=Data-Table (unfortunately there is no PUG-REST method for downloading compressed full data tables). We use a rate limited to be respectful, but keep in mind that this is not an official API end point and might error mid-run. The script is intelligent enough to not redownload data, so continuing is as easy as rerunning the script later. The FTP server could also be used, but it groups the assays by index into ranges of 1000 assays, where you might only need 1 assay in a range, so you are wasting a lot of bandwidth downloading data that is not needed.
3. Convert the gzipped CSVs to parquet for efficiency.
4. Group the data by PUBCHEM_CID to get compound level activity data: aggregate categorical columns (i.e. SMILES) by mode, and numeric columns by median. Sometimes the parquet files may have the wrong data type (floats stored as utf8), so we try to cast the columns into float and treat them as numeric if successful.
5. Manually go through every single assay that has at least one numeric column to see if they could be used for computing primary screen activity R-scores.
   
> The CLI tool shows some important stats including how many screens the assay has, how many of them would be hits if using the outcome-based hit definitions, and how many of those would be flagged as hits using the R-score based definition, as well as the coverage of the column (ratio of non null values). While going through the assays one by one, we read the descriptions on https://pubchem.ncbi.nlm.nih.gov/bioassay/{AID} to understand what the columns that could potentially be used based on their statistics were actually measuring. We required at least 90% coverage and if there were outcome based hits, we required all or nearly all to be flagged in case of confirmatory screens, or in the case of primary screens that used a looser definition than ours, we required the column to flag some of them as hits. In the case of multiple options, we prioritized those with better coverage, and better overlap ratio of the outcome-based hits. As a tie breaker (this only happened for confirmatory assays where both a single concentration activity and "max response" had full overlap and full coverage, we chose the single concentration for consistency, unless it resulted in dramatically less compounds marked as hits (-20% or more). Usually, the values were nearly identical as the max response is very often the highest concentration activity. There are many ways in which an assay might be deemed ineligible. The first is lack of coverage: if no column has >90% coverage on the assay, those measurements do not reflect a full range of activity (they might e.g. be recorded only for "Active" outcomes) and the R-scores would not be meaningful. 90% is not actually a real threshold, we basically saw either 0, less than 25%, or greater than 95% for all the assays. So in principle the selections would be the same with something like a 60% minimum coverage.

6. For those assays with a selected column, get the target type and bioactivity type annotations from Hit Dexter 3.0. For any assay without reported annotations, manually go through the descriptions https://pubchem.ncbi.nlm.nih.gov/bioassay/{AID} and annotate according to the same definitions as Hit Dexter 3.0.
7. Compute the R-scores based on the selected columns and compile all of the assay data into a single portable parquet file with the columns assay_id, compound_id, smiles, r_score.

`assay_cleaning`:

1. Clean the compounds according to Hit Dexter 3.0 rules.
2. Convert R-scores to "active" labels.
3. Split the parquet file into separate biochemical and cellular files according to metadata.

`dataset_pipeline`:

1. Compute scaffold mapping for all compounds (parquet file with smiles, scaffold_smiles) if not provided.
2. Filter out assays that do not pass minimum screen count filter.
3. Filter out assays that have an outlier assay hit rate (mean + 3std)
4. Aggregate assay, compound hit labels into compound hit and screen counts for biochemical and cellular assays separately.
5. Compute beta prior with method of moments on the hit and screen counts of compounds with higher than set minimum of screens.
6. Compute the posterior empirical Bayes smoothed hit rates.
7. Calculate percentiles for the scores and store them in a JSON file for use with the threshold binary classifiers and evaluation of the models with classification metrics.
8. Scaffold split the compounds into train, val, calib, test sets.
9. For multi-task data, do not aggregate into hit and screen counts, but instead pivot the long format data into wide form, where each assay_id becomes its own column and each compound has a single row with their hit labels. Save multi-task datasets in their own files.
10. For all other models, save the compound_id, smiles, screen count, hit count and posterior hit rate (score) data in parquet files.

`
