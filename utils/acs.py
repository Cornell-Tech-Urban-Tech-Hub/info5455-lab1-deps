import pandas as pd 

def parse_md(md):
    cols = md['variables'].index.tolist()

    print(f"Found {len(cols)} columns in the dataset")
    md = pd.json_normalize(md['variables']).set_index(md.index)
    

   

    min_sep = min(md['label'].apply(lambda x: x.count('!!')))
    max_sep = max(md['label'].apply(lambda x: x.count('!!')))

    for i in range(min_sep + 1, max_sep + 2):  # Adjusting range to account for correct indexing
        md[f'desc_{i}'] = md['label'].apply(
            lambda x: x.split('!!')[i-1] if len(x.split('!!')) >= i else None
        )
    

    md['column'] = cols

    TO_DROP = ['label','concept','predicateType','group','limit','predicateOnly']
    md = md.drop(columns=TO_DROP)

    desc_1_filter = ['Estimate']
    md = md[md['desc_1'].isin(desc_1_filter)]
   
    md = md.sort_index()

    # put the 'column' column first 
    md = md[['column'] + [col for col in md.columns if col != 'column']]

    return md

def parse_acs(acs, cols: dict):

    acs.columns = acs.iloc[0]
    acs = acs[1:]
    acs['tract_id'] = acs['GEO_ID'].str.split('US', expand=True)[1]
    acs = acs.set_index('tract_id')

    acs = acs[list(cols.keys())]
    acs.columns = acs.columns.map(lambda x: cols[x])

    acs = acs.astype(int)
    return acs



def get_acs_data(year, identifier, cols_to_keep):
    raw = pd.read_json(f"data/acs{year}_{identifier}.json")
    parsed_data = parse_acs(raw, cols_to_keep)
    # Add year column after parsing
    parsed_data['year'] = year

    # quick validate 
    quick_validate_acs(parsed_data)

    return parsed_data

def get_acs_data_range(start, end, identifier, cols_to_keep):
    return pd.concat([get_acs_data(year, identifier, cols_to_keep) 
                     for year in range(start, end+1)])


def quick_validate_acs(datset): 
    # check for missing values
    missing = datset.isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values")

import pandas as pd
import geopandas as gpd

def combine_acs_years(data_dict, year_range):
    """
    Combines ACS data from multiple years into a single DataFrame with year as a column.
    
    Parameters:
    data_dict (dict): Dictionary of DataFrames, keyed by year
    year_range (range or list): Range of years to process
    
    Returns:
    DataFrame: Combined data with year column
    """
    combined = []
    for year in year_range:
        if year in data_dict:
            df = data_dict[year].copy()
            df['year'] = year
            combined.append(df)
    
    if not combined:
        return pd.DataFrame()
    
    return pd.concat(combined, ignore_index=True)

def merge_acs_data(ct_nyc, year_start, year_end, acs_columns):
    """
    Merges ACS data for specified years into the census tract GeoDataFrame.
    Handles multiple years correctly by creating separate rows for each year.
    
    Parameters:
    ct_nyc (GeoDataFrame): Base census tract geodataframe
    year_start (int): Start year
    year_end (int): End year
    acs_columns (dict): Nested dictionary containing ACS dataset configurations
                       Format: {
                           'dataset_code': {
                               'name': 'Dataset Name',
                               'columns': {
                                   'acs_code': 'friendly_name',
                                   ...
                               }
                           },
                           ...
                       }
    
    Returns:
    GeoDataFrame: Merged dataset with proper year handling
    """
    # Create a list to store DataFrames for each year
    yearly_dfs = []
    
    # Process each year
    for year in range(year_start, year_end + 1):
        # Start with a copy of the base census tracts
        ct_year = ct_nyc.copy()
        ct_year['year'] = year
        
        merged_year = ct_year
        
        # Process each ACS dataset
        for dataset_code, dataset_info in acs_columns.items():
            # Get data for this dataset and year
            dataset_df = get_acs_data(year, dataset_code, dataset_info['columns'])
            dataset_df['year'] = year
            
            # Merge with the growing result
            merged_year = merged_year.merge(
                dataset_df,
                left_on=['GEOID', 'year'],
                right_on=['tract_id', 'year'],
                how='left'
            )
        
        yearly_dfs.append(merged_year)
    
    # Combine all years
    result = pd.concat(yearly_dfs, ignore_index=True)
    
    # Add helpful metadata
    result.attrs['acs_years'] = list(range(year_start, year_end + 1))
    result.attrs['acs_datasets'] = list(acs_columns.keys())
    
    return result

def verify_acs_data(merged_data, acs_columns):
    """
    Verifies that all expected columns are present in the merged dataset.
    
    Parameters:
    merged_data (GeoDataFrame): The merged ACS data
    acs_columns (dict): The ACS columns configuration dictionary
    
    Returns:
    tuple: (bool, list) - (whether all columns are present, list of missing columns)
    """
    expected_columns = []
    for dataset_info in acs_columns.values():
        expected_columns.extend(dataset_info['columns'].values())
    
    actual_columns = merged_data.columns
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    
    print(f"\nData Verification Report:")
    print(f"Number of census tracts: {len(merged_data['GEOID'].unique())}")
    print(f"Years present: {sorted(merged_data['year'].unique())}")
    print(f"Expected columns present: {len(expected_columns) - len(missing_columns)}/{len(expected_columns)}")
    
    if missing_columns:
        print("\nMissing columns:")
        for col in missing_columns:
            print(f"- {col}")
    else:
        print("\nAll expected columns are present!")
    
    return len(missing_columns) == 0, missing_columns