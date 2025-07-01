import numpy as np
import pandas as pd
from tqdm import tqdm

def add_shifted_series(df, series_data, new_column_name=None):
    if new_column_name is None:
        series_name = series_data.name if series_data.name is not None else 'series'
        new_column_name = f'shifted_{series_name}'
    
    # Create a new DataFrame with the same structure as the original
    # Using a copy preserves the index structure
    result_df = df.copy()
    
    # Initialize the new column with zeros
    result_df[new_column_name] = 0.0
    
    # Extract index levels for cell_num and cycle for faster lookup
    cell_indices = df.index.get_level_values('cell_num')
    cycle_indices = df.index.get_level_values('cycle')
    
    # Get unique combinations
    unique_pairs = set(zip(cell_indices, cycle_indices))
    
    # Process each (cell_num, cycle) group
    for cell_num, cycle in tqdm(unique_pairs):
        # Find positions in df for this group
        mask = (cell_indices == cell_num) & (cycle_indices == cycle)
        positions = np.where(mask)[0]
        
        # Skip if only one or zero positions
        if len(positions) <= 1:
            continue
            
        # Find matching series values for this group
        series_cell = series_data.index.get_level_values('cell_num')
        series_cycle = series_data.index.get_level_values('cycle')
        series_mask = (series_cell == cell_num) & (series_cycle == cycle)
        
        if not np.any(series_mask):
            continue
            
        series_values = series_data.iloc[np.where(series_mask)[0]].values
        
        # Create the shifted values array (all zeros by default)
        shifted_values = np.zeros(len(positions))
        
        # Apply the shift in one vectorized operation
        shift_length = min(len(positions) - 1, len(series_values))
        shifted_values[1:shift_length+1] = series_values[:shift_length]
        
        # Update the result DataFrame
        for i, pos in enumerate(positions):
            result_df.iloc[pos, result_df.columns.get_loc(new_column_name)] = shifted_values[i]
    
    return result_df

def train_test_split(x_data, y_data, typ, test, masterList):
    if typ == 'cell':
        cell_test = test
        cell_train = [x for x in masterList.keys() if x not in cell_test]
        x_train, y_train =  x_data.loc[pd.IndexSlice[cell_train, : ], :] , y_data.loc[pd.IndexSlice[cell_train, :]]
        x_test, y_test = x_data.loc[pd.IndexSlice[cell_test, : ], :] , y_data.loc[pd.IndexSlice[cell_test, :]]
        print('Training list: ', cell_train)
        print('Testing list: ', cell_test) 
    elif typ == 'cycle':
        #assuming one cell
        # max_cycle = int(x_data.iloc[x_data.reset_index()['cycle'].idxmax()].name)
        cycles = x_data.reset_index()['cycle'].unique()

        cycle_test = list(test)
        cycle_train = list(np.array([x for x in cycles if x not in cycle_test]))
        print('Training list: ', cycle_train)
        print('Testing list: ', cycle_test) 
        
        x_train, y_train =  x_data.loc[pd.IndexSlice[cycle_train], :] , y_data.loc[pd.IndexSlice[cycle_train]]
        x_test, y_test = x_data.loc[pd.IndexSlice[cycle_test], :] , y_data.loc[pd.IndexSlice[cycle_test]]

    return x_train, y_train, x_test, y_test
