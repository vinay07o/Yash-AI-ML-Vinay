import numpy as np
import pandas as pd
import pytest
from train import split_data, data_balance


@pytest.fixture
def syn_df():
    np.random.seed(7)
    df = pd.DataFrame({'A':np.random.randint(0, 9, 10), 
                       'B':np.random.randint(0, 9, 10), 
                       'Y':np.random.randint(0, 2, 10)})
    return df

def test_split_data(syn_df):
    """Running test cases for split data."""
  
    split_dict = split_data(syn_df, 'Y')
    
    # testing 20% of data is splitted or not
    np.testing.assert_almost_equal(20, int(len(split_dict['test']['X'])/len(syn_df)*100))
    np.testing.assert_almost_equal(80, int(len(split_dict['train']['X'])/len(syn_df)*100))


def test_data_balance(syn_df):
    """testing Balancing imbalanced dataframe."""

    baldf = data_balance(syn_df[['A', 'B']], syn_df['Y'])
    np.testing.assert_almost_equal(baldf['balanced_y'].value_counts()[0], 
                                   baldf['balanced_y'].value_counts()[1])