#tests for retrieving spectral data and making calculated fields
import pytest
import numpy as np
import pandas as pd
from TunaSimNetwork.datasetBuilder import foldCreation

class Test_folding_worflow:

    def test_inchikey_base_mz_map_0(self):
        
        folder = foldCreation()

        base_retrieval_data = pd.DataFrame({'inchikey_base': ['a','b'],
                                            'retrieved_mass': [10,20],
                                            'calculated_mass': [11,21],
                                            'monoisotopic_mass': [12,22]})
        
        bases = ['a','b']
        masses = [10, 20]

        mapping = folder.generate_inchikey_base_mz_map(base_retrieval_data,
                                                    bases,
                                                    masses) 
        
        res2 = [mapping[i] for i in base_retrieval_data['inchikey_base']]

        answer = [set([10,11,12]), set([20,21,22])]
        assert len(mapping) == 2 and np.all(np.array([res2[i]==answer[i] for i in range(len(res2))]))

    def test_inchikey_base_mz_map_1(self):
        
        folder = foldCreation()

        base_retrieval_data = pd.DataFrame({'inchikey_base': ['a','b'],
                                            'retrieved_mass': [10,20],
                                            'calculated_mass': [11,20],
                                            'monoisotopic_mass': [12,20]})
        
        bases = ['a','b','b']
        masses = [10, 21, 21]

        mapping = folder.generate_inchikey_base_mz_map(base_retrieval_data,
                                                    bases,
                                                    masses) 
        
        res2 = [mapping[i] for i in base_retrieval_data['inchikey_base']]

        answer = [set([10,11,12]), set([20,21])]
        assert len(mapping) == 2 and np.all(np.array([res2[i]==answer[i] for i in range(len(res2))]))

    def test_inchikey_base_mz_map_3(self):
        
        folder = foldCreation()

        base_retrieval_data = pd.DataFrame({'inchikey_base': ['a','b'],
                                            'retrieved_mass': [10,20],
                                            'calculated_mass': [11,21],
                                            'monoisotopic_mass': [12,22]})
        
        bases = ['a','b','b','b','a']
        masses = [14, 20, 20, 23, 13]

        mapping = folder.generate_inchikey_base_mz_map(base_retrieval_data,
                                                    bases,
                                                    masses) 
        
        res1 = np.array(mapping.keys())
        res2 = [mapping[i] for i in base_retrieval_data['inchikey_base']]

        answer = [set([10,11,12,13,14]), set([20,21,22,23])]
        assert len(mapping) == 2 and np.all(np.array([res2[i]==answer[i] for i in range(len(res2))]))