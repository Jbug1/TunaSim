import pandas as pd
import numpy as np
import time
import re
import requests
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
import sqlite3


class compoundRetriever:

    def __init__(self, inter_query_sleep_time: float = 0.2):
        self.inter_query_sleep_time = inter_query_sleep_time

    def get_pubchem_data_from_inchikey(self, inchikey: str):
        """
        simply returns first hit
        """

        response = self.request_retry(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSmiles,InChI,MolecularFormula,MonoisotopicMass/json')

        if response.status_code == 200:

            response_data = response.json()['PropertyTable']['Properties']

            return (inchikey, response_data[0]['CID'],
                    response_data[0]['InChI'],
                    response_data[0]['ConnectivitySMILES'],
                    response_data[0]['MolecularFormula'],
                    response_data[0]['MonoisotopicMass'])

        return response.status_code

    def request_retry(self, url, retries = 10):

        attempts = 1
        while attempts < retries:

            try:
                return requests.get(url)

            except Exception as err:
                if attempts < retries:
                    attempts += 1

                else:
                    raise err

    def get_CTS_data_from_inchikey(self, inchikey):


        inchi = [i['results'] for i in self.request_retry(f'https://cts.fiehnlab.ucdavis.edu/rest/convert/InChIKey/inchi Code/{inchikey}').json()]

        cid = [i['results'] for i in self.request_retry(f'https://cts.fiehnlab.ucdavis.edu/rest/convert/InChIKey/PubChem CID/{inchikey}').json()]

        if len(inchi) != len(cid):
            return 'length mismatch'

        return [(inchikey, cid[i], inchi[i]) for i in range(len(inchi))]

    def get_annotations_for_inchikeys(self, inchikeys):
        """
        Try to gather from pubchem first and then check CTS for those we cannot find

        """
        errored_keys = list()

        pubchem_data = list()
        for key in inchikeys:

            try:
                response = self.get_pubchem_data_from_inchikey(key)

            except Exception as err:
                errored_keys.append((key, err, 'pubchem'))

            if type(response) != int:

                pubchem_data.append(response)

            time.sleep(self.inter_query_sleep_time)

        unsuccessful_keys = list(set(inchikeys) - set([i[0] for i in pubchem_data]))

        cts_data = list()
        for key in unsuccessful_keys:

            try:
                response = self.get_CTS_data_from_inchikey(key)

            except Exception as err:
                errored_keys.append((key, err, 'CTS'))

            for group in response:

                if np.any((len(group[1]) > 0, len(group[2]) > 0)):

                    cid = group[1][0] if len(group[1]) > 0 else ''
                    inchi = group[2][0] if len(group[2]) > 0 else ''

                    cts_data.append((group[0], cid, inchi))

                #otherwise make a note that we were unable to find this key
                else:
                    errored_keys.append((key, 'not found'))

        cts_data = self.integrate_cts_data(cts_data)

        pubchem_data = self.integrate_pubchem_data(pubchem_data)

        return (pd.concat((pubchem_data, cts_data)), errored_keys)

    def integrate_pubchem_data(self, pubchem_data):

        calculated_masses = [ExactMolWt(Chem.MolFromInchi(i[2])) for i in pubchem_data]

        return pd.DataFrame({'inchikey': [i[0] for i in pubchem_data],
                             'inchikey_base': [i[0].split('-')[0] for i in pubchem_data],
                            'CID': [i[1] for i in pubchem_data],
                            'inchi': [i[2] for i in pubchem_data],
                            'smiles': [i[3] for i in pubchem_data],
                            'formula': [i[4] for i in pubchem_data],
                            'retrieved_mass': [float(i[5]) for i in pubchem_data],
                            'calculated_mass': calculated_masses,
                            'monoisotopic_mass': [self.formula_to_mass(i[4]) for i in pubchem_data],
                            'source': ['pubchem' for _ in pubchem_data]})


    def integrate_cts_data(self, cts_data):
        """
        reshapes and converts to df, mondful of missing smiles
        """

        formulas = list()
        calculated_masses = list()
        for inchi in [i[2] for i in cts_data]:

            mol = Chem.MolFromInchi(inchi)
            formula = CalcMolFormula(mol)
            calculated_mass = ExactMolWt(mol)

            formulas.append(formula)
            calculated_masses.append(calculated_mass)

        #monoisotopic masses will be negative here to screen out at future point
        return pd.DataFrame({'inchikey': [i[0] for i in cts_data],
                             'inchikey_base': [i[0].split('-')[0] for i in cts_data],
                             'CID': [i[1] for i in cts_data],
                             'inchi': [i[2] for i in cts_data],
                             'smiles': ['' for _ in cts_data],
                             'formula': formulas,
                             'retrieved_mass': [-1 for _ in range(len(cts_data))],
                             'calculated_mass': calculated_masses,
                             'monoisotopic_mass': [self.formula_to_mass(i) for i in formulas],
                             'source': ['cts' for _ in cts_data]})

    def parse_formula(self, formula):
        """Parse a molecular formula string like 'H2O' into element counts: {'H': 2, 'O': 1}"""
        composition = {}
        for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
            if not element:
                continue
            composition[element] = composition.get(element, 0) + (int(count) if count else 1)
        return composition

    def formula_to_mass(self, formula):
        """Convert a molecular formula string to its monoisotopic mass.

        Args:
            formula: e.g. 'C6H12O6', 'H2O', 'NaCl'
        Returns:
            monoisotopic mass in Da
        """

        MONOISOTOPIC_MASSES = {
            'H':   1.00782503, 'He':  4.00260325, 'Li':  7.01600450,
            'Be':  9.01218220, 'B':  11.00930540,  'C':  12.00000000,
            'N':  14.00307401, 'O':  15.99491462,  'F':  18.99840322,
            'Ne': 19.99244018, 'Na': 22.98922070,  'Mg': 23.98504170,
            'Al': 26.98153860, 'Si': 27.97692653,  'P':  30.97376163,
            'S':  31.97207100, 'Cl': 34.96885268,  'Ar': 39.96238312,
            'K':  38.96370668, 'Ca': 39.96259098,  'Sc': 44.95591190,
            'Ti': 47.94794630, 'V':  50.94395950,  'Cr': 51.94050750,
            'Mn': 54.93804510, 'Fe': 55.93493750,  'Co': 58.93319500,
            'Ni': 57.93534290, 'Cu': 62.92959750,  'Zn': 63.92914220,
            'Ga': 68.92557360, 'Ge': 73.92117780,  'As': 74.92159650,
            'Se': 79.91652130, 'Br': 78.91833710,  'Kr': 83.91150700,
            'Rb': 84.91178974, 'Sr': 87.90561210,  'Y':  88.90584800,
            'Zr': 89.90470440, 'Nb': 92.90637300,  'Mo': 97.90540820,
            'Tc': 97.90721600, 'Ru': 101.9043493,  'Rh': 102.9055040,
            'Pd': 105.9034860, 'Ag': 106.9050930,  'Cd': 113.9033585,
            'In': 114.9038780, 'Sn': 119.9022016,  'Sb': 120.9038157,
            'Te': 129.9062244, 'I':  126.9044680,  'Xe': 131.9041535,
            'Cs': 132.9054519, 'Ba': 137.9052470,  'La': 138.9063530,
            'Ce': 139.9054387, 'Pr': 140.9076528,  'Nd': 141.9077233,
            'Pm': 144.9127440, 'Sm': 151.9197324,  'Eu': 152.9212303,
            'Gd': 157.9241039, 'Tb': 158.9253468,  'Dy': 163.9291748,
            'Ho': 164.9303221, 'Er': 165.9302931,  'Tm': 168.9342133,
            'Yb': 173.9388621, 'Lu': 174.9407718,  'Hf': 179.9465500,
            'Ta': 180.9479958, 'W':  183.9509312,  'Re': 186.9557531,
            'Os': 191.9614807, 'Ir': 192.9629264,  'Pt': 194.9647911,
            'Au': 196.9665870, 'Hg': 201.9706430,  'Tl': 204.9744275,
            'Pb': 207.9766521, 'Bi': 208.9803987,  'Po': 208.9824304,
            'At': 209.9871480, 'Rn': 222.0175777,  'Fr': 223.0197359,
            'Ra': 226.0254098, 'Ac': 227.0277521,  'Th': 232.0380553,
            'Pa': 231.0358840, 'U':  238.0507882,
        }

        composition = self.parse_formula(formula)
        mass = 0.0
        for element, count in composition.items():
            if element not in MONOISOTOPIC_MASSES:
                raise ValueError(f"Unknown element: {element}")
            mass += MONOISOTOPIC_MASSES[element] * count
        return mass

    def augment_hydrogen(self, formula: str, add: bool = True):

        if 'H' not in formula:
            raise ValueError('cannot correct formula lacking hydrogen')

        for i in range(len(formula)):

            if formula[i] == 'H' and formula[i+1].isdigit():

                hydrogen_index = i
                break

        int_length = 1
        while hydrogen_index + int_length < len(formula) and formula[hydrogen_index + int_length].isdigit():

            int_length +=1

        if add:
            formula = formula[:hydrogen_index + 1] \
                    + str(int(formula[hydrogen_index + 1: hydrogen_index + int_length]) + 1) \
                    + formula[hydrogen_index + int_length:]

        else:
            formula = formula[:hydrogen_index + 1] \
                    + str(int(formula[hydrogen_index + 1: hydrogen_index + int_length]) - 1) \
                    + formula[hydrogen_index + int_length:]

        return formula
    

class simDB:

    def __init__(self, db_path):
        self.db_path = db_path

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS similarities (query INTEGER, target INTEGER, mces INTEGER, mz_match INTEGER)')
        
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS error_instances (query INTEGER, target INTEGER)')
        
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS inchikey_core_mapping (inchikey_core TEXT, index_map INTEGER)')
        

    def write_sims(self, results):
        self._conn.executemany(
            'INSERT INTO similarities VALUES (?, ?, ?, ?)', results)
        self._conn.commit()

    def write_errors(self, results):
        self._conn.executemany(
            'INSERT INTO error_instances VALUES (?, ?)', results)
        self._conn.commit()

    def write_inchikey_core_mapping(self, results):
        self._conn.executemany(
            'INSERT INTO inchikey_core_mapping VALUES (?, ?)', results)
        self._conn.commit()

    def index_sims(self):
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_sims_mz_mces ON similarities (query, mz_match DESC, mces DESC)')
        self._conn.commit()

    def read_table(self, table_name):
        return pd.read_sql_query(f'SELECT * FROM {table_name}', self._conn)

    def close(self):
        self._conn.close()

