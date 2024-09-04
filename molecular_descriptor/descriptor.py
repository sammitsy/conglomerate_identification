import os
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths for the list of SMILES strings
congloms_file_path_csv = 'conquest_search/congloms_smiles.csv'
non_congloms_file_path_csv = 'conquest_search/non-congloms_smiles.csv'

# List of descriptor names
descriptor_names = [
    'SMR_VSA5', 'SlogP_VSA2', 'TPSA', 'FpDensityMorgan3', 'Kappa3',
    'SlogP_VSA6', 'BalabanJ', 'MaxAbsEStateIndex', 'PEOE_VSA9', 'SlogP_VSA5',
    'SlogP_VSA3', 'FpDensityMorgan1', 'VSA_EState6', 'VSA_EState7', 'EState_VSA10',
    'VSA_EState8', 'PEOE_VSA7', 'VSA_EState2', 'fr_Al_OH_noTert', 'MinAbsEStateIndex',
    'PEOE_VSA10', 'VSA_EState4', 'VSA_EState9', 'MolLogP', 'VSA_EState1',
    'MinEStateIndex', 'MinAbsPartialCharge', 'SlogP_VSA4', 'FractionCSP3', 'MinPartialCharge',
    'Chi4n', 'MaxEStateIndex', 'qed', 'MaxPartialCharge', 'Chi4v',
    'EState_VSA3', 'MaxAbsPartialCharge', 'PEOE_VSA8', 'Chi0n', 'EState_VSA4',
    'EState_VSA2', 'HallKierAlpha', 'PEOE_VSA6', 'fr_NH0', 'EState_VSA7'
]

# Initialize the molecular descriptor calculator with the list of selected descriptors
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Create a function to calculate molecular descriptors for a list of SMILES strings
def calculate_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        # Convert the SMILES strings to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Calculate descriptors and add to the list
            descriptors.append(calculator.CalcDescriptors(mol))
        else:
            # Handle invalid SMILES by appending None values
            descriptors.append([None] * len(descriptor_names))
            logging.warning(f"Failed to convert SMILES: {smiles}")
    return pd.DataFrame(descriptors, columns=descriptor_names)

# Create a function to load molecular data from a specified file.
def load_data(path):
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path, sep='\t', header=None)

# Create a main function to load molecular data, calculate descriptors, and save the results
def main():
    # Load the data for conglomerates and non-conglomerates
    congloms_data = load_data(congloms_file_path_csv)
    non_congloms_data = load_data(non_congloms_file_path_csv)

    # Calculate descriptors for conglomerate molecules
    congloms_descriptors_df = calculate_descriptors(congloms_data.iloc[:, 0].dropna())
    # Calculate descriptors for non-conglomerate molecules
    non_congloms_descriptors_df = calculate_descriptors(non_congloms_data.iloc[:, 0].dropna())

    # Save the calculated descriptors to CSV files
    congloms_descriptors_df.to_csv('congloms_descriptors_cut8.csv', index=False)
    non_congloms_descriptors_df.to_csv('non_congloms_descriptors_cut8.csv', index=False)

    logging.info("Descriptors computed and saved successfully.")

# Entry point for the script
if __name__ == '__main__':
    main()
