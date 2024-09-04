import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths for the list of SMILES strings and the pre-trained model
new_smiles_file_path = 'conquest_search/2021all_smiles.smi'
model_output_path = 'xgboost_best_model_descriptors.pkl'
congloms_smiles_path = 'conquest_search/2021congloms_smiles.smi'
non_congloms_smiles_path = 'conquest_search/2021noncongloms_smiles.smi'

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
    'EState_VSA2', 'HallKierAlpha', 'PEOE_VSA6', 'fr_NH0', 'EState_VSA7',
    'Kappa2', 'SMR_VSA1', 'PEOE_VSA1', 'SMR_VSA3', 'SMR_VSA10',
    'VSA_EState3', 'PEOE_VSA12', 'Chi0v', 'NumAromaticHeterocycles', 'SMR_VSA4',
    'FpDensityMorgan2', 'EState_VSA9', 'Kappa1'
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

# Function to load SMILES strings and their corresponding refcodes from a file
def load_smiles_and_refcodes(path):
    data = pd.read_csv(path, header=None)
    smiles_list = data.iloc[:, 0].dropna().tolist() # Extract SMILES strings
    refcodes_list = data.iloc[:, 1].dropna().tolist() if data.shape[1] > 1 else smiles_list
    return smiles_list, refcodes_list

def main():
    # List of thresholds to evaluate
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # Load the trained model
    logging.info(f"Loading model from {model_output_path}")
    model = joblib.load(model_output_path)

    # Load the new SMILES data
    new_smiles_list, refcodes_list = load_smiles_and_refcodes(new_smiles_file_path)
    
    # Calculate descriptors for new SMILES strings
    logging.info("Calculating descriptors for new SMILES strings")
    new_smiles_descriptors_df = calculate_descriptors(new_smiles_list)
    
    # Handle infinite values and NaN values in new data
    new_smiles_descriptors_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    new_smiles_descriptors_df = pd.DataFrame(imputer.fit_transform(new_smiles_descriptors_df), columns=new_smiles_descriptors_df.columns)

    # Standardize the new data
    logging.info("Standardizing the new data")
    scaler = StandardScaler()
    new_smiles_descriptors_df = pd.DataFrame(scaler.fit_transform(new_smiles_descriptors_df), columns=new_smiles_descriptors_df.columns)

    # Predict the probability of each SMILES being a conglomerate using the loaded model
    logging.info("Predicting with the loaded model")
    prediction_probas = model.predict_proba(new_smiles_descriptors_df)[:, 1]

    # Load the actual conglomerates and non-conglomerates data for validation
    congloms_smiles_list, _ = load_smiles_and_refcodes(congloms_smiles_path)
    non_congloms_smiles_list, _ = load_smiles_and_refcodes(non_congloms_smiles_path)

    # Create a dictionary to map SMILES to their actual labels (1 for conglomerate, 0 for non-conglomerate)
    actual_labels = {smiles: 1 for smiles in congloms_smiles_list}
    actual_labels.update({smiles: 0 for smiles in non_congloms_smiles_list})

    # Prepare the results DataFrame by mapping predictions to actual labels
    results_df = pd.DataFrame({
        'SMILES': new_smiles_list,
        'Prediction_Probability': prediction_probas
    })
    results_df['Actual'] = results_df['SMILES'].map(actual_labels)
    results_df = results_df.dropna(subset=['Actual'])   # Remove entries without actual labels

    # Convert the actual labels to integers
    results_df['Actual'] = results_df['Actual'].astype(int)

    # Evaluate and print results for each threshold
    for threshold in thresholds:
        results_df['Prediction'] = (results_df['Prediction_Probability'] >= threshold).astype(int)

        print(f"\nThreshold: {threshold}")
        print("Classification Report:")
        print(classification_report(results_df['Actual'], results_df['Prediction']))

        print("Confusion Matrix:")
        print(confusion_matrix(results_df['Actual'], results_df['Prediction']))

    # Save the comparison results
    comparison_output_file = 'comparison_results_2021_smiles.csv'
    results_df.to_csv(comparison_output_file, index=False)
    logging.info(f"Comparison results saved to {comparison_output_file}")

if __name__ == '__main__':
    main()
