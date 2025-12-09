import os
import pandas as pd
import numpy as np
import anndata as ad

class MIADataLoader:
    def __init__(self, 
                synthetic_file: str,
                membership_test_file: str,
                membership_lbl_file:str,
                membership_label_col: str,
                generator_model: str,
                reference_file: str = None):
        self.generator_model = generator_model
        self.membership_test_file = membership_test_file
        self.membership_lbl_file = membership_lbl_file
        self.membership_label_col = membership_label_col
        self.synthetic_file = synthetic_file
        self.reference_file = reference_file


    def load_synthetic_data(self):
        # synthetic_data = pd.read_csv(self.synthetic_file).values
        synthetic_data = ad.read_h5ad(self.synthetic_file).to_df()
        print("Synth shape: ", synthetic_data.shape)

        return synthetic_data
    
    
    def load_membership_dataset(self):
        if not os.path.exists(self.membership_test_file):
            raise FileNotFoundError("Membership test dataset is missing.")
        # dataset = pd.read_csv(self.membership_test_file,
        #                                  sep="\t",
        #                                  index_col=0).T.values
        dataset = ad.read_h5ad(self.membership_test_file).to_df()
        
        print("Target shape: ", dataset.shape)

        return dataset
    
    def load_membership_labels(self):
        labels = None
        if self.membership_lbl_file is not None:
            labels = pd.read_csv(self.membership_lbl_file, index_col=0)[
                                        self.membership_label_col].values
        return labels



    
    def load_reference_data(self):
        if self.reference_file:
            # reference = pd.read_csv(self.reference_file,
            #                              sep="\t",
            #                              index_col=0).T
            reference = ad.read_h5ad(self.reference_file).to_df()
            print("Aux shape:", reference.shape)
            return reference
        else:
            return None
    
    
    @staticmethod
    def save_files(save_dir, file_name_list, array_list):

        assert len(file_name_list) == len(array_list)

        for i in range(len(file_name_list)):
            np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)
