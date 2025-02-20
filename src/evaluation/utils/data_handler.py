import os
import pandas as pd

  
class EvaluationDataLoader:
    def __init__(self, 
                 dataset_name = "",
                 generator_name = "",
                 save_dir = "", 
                 param_path = None,
                 split_num = None):
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.split_num = split_num
        self.param_path = param_path
        self.generator_name = generator_name

    def load_data(self):
        real_save_dir = os.path.join(self.save_dir, self.dataset_name, "real")
        syn_save_dir = os.path.join(
                                    self.save_dir, 
                                    self.dataset_name, 
                                    "synthetic", 
                                    self.generator_name, 
                                    self.param_path
                                    )
        ## synthetic data 
        synthetic_data = pd.read_csv(os.path.join(
            syn_save_dir, f"synthetic_data_split_{self.split_num}.csv")).values
        synthetic_labels = pd.read_csv(os.path.join(
            syn_save_dir, f"synthetic_labels_split_{self.split_num}.csv")).values
        
        ## real data
        X_train_real = pd.read_csv(os.path.join(
            real_save_dir, f"X_train_real_split_{self.split_num}.csv")).values
        y_train_real = pd.read_csv(os.path.join(
            real_save_dir, f"y_train_real_split_{self.split_num}.csv")).values
        X_test_real = pd.read_csv(os.path.join(
            real_save_dir, f"X_test_real_split_{self.split_num}.csv")).values
        y_test_real = pd.read_csv(os.path.join(
            real_save_dir, f"y_test_real_split_{self.split_num}.csv")).values

        print(synthetic_data.shape)
        print(X_train_real.shape)

        return (synthetic_data, synthetic_labels, X_train_real, 
                y_train_real, X_test_real, y_test_real)
    


  



