import yaml
import os
import sys
import importlib
import anndata as ad
import numpy as np
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

# mapping model names to their respective classes
# be consistent with config.yaml about the use of names 
mia_classes = {
    'domias_baselines': ('models.baseline', 'DOMIASBaselineModels'),
    'sc_domias_baselines': ('models.sc_baseline', 'DOMIASSingleCellBaselineModels')
}




def main():
    data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/ok/2d"
    train = ad.read_h5ad(os.path.join(data_dir, "datasets", "train.h5ad"))
    holdout = ad.read_h5ad(os.path.join(data_dir, "datasets", "holdout.h5ad"))

    targets = ad.concat([train, holdout])
    targets.write_h5ad(os.path.join(data_dir, "datasets", "targets.h5ad"))
    labels = pd.DataFrame()
    labels["membership"] = np.concatenate((np.ones(train.shape[0], dtype=int), np.zeros(holdout.shape[0], dtype=int)))
    labels["barcode_col"] = targets.to_df().index
    labels["individual"] = targets.obs["individual"].values
    labels.to_csv(os.path.join(data_dir, "datasets", "labels.csv"))



    print("Train shape: ", train.shape)
    print("Holdout shape: ", holdout.shape)

    cfg = dict()
    cfg["dir_list"] = dict()
    cfg["dir_list"]["home"] = os.path.join(data_dir, "datasets")
    cfg["dir_list"]["mia_files"] = os.path.join(data_dir, "datasets")
    cfg["generator_config"] = dict()
    cfg["generator_config"]["model_name"] = ""
    cfg["generator_config"]["experiment_name"] = ""
    cfg["attack_model"] = ""
    cfg["dataset_config"] = dict()
    cfg["dataset_config"]["name"] = ""
    cfg["dataset_config"]["membership_label_col"] = "membership"


    synthetic_file = os.path.join(data_dir, "datasets", "synthetic.h5ad")
    mmb_test_file = os.path.join(data_dir, "datasets", "targets.h5ad")
    mia_experiment_name = ""
    mmb_labels_file = os.path.join(data_dir, "datasets", "labels.csv")
    reference_file = os.path.join(data_dir, "datasets", "auxiliary.h5ad")

    module = importlib.import_module("models.sc_baseline")
    MIAClass = getattr(module, "DOMIASSingleCellBaselineModels")

    mia_model = MIAClass(cfg, synthetic_file, mmb_test_file, mmb_labels_file, mia_experiment_name, reference_file)
    
    predictions, y_test = mia_model.run_attack()
    mia_model.results_save_dir = os.path.join(data_dir, "results")
    mia_model.save_predictions(predictions)

    if y_test is not None:
        grp_preds, grp_y = mia_model.perform_donor_level_avg(predictions, y_test)
        mia_model.evaluate_attack(grp_preds, 
                                  grp_y, 
                                  "evaluation_results.csv")





if __name__ == '__main__':
    main()


