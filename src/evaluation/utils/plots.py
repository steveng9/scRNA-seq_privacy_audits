import pandas as pd
from sklearn.decomposition import PCA
from plotnine import ggplot, aes, geom_point, labs, scale_color_manual

class Plotting:
    @staticmethod
    def perform_pca(data):
        pca = PCA(n_components=2)
        ## standardize here 
        principal_components = pca.fit_transform(data)
        return pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    @staticmethod
    def plot_pca_and_save(original_pca, synthetic_pca, save_file):
        original_pca['Type'] = 'Original'
        synthetic_pca['Type'] = 'Synthetic'

        combined_pca = pd.concat([original_pca, synthetic_pca], ignore_index=True)

        plot = (ggplot(combined_pca, aes(x='PC1', y='PC2', color='Type'))
                + geom_point(alpha=0.5)
                + labs(title='PCA of Original and Synthetic Data',
                       x='Principal Component 1',
                       y='Principal Component 2')
                + scale_color_manual(values=['blue', 'orange']))
        
        plot.save(save_file)

        #return plot
