import pandas as pd
import numpy as np
from scipy import stats
from mixnb import MixNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_bic(model, exp_matrix):
    """
    Compute BIC score following MATLAB implementation.
    """
    # Convert pandas DF to numpy array if needed
    if isinstance(exp_matrix, pd.DataFrame):
        exp_matrix = exp_matrix.values
        
    # base model likelihood (single cluster)
    mu0 = (np.sum(exp_matrix[model.active, :], axis=1) + model.reg_n) / (model.n_cells + model.reg_d)
    p0 = mu0 / (mu0 + model.r)
    L0 = model.n_cells * model.r * np.log(1-p0) + np.sum(exp_matrix[model.active, :], axis=1) * np.log(p0)
    base_likelihood = np.sum(L0)
    
    # likelihood of current model
    log_likelihood = 0
    for k in range(model.n_clusters):
        cells_in_k = model.class_ == k
        if not any(cells_in_k):
            continue
            
        x_sub = exp_matrix[model.active, :][:, cells_in_k]
        n_cells_k = np.sum(cells_in_k)
        
        # Cluster-specific parameters
        mu_k = (np.sum(x_sub, axis=1) + model.reg_n) / (n_cells_k + model.reg_d)
        p_k = mu_k / (mu_k + model.r)
        
        # Log likelihood for this cluster
        L_k = n_cells_k * model.r * np.log(1-p_k) + np.sum(x_sub, axis=1) * np.log(p_k)
        log_likelihood += np.sum(L_k)
    
    # improvement over base model
    likelihood_improvement = log_likelihood - base_likelihood
    
    # Number of free parameters 
    n_params = model.n_active * model.n_clusters
    
    # BIC calculate
    bic = -2 * likelihood_improvement + model.bic * n_params * np.log(model.n_cells)/2
    
    return bic

def run_mixnb_analysis(exp_matrix, n_clusters):
    """Run MixNB analysis with fixed number of clusters"""
    # Convert pandas DF to numpy array if needed
    if isinstance(exp_matrix, pd.DataFrame):
        data_values = exp_matrix.values
    else:
        data_values = exp_matrix
        
    model = MixNB(data_values)
    model.n_clusters = n_clusters
    
    # Initial clustering using k-means
    print("Running initial k-means clustering...")
    log_data = np.log1p(data_values.T)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    model.class_ = kmeans.fit_predict(log_data)
    
    # Initial M-step
    print("Initial M-step...")
    model.mstep()
    
    # Run EM iterations
    print("Running EM iterations...")
    prev_score = -np.inf
    
    for i in range(50):
        print(f"EM Iteration {i+1}")
        prev_class = model.class_.copy()
        
        changes = model.estep()
        
        # Check for cluster collapse
        unique_clusters = np.unique(model.class_)
        if len(unique_clusters) < n_clusters:
            print("Warning: Cluster collapse detected, reverting to previous state")
            model.class_ = prev_class
            break
        
        model.mstep(changes)
        
        unique_clusters, counts = np.unique(model.class_, return_counts=True)
        print(f"  Current cluster sizes: {dict(zip(unique_clusters, counts))}")
        
        if model.score < prev_score + model.tol:
            print("Score converged!")
            break
        prev_score = model.score
        
        if len(changes) == 0:
            print("Assignments converged!")
            break
        
        print(f"  Changes: {len(changes)} cells")
    
    return model

def run_mixnb_analysis_with_bic(exp_matrix, max_clusters=50, min_clusters=2):
    """Run MixNB analysis with BIC-based cluster selection"""
    print("Running BIC-based cluster selection...")
    
    best_bic = np.inf
    best_model = None
    bic_scores = []
    consecutive_increases = 0
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        print(f"\nTrying {n_clusters} clusters...")
        
        try:
            model = run_mixnb_analysis(exp_matrix, n_clusters)
            
            # Compute BIC
            bic = compute_bic(model, exp_matrix)
            bic_scores.append(bic)
            print(f"BIC score: {bic:0.2e}")
            
            # Check if BIC improved
            if bic < best_bic:
                best_bic = bic
                best_model = model
                consecutive_increases = 0
                print(f"New best model with {n_clusters} clusters")
            else:
                consecutive_increases += 1
                if consecutive_increases >= 3:
                    print("BIC hasn't improved for 3 iterations, stopping...")
                    break
                    
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {str(e)}")
            if best_model is None:
                raise  # Re-raise if we haven't found any valid model
            break
    
    if best_model is None:
        raise ValueError("Failed to find any valid model")
        
    return best_model

def analyze_results(model, exp_matrix):
    """Analyze and visualize the clustering results"""
    if model is None:
        raise ValueError("No model provided for analysis")
        
    if not hasattr(model, 'class_'):
        raise AttributeError("Model does not have required 'class_' attribute")
        
    # Get cluster assignments
    clusters = pd.Series(model.class_, index=exp_matrix.columns)
    
    # Plot cluster sizes
    plt.figure(figsize=(12, 6))
    cluster_sizes = clusters.value_counts().sort_index()
    bars = plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Number of Cells per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.xticks(range(len(cluster_sizes)), cluster_sizes.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return clusters

def main():
 
    exp_matrix = pd.read_csv('expression.tsv', sep='\t', index_col=0)
    print("Initial matrix shape:", exp_matrix.shape, "(genes x cells)")
    
    # Basic QC filters
    min_cells = 10
    min_umis = 5
    genes_pass = ((exp_matrix > min_umis).sum(axis=1) >= min_cells)
    exp_matrix = exp_matrix.loc[genes_pass]
    
    total_umis = exp_matrix.sum(axis=0)
    lower_bound = np.percentile(total_umis, 1)
    upper_bound = np.percentile(total_umis, 99)
    cells_pass = (total_umis > lower_bound) & (total_umis < upper_bound)
    exp_matrix = exp_matrix.loc[:, cells_pass]
    
    print("Final matrix shape:", exp_matrix.shape, "(genes x cells)")
    
    # Run analysis
    model = run_mixnb_analysis_with_bic(exp_matrix)
    clusters = analyze_results(model, exp_matrix)
    
    # Save results
    results = pd.DataFrame({'cluster': clusters})
    results.to_csv('mixnb_results.csv')
    
    print("\nAnalysis complete!")
    return model

plt.savefig('cluster_sizes_histogram.png', dpi=300)

if __name__ == "__main__":
    model = main()