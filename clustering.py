import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize

# Setup logger
logger = logging.getLogger("IngestionPipeline.Clustering")

def normalize_embeddings(embeddings: List[List[float]]) -> np.ndarray:
    """
    Performs L2 normalization on every embedding vector using scikit-learn.
    
    Args:
        embeddings: A list of embedding vectors.
        
    Returns:
        A L2-normalized numpy array of shape (n_embeddings, embedding_dim).
    """
    if not embeddings:
        logger.warning("No embeddings provided for normalization.")
        return np.empty((0, 0))
    
    # Convert to numpy array
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    normalized = normalize(embeddings_arr, norm='l2', axis=1)
    
    logger.info(f"Normalization completed for {len(embeddings)} embeddings.")
    return normalized

def cluster_embeddings(normalized_embeddings: np.ndarray, min_cluster_size: int = 3, min_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clusters the normalized embeddings using HDBSCAN.
    
    Args:
        normalized_embeddings: A numpy array of L2-normalized embeddings.
        min_cluster_size: The minimum size of clusters.
        min_samples: The number of samples in a neighborhood for a point to be considered a core point.
        
    Returns:
        A tuple containing (labels, probabilities).
    """
    n_samples = normalized_embeddings.shape[0]
    if n_samples == 0:
        logger.warning("No embeddings to cluster.")
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    
    # Dynamically adjust min_cluster_size to prevent errors for small datasets
    # If we have very few samples, min_cluster_size is set to a smaller value (minimum 2)
    actual_min_cluster_size = max(2, min(min_cluster_size, n_samples))
    actual_min_samples = max(1, min(min_samples, actual_min_cluster_size - 1))
    
    logger.info(f"Clustering {n_samples} embeddings with HDBSCAN (min_cluster_size={actual_min_cluster_size}, min_samples={actual_min_samples})...")
    
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=actual_min_cluster_size,
            min_samples=actual_min_samples,
            metric='euclidean'  # Euclidean distance on L2-normalized vectors is monotonic with cosine distance
        )
        clusterer.fit(normalized_embeddings)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}. Defaulting all elements to noise.")
        labels = np.full(n_samples, -1, dtype=np.int32)
        probabilities = np.zeros(n_samples, dtype=np.float32)
        
    return labels, probabilities

def assign_metadata_to_chunks(
    chunks: List[Dict[str, Any]], 
    labels: np.ndarray, 
    probabilities: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Assigns cluster_id, cluster_probability, and is_noise back to each chunk.
    
    Args:
        chunks: The list of chunk dicts.
        labels: The HDBSCAN cluster label for each chunk.
        probabilities: The HDBSCAN cluster probability/confidence for each chunk.
        
    Returns:
        The updated list of chunks with the new metadata fields.
    """
    if len(chunks) != len(labels):
        logger.error(f"Length mismatch: {len(chunks)} chunks vs {len(labels)} cluster labels.")
        return chunks
        
    unique_labels = set(labels)
    num_clusters = len([l for l in unique_labels if l != -1])
    noise_points = int(np.sum(labels == -1))
    
    # Calculate average cluster size
    if num_clusters > 0:
        cluster_sizes = [int(np.sum(labels == label)) for label in unique_labels if label != -1]
        avg_cluster_size = float(np.mean(cluster_sizes))
    else:
        avg_cluster_size = 0.0
        
    logger.info(f"Number of clusters found: {num_clusters}")
    logger.info(f"Number of noise points: {noise_points}")
    logger.info(f"Average cluster size: {avg_cluster_size:.2f}")
    
    for idx, chunk in enumerate(chunks):
        cluster_id = int(labels[idx])
        prob = float(probabilities[idx])
        is_noise = (cluster_id == -1)
        
        chunk["cluster_id"] = cluster_id
        chunk["cluster_probability"] = prob
        chunk["is_noise"] = is_noise
        
    return chunks
