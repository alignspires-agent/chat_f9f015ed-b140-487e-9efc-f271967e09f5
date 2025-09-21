
import sys
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSESFramework:
    """
    Simplified implementation of the MoSEs (Mixture of Stylistic Experts) framework
    for AI-generated text detection based on the research paper.
    """
    
    def __init__(self, n_prototypes=5, n_components=32, random_state=42):
        """
        Initialize the MoSEs framework.
        
        Args:
            n_prototypes: Number of prototypes per style category
            n_components: Number of PCA components for semantic feature compression
            random_state: Random seed for reproducibility
        """
        self.n_prototypes = n_prototypes
        self.n_components = n_components
        self.random_state = random_state
        
        # Core components
        self.srr_prototypes = {}  # Stylistics Reference Repository prototypes
        self.srr_features = {}    # Reference features for each prototype
        self.srr_labels = {}      # Reference labels for each prototype
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.cte_model = None     # Conditional Threshold Estimator
        
        logger.info("MoSEs framework initialized with %d prototypes and %d PCA components", 
                   n_prototypes, n_components)
    
    def extract_features(self, texts):
        """
        Extract linguistic and semantic features from texts.
        This is a simplified version that simulates feature extraction.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of extracted features
        """
        logger.info("Extracting features from %d texts", len(texts))
        
        # Simulate feature extraction (in a real implementation, this would use actual NLP techniques)
        features = []
        for text in texts:
            # Simulate linguistic features
            text_length = len(text.split())
            log_prob_mean = np.random.normal(0, 1)  # Simulated log probability mean
            log_prob_var = np.random.normal(0.5, 0.2)  # Simulated log probability variance
            ngram_repetition_2 = np.random.uniform(0, 0.3)  # Simulated 2-gram repetition
            ngram_repetition_3 = np.random.uniform(0, 0.2)  # Simulated 3-gram repetition
            type_token_ratio = np.random.uniform(0.4, 0.8)  # Simulated type-token ratio
            
            # Simulate semantic embeddings (would use BGE-M3 in real implementation)
            semantic_embedding = np.random.normal(0, 1, 100)  # 100-dim random embedding
            
            # Combine all features
            linguistic_features = np.array([
                text_length, log_prob_mean, log_prob_var, 
                ngram_repetition_2, ngram_repetition_3, type_token_ratio
            ])
            
            # Compress semantic features with PCA
            if hasattr(self, 'pca_fitted'):
                semantic_compressed = self.pca.transform(semantic_embedding.reshape(1, -1))[0]
            else:
                semantic_compressed = np.random.normal(0, 1, self.n_components)
            
            # Combine all features
            combined_features = np.concatenate([linguistic_features, semantic_compressed])
            features.append(combined_features)
        
        return np.array(features)
    
    def build_srr(self, texts, labels, styles):
        """
        Build the Stylistics Reference Repository (SRR) with prototype-based approximation.
        
        Args:
            texts: List of reference texts
            labels: List of labels (0=human, 1=AI)
            styles: List of style categories for each text
        """
        logger.info("Building Stylistics Reference Repository with %d samples", len(texts))
        
        # Extract features from all texts
        features = self.extract_features(texts)
        
        # Fit PCA on semantic features (first fit on all data)
        semantic_features = np.array([np.random.normal(0, 1, 100) for _ in texts])  # Simulated
        self.pca.fit(semantic_features)
        self.pca_fitted = True
        
        # Standardize features
        features = self.scaler.fit_transform(features)
        
        # Group texts by style and create prototypes
        unique_styles = set(styles)
        for style in unique_styles:
            style_mask = [s == style for s in styles]
            style_features = features[style_mask]
            style_labels = np.array(labels)[style_mask]
            
            # Cluster to find prototypes
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=self.random_state, n_init=10)
            kmeans.fit(style_features)
            
            # Store prototypes and their associated data
            self.srr_prototypes[style] = kmeans.cluster_centers_
            self.srr_features[style] = style_features
            self.srr_labels[style] = style_labels
            
            logger.info("Created %d prototypes for style: %s", self.n_prototypes, style)
    
    def sar_router(self, text_features, m=3):
        """
        Stylistics-Aware Router: Find m-nearest prototypes to the input text.
        
        Args:
            text_features: Features of the input text
            m: Number of nearest prototypes to retrieve
            
        Returns:
            Indices of activated reference samples
        """
        logger.info("Routing text to nearest prototypes")
        
        # Standardize input features
        text_features = self.scaler.transform(text_features.reshape(1, -1))[0]
        
        # Find distances to all prototypes across all styles
        all_prototypes = []
        prototype_info = []  # Store (style, cluster_idx) for each prototype
        
        for style, prototypes in self.srr_prototypes.items():
            for idx, prototype in enumerate(prototypes):
                all_prototypes.append(prototype)
                prototype_info.append((style, idx))
        
        all_prototypes = np.array(all_prototypes)
        
        # Calculate distances
        distances = cdist(text_features.reshape(1, -1), all_prototypes)[0]
        
        # Get m-nearest prototypes
        nearest_indices = np.argsort(distances)[:m]
        activated_samples = []
        
        for idx in nearest_indices:
            style, cluster_idx = prototype_info[idx]
            # Get all samples from this cluster
            cluster_mask = self._get_cluster_samples(style, cluster_idx)
            activated_samples.extend(cluster_mask)
        
        logger.info("Activated %d reference samples via %d nearest prototypes", 
                   len(activated_samples), m)
        
        return activated_samples
    
    def _get_cluster_samples(self, style, cluster_idx):
        """Get indices of samples belonging to a specific cluster."""
        # This is a simplified version - in a real implementation, we would
        # store cluster assignments during SRR construction
        features = self.srr_features[style]
        prototypes = self.srr_prototypes[style]
        
        # Find samples closest to this prototype
        distances = cdist(prototypes[cluster_idx].reshape(1, -1), features)[0]
        nearest_indices = np.argsort(distances)[:10]  # Top 10 samples per prototype
        
        return [(style, i) for i in nearest_indices]
    
    def train_cte(self, activated_samples):
        """
        Train Conditional Threshold Estimator on activated reference samples.
        
        Args:
            activated_samples: Indices of activated reference samples
            
        Returns:
            Trained CTE model
        """
        logger.info("Training Conditional Threshold Estimator on %d samples", len(activated_samples))
        
        # Collect features and labels from activated samples
        X_train, y_train = [], []
        
        for style, idx in activated_samples:
            X_train.append(self.srr_features[style][idx])
            y_train.append(self.srr_labels[style][idx])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Train logistic regression model
        try:
            self.cte_model = LogisticRegression(
                random_state=self.random_state, 
                class_weight='balanced'
            )
            self.cte_model.fit(X_train, y_train)
            
            logger.info("CTE model trained successfully with %d samples", len(X_train))
            return self.cte_model
            
        except Exception as e:
            logger.error("Failed to train CTE model: %s", str(e))
            sys.exit(1)
    
    def predict(self, texts):
        """
        Predict whether texts are AI-generated using the MoSEs framework.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            predictions: List of predicted labels (0=human, 1=AI)
            confidences: List of confidence scores
        """
        if self.cte_model is None:
            logger.error("CTE model not trained. Please train the model first.")
            sys.exit(1)
        
        logger.info("Predicting labels for %d texts", len(texts))
        
        predictions, confidences = [], []
        
        for text in texts:
            # Extract features
            features = self.extract_features([text])[0]
            
            # Route to find relevant samples
            activated_samples = self.sar_router(features)
            
            # Standardize features
            features_std = self.scaler.transform(features.reshape(1, -1))[0]
            
            # Predict using CTE
            proba = self.cte_model.predict_proba(features_std.reshape(1, -1))[0]
            prediction = np.argmax(proba)
            confidence = np.max(proba)
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for demonstration purposes.
    In a real implementation, this would be replaced with actual dataset loading.
    
    Returns:
        texts: List of synthetic texts
        labels: List of labels (0=human, 1=AI)
        styles: List of style categories
    """
    logger.info("Generating synthetic data with %d samples", n_samples)
    
    texts, labels, styles = [], [], []
    style_categories = ['news', 'academic', 'dialogue', 'story']
    
    for i in range(n_samples):
        # Randomly assign style and label
        style = np.random.choice(style_categories)
        label = np.random.randint(0, 2)  # 0=human, 1=AI
        
        # Generate simple text based on style and label
        if style == 'news':
            text = "This is a news article about current events. " * np.random.randint(2, 5)
        elif style == 'academic':
            text = "The research findings indicate a significant correlation. " * np.random.randint(3, 6)
        elif style == 'dialogue':
            text = "Person A: How are you? Person B: I'm fine, thank you. " * np.random.randint(2, 4)
        else:  # story
            text = "Once upon a time, in a land far away. " * np.random.randint(3, 7)
        
        # Add some variation based on label
        if label == 1:  # AI-generated
            text += "Furthermore, it is important to note that. " * np.random.randint(1, 3)
        
        texts.append(text)
        labels.append(label)
        styles.append(style)
    
    return texts, labels, styles

def main():
    """Main function to demonstrate the MoSEs framework."""
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Initialize MoSEs framework
        meses = MoSESFramework(n_prototypes=3, n_components=16)
        
        # Generate synthetic data (replace with real data in practice)
        logger.info("Generating training data")
        train_texts, train_labels, train_styles = generate_synthetic_data(500)
        
        logger.info("Generating test data")
        test_texts, test_labels, test_styles = generate_synthetic_data(100)
        
        # Build Stylistics Reference Repository
        meses.build_srr(train_texts, train_labels, train_styles)
        
        # For demonstration, activate samples from first test text
        test_features = meses.extract_features([test_texts[0]])[0]
        activated_samples = meses.sar_router(test_features)
        
        # Train Conditional Threshold Estimator
        meses.train_cte(activated_samples)
        
        # Make predictions on test data
        predictions, confidences = meses.predict(test_texts)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info("Accuracy: %.2f%%", accuracy * 100)
        logger.info("F1 Score: %.4f", f1)
        logger.info("Confidence range: %.2f - %.2f", min(confidences), max(confidences))
        
        # Show sample predictions
        logger.info("Sample predictions:")
        for i in range(min(5, len(test_texts))):
            pred_label = "AI" if predictions[i] == 1 else "Human"
            true_label = "AI" if test_labels[i] == 1 else "Human"
            logger.info("Text %d: Predicted=%s, Actual=%s, Confidence=%.2f", 
                       i+1, pred_label, true_label, confidences[i])
        
        logger.info("MoSEs framework demonstration completed successfully")
        
    except Exception as e:
        logger.error("Error in MoSEs framework: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
