import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ndcg_score
import faiss
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

class UserTower(nn.Module):
    """User Tower for generating user embeddings"""
    
    def __init__(self, user_features_dim, embedding_dim=128, hidden_dims=[256, 128]):
        super(UserTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        layers = []
        input_dim = user_features_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.user_tower = nn.Sequential(*layers)
        
    def forward(self, user_features):
        return self.user_tower(user_features)

class ItemTower(nn.Module):
    """Item Tower for generating item embeddings"""
    
    def __init__(self, item_features_dim, embedding_dim=128, hidden_dims=[256, 128]):
        super(ItemTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        layers = []
        input_dim = item_features_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.item_tower = nn.Sequential(*layers)
        
    def forward(self, item_features):
        return self.item_tower(item_features)

class TwoTowerModel(nn.Module):
    """Two-Tower Model for recommendation"""
    
    def __init__(self, user_tower, item_tower, temperature=1.0):
        super(TwoTowerModel, self).__init__()
        
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature
        
    def forward(self, user_features, item_features):
        user_embeddings = self.user_tower(user_features)
        item_embeddings = self.item_tower(item_features)
        
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
        item_embeddings = nn.functional.normalize(item_embeddings, p=2, dim=1)
        
        return user_embeddings, item_embeddings
    
    def compute_similarity(self, user_embeddings, item_embeddings):
        return torch.mm(user_embeddings, item_embeddings.t()) / self.temperature

class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training"""
    
    def __init__(self, interactions, user_features, item_features, user_encoder, item_encoder):
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        user_id = interaction['user_id']
        item_id = interaction['artist_id']
        rating = interaction['rating']
        
        user_feat = self.user_features[user_id]
        item_feat = self.item_features[item_id]
        
        return {
            'user_features': torch.FloatTensor(user_feat),
            'item_features': torch.FloatTensor(item_feat),
            'rating': torch.FloatTensor([rating])
        }

class FAISSRetriever:
    """FAISS-based retrieval system"""
    
    def __init__(self, embedding_dim=128, index_type='IVF'):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.item_ids = None
        
    def build_index(self, item_embeddings, item_ids):
        """Build FAISS index from item embeddings"""
        print("Building FAISS index...")
        
        if isinstance(item_embeddings, torch.Tensor):
            item_embeddings = item_embeddings.detach().cpu().numpy()
        
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        
        if self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            self.index.train(item_embeddings.astype('float32'))
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(item_embeddings.astype('float32'))
        self.item_ids = item_ids
        
        print(f"Built FAISS index with {len(item_ids)} items")
        
    def search(self, query_embeddings, k=10):
        """Search for similar items"""
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()
        
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_embeddings.astype('float32'), k)
        
        results = []
        for i in range(len(indices)):
            item_ids = [self.item_ids[idx] for idx in indices[i]]
            result_scores = scores[i]
            results.append(list(zip(item_ids, result_scores)))
        
        return results

class ColdStartHandler:
    """Handle cold-start scenarios for new users, songs, and artists"""
    
    def __init__(self, user_tower, item_tower, user_encoder, item_encoder):
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
        self.avg_user_embedding = None
        self.avg_item_embedding = None
        
    def compute_average_embeddings(self, user_features, item_features, device):
        """Compute average embeddings for cold-start fallback"""
        print("Computing average embeddings for cold-start...")
        
        with torch.no_grad():
            user_embeddings = self.user_tower(torch.FloatTensor(user_features).to(device))
            self.avg_user_embedding = user_embeddings.mean(dim=0)
            
            item_embeddings = self.item_tower(torch.FloatTensor(item_features).to(device))
            self.avg_item_embedding = item_embeddings.mean(dim=0)
        
        print("Average embeddings computed")
    
    def get_user_embedding(self, user_id, user_features=None, device=None):
        """Get user embedding, handling cold-start"""
        if user_id in self.user_encoder.classes_:
            if user_features is not None:
                with torch.no_grad():
                    self.user_tower.eval()
                    return self.user_tower(torch.FloatTensor(user_features).unsqueeze(0).to(device))
            else:
                return self.avg_user_embedding.unsqueeze(0).to(device)
        else:
            return self.avg_user_embedding.unsqueeze(0).to(device)
    
    def get_item_embedding(self, item_id, item_features=None, device=None):
        """Get item embedding, handling cold-start"""
        if item_id in self.item_encoder.classes_:
            if item_features is not None:
                with torch.no_grad():
                    self.item_tower.eval()
                    return self.item_tower(torch.FloatTensor(item_features).unsqueeze(0).to(device))
            else:
                return self.avg_item_embedding.unsqueeze(0).to(device)
        else:
            return self.avg_item_embedding.unsqueeze(0).to(device)

class EvaluationFramework:
    """Comprehensive evaluation framework"""
    
    def __init__(self):
        self.metrics = {}
    
    def precision_at_k(self, recommendations, ground_truth, k):
        """Calculate Precision@K"""
        if len(recommendations) == 0:
            return 0.0
        
        relevant_items = set(ground_truth)
        recommended_items = set(recommendations[:k])
        
        if len(recommended_items) == 0:
            return 0.0
        
        precision = len(relevant_items.intersection(recommended_items)) / len(recommended_items)
        return precision
    
    def recall_at_k(self, recommendations, ground_truth, k):
        """Calculate Recall@K"""
        if len(ground_truth) == 0:
            return 0.0
        
        relevant_items = set(ground_truth)
        recommended_items = set(recommendations[:k])
        
        recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items)
        return recall
    
    def ndcg_at_k(self, recommendations, ground_truth, k):
        """Calculate NDCG@K"""
        if len(ground_truth) == 0:
            return 0.0
        
        relevance_scores = [1 if item in ground_truth else 0 for item in recommendations[:k]]
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
        
        ideal_relevance = [1] * min(len(ground_truth), k)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_recommendations(self, recommendations_dict, ground_truth_dict, k_values=[5, 10, 20]):
        """Evaluate recommendations for multiple users"""
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id, recommendations in recommendations_dict.items():
                if user_id in ground_truth_dict:
                    ground_truth = ground_truth_dict[user_id]
                    
                    precision = self.precision_at_k(recommendations, ground_truth, k)
                    recall = self.recall_at_k(recommendations, ground_truth, k)
                    ndcg = self.ndcg_at_k(recommendations, ground_truth, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    ndcg_scores.append(ndcg)
            
            results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return results

class TwoTowerRecommender:
    """Two-Tower Architecture with FAISS Retrieval"""
    
    def __init__(self):
        self.spotify_df = None
        self.lastfm_interactions = None
        self.lastfm_artists = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.model = None
        self.retriever = None
        self.cold_start_handler = None
        self.evaluator = EvaluationFramework()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data for Two-Tower...")
        
        self.spotify_df = pd.read_csv("archive/spotify_songs.csv")
        print(f"Loaded {len(self.spotify_df)} Spotify songs")
        
        self.lastfm_interactions = pd.read_csv("hetrec2011-lastfm-2k/user_artists.dat", 
                                              sep='\t', names=['user_id', 'artist_id', 'weight'])
        self.lastfm_interactions = self.lastfm_interactions.iloc[1:]
        self.lastfm_interactions['user_id'] = pd.to_numeric(self.lastfm_interactions['user_id'])
        self.lastfm_interactions['artist_id'] = pd.to_numeric(self.lastfm_interactions['artist_id'])
        self.lastfm_interactions['weight'] = pd.to_numeric(self.lastfm_interactions['weight'])
        print(f"Loaded {len(self.lastfm_interactions)} Last.fm interactions")
        
        self.lastfm_artists = pd.read_csv("hetrec2011-lastfm-2k/artists.dat", 
                                         sep='\t', names=['id', 'name', 'url', 'pictureURL'])
        self.lastfm_artists = self.lastfm_artists.iloc[1:]
        self.lastfm_artists['id'] = pd.to_numeric(self.lastfm_artists['id'])
        print(f"Loaded {len(self.lastfm_artists)} Last.fm artists")
        
        return True
    
    def prepare_features(self):
        """Prepare user and item features"""
        print("Preparing user and item features...")
        
        user_stats = self.lastfm_interactions.groupby('user_id').agg({
            'weight': ['sum', 'mean', 'std', 'count'],
            'artist_id': 'nunique'
        }).round(3)
        
        user_stats.columns = ['total_listens', 'avg_weight', 'weight_std', 'num_interactions', 'unique_artists']
        user_stats = user_stats.fillna(0)
        
        user_stats['total_listens_log'] = np.log1p(user_stats['total_listens'])
        user_stats['avg_weight_log'] = np.log1p(user_stats['avg_weight'])
        user_stats['weight_std_log'] = np.log1p(user_stats['weight_std'])
        
        user_features = user_stats[['total_listens_log', 'avg_weight_log', 'weight_std_log', 'num_interactions', 'unique_artists']]
        
        item_stats = self.lastfm_interactions.groupby('artist_id').agg({
            'weight': ['sum', 'mean', 'std', 'count'],
            'user_id': 'nunique'
        }).round(3)
        
        item_stats.columns = ['total_plays', 'avg_weight', 'weight_std', 'num_interactions', 'unique_users']
        item_stats = item_stats.fillna(0)
        
        item_stats['total_plays_log'] = np.log1p(item_stats['total_plays'])
        item_stats['avg_weight_log'] = np.log1p(item_stats['avg_weight'])
        item_stats['weight_std_log'] = np.log1p(item_stats['weight_std'])
        
        item_features = item_stats[['total_plays_log', 'avg_weight_log', 'weight_std_log', 'num_interactions', 'unique_users']]
        
        self.user_encoder.fit(user_stats.index)
        self.item_encoder.fit(item_stats.index)
        
        user_features = self.user_scaler.fit_transform(user_features.values)
        item_features = self.item_scaler.fit_transform(item_features.values)
        
        user_features_dict = dict(zip(user_stats.index, user_features))
        item_features_dict = dict(zip(item_stats.index, item_features))
        
        print(f"Created features for {len(user_features_dict)} users and {len(item_features_dict)} items")
        print(f"User features dimension: {user_features.shape[1]}")
        print(f"Item features dimension: {item_features.shape[1]}")
        
        return user_features_dict, item_features_dict, user_features, item_features
    
    def create_interactions(self, user_features_dict, item_features_dict):
        """Create interaction dataset"""
        print("Creating interaction dataset...")
        
        valid_interactions = self.lastfm_interactions[
            (self.lastfm_interactions['user_id'].isin(user_features_dict.keys())) &
            (self.lastfm_interactions['artist_id'].isin(item_features_dict.keys()))
        ].copy()
        
        valid_interactions['rating'] = np.log1p(valid_interactions['weight'])
        
        min_rating = valid_interactions['rating'].min()
        max_rating = valid_interactions['rating'].max()
        valid_interactions['rating'] = 1 + 4 * (valid_interactions['rating'] - min_rating) / (max_rating - min_rating)
        
        print(f"Created {len(valid_interactions)} valid interactions")
        
        return valid_interactions
    
    def train_model(self, interactions, user_features_dict, item_features_dict, 
                   user_features, item_features, epochs=7, batch_size=512, learning_rate=0.001):
        """Train the Two-Tower model"""
        print(f"Training Two-Tower model for {epochs} epochs...")
        
        train_dataset = TwoTowerDataset(
            interactions, user_features_dict, item_features_dict,
            self.user_encoder, self.item_encoder
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        user_tower = UserTower(user_features.shape[1], embedding_dim=128)
        item_tower = ItemTower(item_features.shape[1], embedding_dim=128)
        self.model = TwoTowerModel(user_tower, item_tower, temperature=1.0)
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Enabled mixed precision training for GPU acceleration")
        else:
            self.scaler = None
        
        self.cold_start_handler = ColdStartHandler(
            user_tower, item_tower, self.user_encoder, self.item_encoder
        )
        self.cold_start_handler.compute_average_embeddings(user_features, item_features, self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                user_features_batch = batch['user_features'].to(self.device, non_blocking=True)
                item_features_batch = batch['item_features'].to(self.device, non_blocking=True)
                ratings_batch = batch['rating'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if torch.cuda.is_available() and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        user_embeddings, item_embeddings = self.model(user_features_batch, item_features_batch)
                        predictions = torch.sum(user_embeddings * item_embeddings, dim=1)
                        loss = criterion(predictions, ratings_batch.squeeze())
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    user_embeddings, item_embeddings = self.model(user_features_batch, item_features_batch)
                    predictions = torch.sum(user_embeddings * item_embeddings, dim=1)
                    loss = criterion(predictions, ratings_batch.squeeze())
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss
                })
        
        print("Training completed!")
        return train_losses
    
    def build_retrieval_system(self, user_features_dict, item_features_dict):
        """Build FAISS retrieval system"""
        print("Building FAISS retrieval system...")
        
        item_ids = list(item_features_dict.keys())
        item_features_matrix = np.array([item_features_dict[item_id] for item_id in item_ids])
        
        with torch.no_grad():
            self.model.eval()
            item_embeddings = self.model.item_tower(torch.FloatTensor(item_features_matrix).to(self.device))
        
        self.retriever = FAISSRetriever(embedding_dim=128)
        self.retriever.build_index(item_embeddings, item_ids)
        
        print("FAISS retrieval system built!")
    
    def generate_recommendations(self, user_id, user_features_dict, k=10):
        """Generate recommendations for a user"""
        if user_id in user_features_dict:
            user_features = user_features_dict[user_id]
            user_embedding = self.cold_start_handler.get_user_embedding(user_id, user_features, self.device)
        else:
            user_embedding = self.cold_start_handler.get_user_embedding(user_id, device=self.device)
        
        recommendations = self.retriever.search(user_embedding, k=k)
        
        return [item_id for item_id, score in recommendations[0]]
    
    def evaluate_model(self, interactions, user_features_dict, k_values=[5, 10, 20]):
        """Evaluate the model comprehensively"""
        print(f"Evaluating model with comprehensive metrics...")
        
        test_users = interactions['user_id'].unique()[:100]
        test_interactions = interactions[interactions['user_id'].isin(test_users)]
        
        ground_truth = {}
        for user_id in test_users:
            user_items = test_interactions[test_interactions['user_id'] == user_id]['artist_id'].tolist()
            ground_truth[user_id] = user_items
        
        recommendations = {}
        for user_id in test_users:
            recs = self.generate_recommendations(user_id, user_features_dict, k=max(k_values))
            recommendations[user_id] = recs
        
        results = self.evaluator.evaluate_recommendations(recommendations, ground_truth, k_values)
        
        for k in k_values:
            print(f"Precision@{k}: {results[f'precision@{k}']:.4f}")
            print(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
            print(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log(results)
        
        return results
    
    def demo_recommendations(self, user_features_dict, num_demos=3):
        """Generate demo recommendations"""
        print(f"Demo Two-Tower Recommendations:")
        print("=" * 60)
        
        demo_users = np.random.choice(list(user_features_dict.keys()), num_demos, replace=False)
        
        for i, user_id in enumerate(demo_users):
            print(f"User {user_id} (Demo {i+1}):")
            
            recommendations = self.generate_recommendations(user_id, user_features_dict, k=5)
            
            print(f"Top 5 recommendations:")
            for j, artist_id in enumerate(recommendations):
                artist_info = self.lastfm_artists[self.lastfm_artists['id'] == artist_id]
                if not artist_info.empty:
                    artist_name = artist_info['name'].iloc[0]
                else:
                    artist_name = f"Artist_{artist_id}"
                print(f"  {j+1}. {artist_name}")
    
    def run_two_tower(self):
        """Run complete Two-Tower pipeline"""
        print("Starting Two-Tower Architecture with FAISS Retrieval")
        print("=" * 80)
        
        if WANDB_AVAILABLE:
            wandb.init(project="music-recommendation", name="two-tower")
        
        self.load_data()
        user_features_dict, item_features_dict, user_features, item_features = self.prepare_features()
        interactions = self.create_interactions(user_features_dict, item_features_dict)
        
        train_losses = self.train_model(interactions, user_features_dict, item_features_dict, 
                                      user_features, item_features, 
                                      epochs=7, batch_size=512, learning_rate=0.001)
        
        self.build_retrieval_system(user_features_dict, item_features_dict)
        metrics = self.evaluate_model(interactions, user_features_dict)
        self.demo_recommendations(user_features_dict)
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        print("Two-Tower evaluation completed successfully!")
        
        return {
            "train_losses": train_losses,
            "metrics": metrics
        }

if __name__ == "__main__":
    recommender = TwoTowerRecommender()
    results = recommender.run_two_tower()
