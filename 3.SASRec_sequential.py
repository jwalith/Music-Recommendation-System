import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MusicSequenceDataset(Dataset):
    """Dataset for sequential music recommendations"""
    
    def __init__(self, sequences, max_length=50):
        self.sequences = sequences
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length:]
        else:
            sequence = [0] * (self.max_length - len(sequence)) + sequence
        
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target = torch.tensor(sequence[-1], dtype=torch.long)
        
        return input_seq, target

class SASRecModel(nn.Module):
    """Self-Attentive Sequential Recommendation Model"""
    
    def __init__(self, vocab_size, hidden_size=128, num_heads=4, num_layers=2, max_length=50):
        super(SASRecModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        self.item_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        item_emb = self.item_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        embeddings = item_emb + pos_emb
        transformer_output = self.transformer(embeddings)
        last_hidden = transformer_output[:, -1, :]
        logits = self.output_layer(last_hidden)
        
        return logits

class SASRecRecommender:
    """SASRec Sequential Recommendations"""
    
    def __init__(self):
        self.spotify_df = None
        self.lastfm_interactions = None
        self.lastfm_artists = None
        self.artist_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data for SASRec...")
        
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
    
    def prepare_sequences(self, min_sequence_length=5, max_sequence_length=50):
        """Prepare user listening sequences"""
        print("Preparing user listening sequences...")
        
        interactions_sorted = self.lastfm_interactions.sort_values(['user_id', 'weight'], ascending=[True, False])
        
        user_sequences = []
        user_ids = []
        
        for user_id, user_data in interactions_sorted.groupby('user_id'):
            artist_ids = user_data['artist_id'].tolist()
            
            if len(artist_ids) >= min_sequence_length:
                if len(artist_ids) > max_sequence_length:
                    artist_ids = artist_ids[:max_sequence_length]
                
                user_sequences.append(artist_ids)
                user_ids.append(user_id)
        
        print(f"Created {len(user_sequences)} user sequences")
        print(f"Average sequence length: {np.mean([len(seq) for seq in user_sequences]):.1f}")
        
        all_artists = [artist_id for seq in user_sequences for artist_id in seq]
        self.artist_encoder.fit(all_artists)
        
        encoded_sequences = []
        for seq in user_sequences:
            encoded_seq = self.artist_encoder.transform(seq)
            encoded_sequences.append(encoded_seq)
        
        print(f"Encoded {len(self.artist_encoder.classes_)} unique artists")
        
        return encoded_sequences, user_ids
    
    def create_datasets(self, sequences, train_ratio=0.8):
        """Create training and validation datasets"""
        print("Creating datasets...")
        
        split_idx = int(len(sequences) * train_ratio)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        train_dataset = MusicSequenceDataset(train_sequences, max_length=50)
        val_dataset = MusicSequenceDataset(val_sequences, max_length=50)
        
        print(f"Training sequences: {len(train_dataset)}")
        print(f"Validation sequences: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the SASRec model"""
        print(f"Training SASRec model for {epochs} epochs...")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        vocab_size = len(self.artist_encoder.classes_) + 1
        self.model = SASRecModel(vocab_size=vocab_size, hidden_size=128, num_heads=4, num_layers=2)
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Enabled mixed precision training for GPU acceleration")
        else:
            self.scaler = None
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if torch.cuda.is_available() and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_inputs)
                        loss = criterion(outputs, batch_targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    if torch.cuda.is_available() and self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_inputs)
                            loss = criterion(outputs, batch_targets)
                    else:
                        outputs = self.model(batch_inputs)
                        loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                })
        
        print("Training completed!")
        return train_losses, val_losses
    
    def evaluate_model(self, val_dataset, k_values=[5, 10, 20]):
        """Evaluate the model with ranking metrics"""
        print(f"Evaluating model with Hit Rate@{k_values} and NDCG@{k_values}...")
        
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.model.eval()
        
        all_hit_rates = {k: [] for k in k_values}
        all_ndcg_scores = {k: [] for k in k_values}
        
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(val_loader, desc="Evaluating"):
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_inputs)
                predictions = torch.softmax(outputs, dim=1)
                
                for k in k_values:
                    _, top_k_indices = torch.topk(predictions, k, dim=1)
                    
                    hits = (top_k_indices == batch_targets.unsqueeze(1)).any(dim=1).float()
                    hit_rate = hits.mean().item()
                    all_hit_rates[k].append(hit_rate)
                    
                    ndcg_scores = []
                    for i, target in enumerate(batch_targets):
                        target_idx = target.item()
                        if target_idx in top_k_indices[i]:
                            position = (top_k_indices[i] == target_idx).nonzero(as_tuple=True)[0].item()
                            ndcg = 1.0 / np.log2(position + 2)
                        else:
                            ndcg = 0.0
                        ndcg_scores.append(ndcg)
                    
                    avg_ndcg = np.mean(ndcg_scores)
                    all_ndcg_scores[k].append(avg_ndcg)
        
        final_metrics = {}
        for k in k_values:
            hit_rate = np.mean(all_hit_rates[k])
            ndcg = np.mean(all_ndcg_scores[k])
            final_metrics[f"hit_rate@{k}"] = hit_rate
            final_metrics[f"ndcg@{k}"] = ndcg
            print(f"Hit Rate@{k}: {hit_rate:.4f}")
            print(f"NDCG@{k}: {ndcg:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log(final_metrics)
        
        return final_metrics
    
    def demo_recommendations(self, sequences, user_ids, num_demos=3):
        """Generate demo recommendations"""
        print(f"Demo Sequential Recommendations:")
        print("=" * 60)
        
        demo_indices = np.random.choice(len(sequences), num_demos, replace=False)
        
        for i, idx in enumerate(demo_indices):
            user_id = user_ids[idx]
            sequence = sequences[idx]
            
            print(f"User {user_id} (Sequence {i+1}):")
            print(f"Sequence length: {len(sequence)}")
            
            artist_names = []
            for artist_id in sequence:
                if artist_id in self.artist_encoder.classes_:
                    encoded_id = self.artist_encoder.transform([artist_id])[0]
                    artist_info = self.lastfm_artists[self.lastfm_artists['id'] == artist_id]
                    if not artist_info.empty:
                        artist_names.append(artist_info['name'].iloc[0])
                    else:
                        artist_names.append(f"Artist_{artist_id}")
                else:
                    artist_names.append(f"Artist_{artist_id}")
            
            print(f"Recent listening history: {', '.join(artist_names[-5:])}")
            
            self.model.eval()
            with torch.no_grad():
                input_seq = torch.tensor(sequence[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                
                outputs = self.model(input_seq)
                predictions = torch.softmax(outputs, dim=1)
                
                _, top_indices = torch.topk(predictions, 5, dim=1)
                top_indices = top_indices.squeeze().cpu().numpy()
                
                recommended_artists = self.artist_encoder.inverse_transform(top_indices)
                
                print(f"Top 5 recommendations:")
                for j, artist_id in enumerate(recommended_artists):
                    artist_info = self.lastfm_artists[self.lastfm_artists['id'] == artist_id]
                    if not artist_info.empty:
                        artist_name = artist_info['name'].iloc[0]
                    else:
                        artist_name = f"Artist_{artist_id}"
                    print(f"  {j+1}. {artist_name}")
    
    def run_sasrec(self):
        """Run complete SASRec pipeline"""
        print("Starting SASRec Sequential Recommendations")
        print("=" * 70)
        
        if WANDB_AVAILABLE:
            wandb.init(project="music-recommendation", name="sasrec-sequential")
        
        self.load_data()
        sequences, user_ids = self.prepare_sequences()
        train_dataset, val_dataset = self.create_datasets(sequences)
        
        train_losses, val_losses = self.train_model(train_dataset, val_dataset, 
                                                   epochs=10, batch_size=128, learning_rate=0.001)
        
        metrics = self.evaluate_model(val_dataset)
        self.demo_recommendations(sequences, user_ids)
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        print("SASRec evaluation completed successfully!")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics
        }

if __name__ == "__main__":
    recommender = SASRecRecommender()
    results = recommender.run_sasrec()
