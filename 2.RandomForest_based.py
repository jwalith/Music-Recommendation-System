#!/usr/bin/env python3
"""
RandomForest-based Music Recommendations
- Feature engineering
- Random Forest ranking model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import wandb
import warnings
warnings.filterwarnings('ignore')

class RandomForestRanker:
    """RandomForest-based ranking system for music recommendations"""
    
    def __init__(self):
        self.spotify_df = None
        self.lastfm_interactions = None
        self.lastfm_artists = None
        
    def load_data(self):
        """Load datasets"""
        print("Loading datasets...")
        
        self.spotify_df = pd.read_csv("archive/spotify_songs.csv")
        print(f"Loaded {len(self.spotify_df)} Spotify songs")
        
        self.lastfm_interactions = pd.read_csv("hetrec2011-lastfm-2k/user_artists.dat", 
                                              sep='\t', names=['user_id', 'artist_id', 'weight'])
        self.lastfm_interactions = self.lastfm_interactions.iloc[1:]
        self.lastfm_interactions['weight'] = pd.to_numeric(self.lastfm_interactions['weight'])
        self.lastfm_interactions['user_id'] = pd.to_numeric(self.lastfm_interactions['user_id'])
        self.lastfm_interactions['artist_id'] = pd.to_numeric(self.lastfm_interactions['artist_id'])
        print(f"Loaded {len(self.lastfm_interactions)} Last.fm interactions")
        
        self.lastfm_artists = pd.read_csv("hetrec2011-lastfm-2k/artists.dat", 
                                         sep='\t', names=['id', 'name', 'url', 'pictureURL'])
        self.lastfm_artists = self.lastfm_artists.iloc[1:]
        self.lastfm_artists['id'] = pd.to_numeric(self.lastfm_artists['id'])
        print(f"Loaded {len(self.lastfm_artists)} Last.fm artists")
        
        return True
    
    def create_features(self):
        """Create basic features"""
        print("Creating features...")
        
        features = self.lastfm_interactions.copy()
        
        # User features
        user_stats = features.groupby('user_id').agg({
            'weight': ['sum', 'mean', 'count'],
            'artist_id': 'nunique'
        }).round(3)
        user_stats.columns = ['user_total_listens', 'user_avg_weight', 'user_interactions', 'user_unique_artists']
        user_stats = user_stats.reset_index()
        
        # Artist features
        artist_stats = features.groupby('artist_id').agg({
            'weight': ['sum', 'mean', 'count'],
            'user_id': 'nunique'
        }).round(3)
        artist_stats.columns = ['artist_total_listens', 'artist_avg_weight', 'artist_interactions', 'artist_unique_users']
        artist_stats = artist_stats.reset_index()
        
        # Merge features
        features = features.merge(user_stats, on='user_id', how='left')
        features = features.merge(artist_stats, on='artist_id', how='left')
        
        # Create derived features
        features['weight_vs_user_avg'] = features['weight'] / features['user_avg_weight']
        features['weight_vs_artist_avg'] = features['weight'] / features['artist_avg_weight']
        features['user_diversity'] = features['user_unique_artists'] / features['user_interactions']
        features['artist_diversity'] = features['artist_unique_users'] / features['artist_interactions']
        features['relative_user_activity'] = features['user_total_listens'] / features['user_total_listens'].max()
        features['relative_artist_popularity'] = features['artist_total_listens'] / features['artist_total_listens'].max()
        features['interaction_strength'] = features['weight'] / (features['user_total_listens'] * features['artist_total_listens'])
        
        print(f"Created {len(features)} feature samples")
        print(f"Features: {list(features.columns)}")
        
        return features
    
    def train_ranking_model(self, features):
        """Train a Random Forest ranking model"""
        print("Training ranking model...")
        
        feature_columns = [
            'user_total_listens', 'user_avg_weight', 'user_interactions', 'user_unique_artists',
            'artist_total_listens', 'artist_avg_weight', 'artist_interactions', 'artist_unique_users',
            'weight_vs_user_avg', 'weight_vs_artist_avg', 'user_diversity', 'artist_diversity',
            'relative_user_activity', 'relative_artist_popularity', 'interaction_strength'
        ]
        
        X = features[feature_columns].fillna(0)
        y = features['weight']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Trained ranking model")
        print("Top 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model, importance_df, X_test, y_test, features
    
    def evaluate_model(self, model, X_test, y_test, features):
        """Evaluate the ranking model"""
        print("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        print("Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }
    
    def demo_recommendations(self, model, features):
        """Demonstrate recommendations for a sample user"""
        print("Demo Recommendations:")
        print("=" * 50)
        
        sample_user = features['user_id'].sample(1).iloc[0]
        user_data = features[features['user_id'] == sample_user].copy()
        
        print(f"User {sample_user} recommendations:")
        print(f"User stats: {user_data['user_total_listens'].iloc[0]:.0f} total listens, {user_data['user_unique_artists'].iloc[0]} unique artists")
        
        current_artists = user_data['artist_id'].tolist()
        print(f"Currently listening to {len(current_artists)} artists")
        
        feature_columns = [
            'user_total_listens', 'user_avg_weight', 'user_interactions', 'user_unique_artists',
            'artist_total_listens', 'artist_avg_weight', 'artist_interactions', 'artist_unique_users',
            'weight_vs_user_avg', 'weight_vs_artist_avg', 'user_diversity', 'artist_diversity',
            'relative_user_activity', 'relative_artist_popularity', 'interaction_strength'
        ]
        
        X_user = user_data[feature_columns].fillna(0)
        predicted_scores = model.predict(X_user)
        
        user_data['predicted_score'] = predicted_scores
        top_recommendations = user_data.nlargest(5, 'predicted_score')
        
        print("Top 5 predicted recommendations:")
        for i, (_, row) in enumerate(top_recommendations.iterrows(), 1):
            artist_name = self.lastfm_artists[self.lastfm_artists['id'] == row['artist_id']]['name'].iloc[0] if len(self.lastfm_artists[self.lastfm_artists['id'] == row['artist_id']]) > 0 else f"Artist {row['artist_id']}"
            print(f"  {i}. {artist_name} (Score: {row['predicted_score']:.3f}, Actual: {row['weight']:.3f})")
    
    def run_stage2(self):
        """Run complete RandomForest-based evaluation"""
        print("Starting RandomForest-based Recommendations")
        print("=" * 60)
        
        wandb.init(
            project="music-recommendation",
            name="randomforest-ranking",
            config={
                "stage": "2",
                "method": "random_forest_ranking",
                "datasets": ["spotify", "lastfm"]
            }
        )
        
        self.load_data()
        features = self.create_features()
        model, importance_df, X_test, y_test, features = self.train_ranking_model(features)
        metrics = self.evaluate_model(model, X_test, y_test, features)
        self.demo_recommendations(model, features)
        
        wandb.log({
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "correlation": metrics['correlation'],
            "total_features": len(importance_df),
            "total_samples": len(features),
            "unique_users": features['user_id'].nunique(),
            "unique_artists": features['artist_id'].nunique()
        })
        
        for i, row in importance_df.head(10).iterrows():
            wandb.log({f"feature_importance_{row['feature']}": row['importance']})
        
        print("RandomForest-based evaluation completed successfully!")
        
        wandb.finish()
        
        return {
            'model': model,
            'importance_df': importance_df,
            'metrics': metrics,
            'features': features
        }

if __name__ == "__main__":
    ranker = RandomForestRanker()
    results = ranker.run_stage2()
