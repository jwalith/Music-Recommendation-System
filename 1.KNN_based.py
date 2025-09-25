#!/usr/bin/env python3
"""
KNN-based Music Recommendations
- Popularity-based recommendations
- Item-Item kNN using audio features
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KNNRecommender:
    """KNN-based recommendation system with popularity and content-based methods"""
    
    def __init__(self):
        self.spotify_df = None
        self.lastfm_interactions = None
        self.lastfm_artists = None
        self.popularity_scores = None
        self.audio_features_scaler = None
        self.knn_model = None
        self.audio_features = ['danceability', 'energy', 'valence', 'tempo', 
                             'loudness', 'speechiness', 'acousticness', 
                             'instrumentalness', 'liveness']
        
    def load_data(self):
        """Load and prepare datasets"""
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
    
    def prepare_popularity_baseline(self):
        """Prepare popularity-based recommendations"""
        print("Preparing popularity baseline...")
        
        artist_popularity = self.lastfm_interactions.groupby('artist_id')['weight'].sum().reset_index()
        artist_popularity.columns = ['artist_id', 'total_listens']
        
        artist_popularity = artist_popularity.merge(
            self.lastfm_artists[['id', 'name']], 
            left_on='artist_id', 
            right_on='id'
        )
        
        artist_popularity = artist_popularity.sort_values('total_listens', ascending=False)
        artist_popularity['popularity_score'] = (
            artist_popularity['total_listens'] / artist_popularity['total_listens'].max()
        )
        
        self.popularity_scores = artist_popularity
        
        print(f"Prepared popularity scores for {len(artist_popularity)} artists")
        print("Top 5 popular artists:")
        for i, row in artist_popularity.head().iterrows():
            print(f"  {row['name']}: {row['total_listens']} listens")
        
        return artist_popularity
    
    def prepare_content_baseline(self):
        """Prepare content-based recommendations using audio features"""
        print("Preparing content-based baseline...")
        
        audio_data = self.spotify_df[self.audio_features].copy()
        audio_data = audio_data.fillna(audio_data.median())
        
        self.audio_features_scaler = StandardScaler()
        audio_features_scaled = self.audio_features_scaler.fit_transform(audio_data)
        
        self.knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
        self.knn_model.fit(audio_features_scaled)
        
        print(f"Trained kNN model on {len(audio_data)} songs")
        print(f"Audio features used: {self.audio_features}")
        
        return True
    
    def get_popularity_recommendations(self, n_recommendations=10):
        """Get popularity-based recommendations"""
        if self.popularity_scores is None:
            self.prepare_popularity_baseline()
        
        recommendations = self.popularity_scores.head(n_recommendations)
        
        return {
            'artist_names': recommendations['name'].tolist(),
            'popularity_scores': recommendations['popularity_score'].tolist(),
            'total_listens': recommendations['total_listens'].tolist()
        }
    
    def get_content_recommendations(self, song_index, n_recommendations=10):
        """Get content-based recommendations for a song"""
        if self.knn_model is None:
            self.prepare_content_baseline()
        
        song_features = self.spotify_df.iloc[song_index][self.audio_features].values.reshape(1, -1)
        song_features_scaled = self.audio_features_scaler.transform(song_features)
        
        distances, indices = self.knn_model.kneighbors(song_features_scaled)
        
        similar_songs = indices[0][1:n_recommendations+1]
        distances = distances[0][1:n_recommendations+1]
        
        recommendations = []
        for idx, dist in zip(similar_songs, distances):
            song_info = self.spotify_df.iloc[idx]
            recommendations.append({
                'track_name': song_info['track_name'],
                'track_artist': song_info['track_artist'],
                'similarity_score': 1 - dist,
                'genre': song_info['playlist_genre']
            })
        
        return recommendations
    
    def evaluate_baseline_methods(self):
        """Evaluate baseline methods using simple metrics"""
        print("Evaluating baseline methods...")
        
        wandb.init(
            project="music-recommendation",
            name="knn-baseline-evaluation",
            config={
                "stage": "1",
                "methods": ["popularity", "content_based"],
                "datasets": ["spotify", "lastfm"]
            }
        )
        
        popularity_recs = self.get_popularity_recommendations(20)
        
        sample_songs = self.spotify_df.sample(n=min(100, len(self.spotify_df)))
        content_similarities = []
        
        for idx, song in sample_songs.iterrows():
            try:
                recs = self.get_content_recommendations(idx, 5)
                avg_similarity = np.mean([r['similarity_score'] for r in recs])
                content_similarities.append(avg_similarity)
            except:
                continue
        
        metrics = {
            'popularity_top_artist_score': popularity_recs['popularity_scores'][0],
            'popularity_diversity': len(set(popularity_recs['artist_names'])),
            'popularity_avg_listens': np.mean(popularity_recs['total_listens']),
            'content_avg_similarity': np.mean(content_similarities) if content_similarities else 0,
            'content_similarity_std': np.std(content_similarities) if content_similarities else 0,
            'content_songs_evaluated': len(content_similarities),
            'total_songs': len(self.spotify_df),
            'total_artists': len(self.lastfm_artists),
            'total_interactions': len(self.lastfm_interactions),
            'unique_users': self.lastfm_interactions['user_id'].nunique()
        }
        
        wandb.log(metrics)
        wandb.log({
            "top_popular_artists": popularity_recs['artist_names'][:10],
            "top_popular_scores": popularity_recs['popularity_scores'][:10]
        })
        
        print("Evaluation Results:")
        print(f"Popularity Baseline:")
        print(f"  - Top artist score: {metrics['popularity_top_artist_score']:.3f}")
        print(f"  - Diversity: {metrics['popularity_diversity']} unique artists")
        print(f"  - Avg listens: {metrics['popularity_avg_listens']:.1f}")
        
        print(f"Content-Based Baseline:")
        print(f"  - Avg similarity: {metrics['content_avg_similarity']:.3f}")
        print(f"  - Similarity std: {metrics['content_similarity_std']:.3f}")
        print(f"  - Songs evaluated: {metrics['content_songs_evaluated']}")
        
        wandb.finish()
        
        return metrics
    
    def demo_recommendations(self):
        """Demonstrate both recommendation methods"""
        print("Demo Recommendations:")
        print("=" * 50)
        
        print("Popularity-Based Recommendations:")
        pop_recs = self.get_popularity_recommendations(5)
        for i, (name, score, listens) in enumerate(zip(
            pop_recs['artist_names'], 
            pop_recs['popularity_scores'], 
            pop_recs['total_listens']
        )):
            print(f"  {i+1}. {name} (Score: {score:.3f}, Listens: {listens})")
        
        print("Content-Based Recommendations:")
        demo_song_idx = np.random.randint(0, len(self.spotify_df))
        demo_song = self.spotify_df.iloc[demo_song_idx]
        
        print(f"  For song: '{demo_song['track_name']}' by {demo_song['track_artist']}")
        print(f"  Genre: {demo_song['playlist_genre']}")
        
        content_recs = self.get_content_recommendations(demo_song_idx, 5)
        for i, rec in enumerate(content_recs):
            print(f"  {i+1}. '{rec['track_name']}' by {rec['track_artist']} "
                  f"(Similarity: {rec['similarity_score']:.3f}, Genre: {rec['genre']})")
    
    def run_stage1(self):
        """Run complete KNN-based evaluation"""
        print("Starting KNN-based Recommendations")
        print("=" * 50)
        
        self.load_data()
        self.prepare_popularity_baseline()
        self.prepare_content_baseline()
        metrics = self.evaluate_baseline_methods()
        self.demo_recommendations()
        
        print("KNN-based evaluation completed successfully!")
        
        return metrics

if __name__ == "__main__":
    recommender = KNNRecommender()
    results = recommender.run_stage1()
