#!/usr/bin/env python3
"""
Data Exploration for Music Recommendation System
"""

import pandas as pd
import numpy as np
import wandb
from pathlib import Path

def explore_datasets():
    """Explore datasets and log insights to W&B"""
    
    print("Starting data exploration...")
    print("=" * 50)
    
    wandb.init(
        project="music-recommendation",
        name="data-exploration",
        config={
            "datasets": ["archive", "lastfm"],
            "focus": "audio_features_and_user_interactions"
        }
    )
    
    # Load Spotify Songs
    print("Loading Spotify songs dataset...")
    spotify_df = pd.read_csv("archive/spotify_songs.csv")
    print(f"Loaded {len(spotify_df)} songs")
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 
                     'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness']
    
    print("Audio features summary:")
    spotify_summary = spotify_df[audio_features].describe()
    print(spotify_summary)
    
    # Genre and artist distributions
    genre_counts = spotify_df['playlist_genre'].value_counts()
    print("Top 10 genres:")
    print(genre_counts.head(10))
    
    artist_counts = spotify_df['track_artist'].value_counts()
    print("Top 10 artists:")
    print(artist_counts.head(10))
    
    # Load Last.fm data
    print("Loading Last.fm dataset...")
    
    artists_df = pd.read_csv("hetrec2011-lastfm-2k/artists.dat", sep='\t', 
                            names=['id', 'name', 'url', 'pictureURL'])
    print(f"Loaded {len(artists_df)} artists")
    
    interactions_df = pd.read_csv("hetrec2011-lastfm-2k/user_artists.dat", sep='\t',
                                 names=['user_id', 'artist_id', 'weight'])
    print(f"Loaded {len(interactions_df)} user-artist interactions")
    
    friends_df = pd.read_csv("hetrec2011-lastfm-2k/user_friends.dat", sep='\t',
                           names=['user_id', 'friend_id'])
    print(f"Loaded {len(friends_df)} user friendships")
    
    # Statistics
    print("Last.fm statistics:")
    print(f"Users: {interactions_df['user_id'].nunique()}")
    print(f"Artists: {interactions_df['artist_id'].nunique()}")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Friendships: {len(friends_df)}")
    
    print("Interaction statistics:")
    print(f"Min weight: {interactions_df['weight'].min()}")
    print(f"Max weight: {interactions_df['weight'].max()}")
    print(f"Mean weight: {interactions_df['weight'].mean():.2f}")
    
    # User and artist activity
    user_activity = interactions_df.groupby('user_id')['weight'].sum()
    artist_popularity = interactions_df.groupby('artist_id')['weight'].sum()
    
    print("User activity:")
    print(f"Most active user: {user_activity.max()} listens")
    print(f"Average user activity: {user_activity.mean():.2f} listens")
    
    print("Artist popularity:")
    print(f"Most popular artist: {artist_popularity.max()} listens")
    print(f"Average artist popularity: {artist_popularity.mean():.2f} listens")
    
    # Log metrics to W&B
    wandb.log({
        "spotify_songs": len(spotify_df),
        "lastfm_artists": len(artists_df),
        "lastfm_interactions": len(interactions_df),
        "lastfm_users": interactions_df['user_id'].nunique(),
        "lastfm_friendships": len(friends_df),
        "avg_danceability": spotify_df['danceability'].mean(),
        "avg_energy": spotify_df['energy'].mean(),
        "avg_valence": spotify_df['valence'].mean(),
        "avg_tempo": spotify_df['tempo'].mean(),
        "avg_user_activity": user_activity.mean(),
        "max_user_activity": user_activity.max(),
        "avg_artist_popularity": artist_popularity.mean(),
        "max_artist_popularity": artist_popularity.max(),
        "unique_genres": spotify_df['playlist_genre'].nunique(),
        "unique_artists_spotify": spotify_df['track_artist'].nunique(),
        "avg_interaction_weight": interactions_df['weight'].mean()
    })
    
    # Log genre distribution
    for genre, count in genre_counts.head(10).items():
        wandb.log({f"genre_{genre}": count})
    
    print("Data exploration completed!")
    
    wandb.finish()
    
    return {
        'spotify_songs': len(spotify_df),
        'lastfm_users': interactions_df['user_id'].nunique(),
        'lastfm_artists': len(artists_df),
        'lastfm_interactions': len(interactions_df),
        'audio_features': audio_features,
        'genres': genre_counts.head(10).to_dict(),
        'artists': artist_counts.head(10).to_dict()
    }

if __name__ == "__main__":
    results = explore_datasets()
    print("Summary:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
