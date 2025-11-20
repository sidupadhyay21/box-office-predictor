"""
Data Preprocessing for TMDB Box Office Prediction
Handles missing values, JSON parsing, feature engineering, and data transformations
Run in terminal: python3 preprocess_data.py
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class TMDBPreprocessor:
    """Preprocessor for TMDB movie data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def parse_json_column(self, df, column):
        """Safely parse JSON string columns"""
        def safe_parse(x):
            if pd.isna(x) or x == '':
                return []
            try:
                return json.loads(x.replace("'", '"'))
            except:
                return []
        return df[column].apply(safe_parse)
    
    def extract_collection_features(self, df):
        """Extract features from belongs_to_collection"""
        df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
        
        def get_collection_name(x):
            if pd.isna(x) or x == '':
                return 'none'
            try:
                data = json.loads(x.replace("'", '"'))
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('name', 'none')
                elif isinstance(data, dict):
                    return data.get('name', 'none')
            except:
                pass
            return 'none'
        
        df['collection_name'] = df['belongs_to_collection'].apply(get_collection_name)
        return df
    
    def extract_genre_features(self, df):
        """Extract genre information"""
        genres_list = self.parse_json_column(df, 'genres')
        
        # Number of genres
        df['num_genres'] = genres_list.apply(len)
        
        # Extract genre names
        df['genre_names'] = genres_list.apply(
            lambda x: '|'.join([g.get('name', '') for g in x]) if x else ''
        )
        
        # Create binary features for top genres
        all_genres = []
        for genres in genres_list:
            all_genres.extend([g.get('name', '') for g in genres])
        
        top_genres = pd.Series(all_genres).value_counts().head(10).index
        for genre in top_genres:
            df[f'genre_{genre.lower().replace(" ", "_")}'] = genres_list.apply(
                lambda x: 1 if any(g.get('name') == genre for g in x) else 0
            )
        
        return df
    
    def extract_cast_crew_features(self, df):
        """Extract cast and crew information"""
        # Cast features
        cast_list = self.parse_json_column(df, 'cast')
        df['num_cast'] = cast_list.apply(len)
        df['has_cast_info'] = (df['num_cast'] > 0).astype(int)
        
        # Extract top 3 actors
        for i in range(3):
            df[f'actor_{i+1}'] = cast_list.apply(
                lambda x: x[i].get('name', 'unknown') if len(x) > i else 'unknown'
            )
        
        # Crew features
        crew_list = self.parse_json_column(df, 'crew')
        df['num_crew'] = crew_list.apply(len)
        
        # Extract director
        def get_director(crew):
            if not crew:
                return 'unknown'
            for person in crew:
                if person.get('job') == 'Director':
                    return person.get('name', 'unknown')
            return 'unknown'
        
        df['director'] = crew_list.apply(get_director)
        
        return df
    
    def extract_production_features(self, df):
        """Extract production company and country features"""
        # Production companies
        companies_list = self.parse_json_column(df, 'production_companies')
        df['num_production_companies'] = companies_list.apply(len)
        df['has_production_info'] = (df['num_production_companies'] > 0).astype(int)
        
        # Production countries
        countries_list = self.parse_json_column(df, 'production_countries')
        df['num_production_countries'] = countries_list.apply(len)
        
        # Primary country
        df['primary_country'] = countries_list.apply(
            lambda x: x[0].get('iso_3166_1', 'unknown') if x else 'unknown'
        )
        
        # Spoken languages
        languages_list = self.parse_json_column(df, 'spoken_languages')
        df['num_spoken_languages'] = languages_list.apply(len)
        
        return df
    
    def extract_keywords_features(self, df):
        """Extract keyword features"""
        keywords_list = self.parse_json_column(df, 'Keywords')
        df['num_keywords'] = keywords_list.apply(len)
        df['has_keywords'] = (df['num_keywords'] > 0).astype(int)
        
        return df
    
    def extract_date_features(self, df):
        """Extract features from release_date"""
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day'] = df['release_date'].dt.day
        df['release_dayofweek'] = df['release_date'].dt.dayofweek
        df['release_quarter'] = df['release_date'].dt.quarter
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['release_season'] = df['release_month'].apply(
            lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
        )
        
        # Holiday releases (US holidays)
        def is_holiday_season(month, day):
            # Summer blockbuster season
            if month in [5, 6, 7]:
                return 1
            # Holiday season
            if month in [11, 12]:
                return 1
            return 0
        
        df['is_holiday_season'] = df.apply(
            lambda row: is_holiday_season(row['release_month'], row['release_day']), axis=1
        )
        
        # Weekend release
        df['is_weekend_release'] = df['release_dayofweek'].isin([4, 5]).astype(int)
        
        return df
    
    def handle_budget_revenue(self, df, is_train=True):
        """Handle budget and revenue features"""
        # Budget features
        df['has_budget'] = (df['budget'] > 0).astype(int)
        df['budget_log'] = np.log1p(df['budget'])
        
        # Create budget categories
        df['budget_category'] = pd.cut(
            df['budget'],
            bins=[-1, 0, 1000000, 10000000, 50000000, np.inf],
            labels=['none', 'low', 'medium', 'high', 'very_high']
        )
        
        # Revenue (target variable) - only for training data
        if is_train and 'revenue' in df.columns:
            df['revenue_log'] = np.log1p(df['revenue'])
        
        return df
    
    def handle_text_features(self, df):
        """Extract features from text columns"""
        # Overview
        df['overview'] = df['overview'].fillna('')
        df['overview_length'] = df['overview'].str.len()
        df['overview_word_count'] = df['overview'].str.split().str.len()
        df['has_overview'] = (df['overview'] != '').astype(int)
        
        # Tagline
        df['tagline'] = df['tagline'].fillna('')
        df['tagline_length'] = df['tagline'].str.len()
        df['has_tagline'] = (df['tagline'] != '').astype(int)
        
        # Title
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        # Homepage
        df['has_homepage'] = df['homepage'].notna().astype(int)
        
        return df
    
    def handle_other_features(self, df):
        """Handle remaining features"""
        # Runtime
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        df['runtime'] = df['runtime'].replace(0, df['runtime'].median())
        df['runtime_log'] = np.log1p(df['runtime'])
        
        # Popularity
        df['popularity_log'] = np.log1p(df['popularity'])
        
        # Status
        if 'status' in df.columns:
            df['is_released'] = (df['status'] == 'Released').astype(int)
        
        # Poster
        df['has_poster'] = df['poster_path'].notna().astype(int)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_cols = ['original_language', 'primary_country', 'budget_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].apply(
                            lambda x: self.label_encoders[col].transform([str(x)])[0]
                            if str(x) in self.label_encoders[col].classes_
                            else -1
                        )
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        # Budget and popularity interaction
        df['budget_popularity_interaction'] = df['budget_log'] * df['popularity_log']
        
        # Budget per genre
        if 'num_genres' in df.columns:
            df['budget_per_genre'] = df['budget'] / (df['num_genres'] + 1)
        
        # Cast to crew ratio
        df['cast_crew_ratio'] = df['num_cast'] / (df['num_crew'] + 1)
        
        return df
    
    def preprocess(self, df, is_train=True, fit=True):
        """
        Main preprocessing pipeline
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        is_train : bool
            Whether this is training data (has revenue column)
        fit : bool
            Whether to fit encoders/scalers (True for train, False for test)
        
        Returns:
        --------
        DataFrame : Preprocessed dataframe
        """
        print("Starting preprocessing...")
        df = df.copy()
        
        # 1. Extract collection features
        print("Extracting collection features...")
        df = self.extract_collection_features(df)
        
        # 2. Extract genre features
        print("Extracting genre features...")
        df = self.extract_genre_features(df)
        
        # 3. Extract cast and crew features
        print("Extracting cast/crew features...")
        df = self.extract_cast_crew_features(df)
        
        # 4. Extract production features
        print("Extracting production features...")
        df = self.extract_production_features(df)
        
        # 5. Extract keyword features
        print("Extracting keyword features...")
        df = self.extract_keywords_features(df)
        
        # 6. Extract date features
        print("Extracting date features...")
        df = self.extract_date_features(df)
        
        # 7. Handle budget and revenue
        print("Handling budget/revenue...")
        df = self.handle_budget_revenue(df, is_train=is_train)
        
        # 8. Handle text features
        print("Extracting text features...")
        df = self.handle_text_features(df)
        
        # 9. Handle other features
        print("Handling other features...")
        df = self.handle_other_features(df)
        
        # 10. Encode categorical features
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=fit)
        
        # 11. Create interaction features
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Preprocessing complete!")
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns to use for modeling"""
        # Numeric features
        numeric_features = [
            'budget', 'budget_log', 'has_budget',
            'popularity', 'popularity_log',
            'runtime', 'runtime_log',
            'num_genres', 'num_cast', 'num_crew', 'num_production_companies',
            'num_production_countries', 'num_spoken_languages', 'num_keywords',
            'release_year', 'release_month', 'release_day', 'release_dayofweek',
            'release_quarter', 'release_season',
            'overview_length', 'overview_word_count', 'tagline_length',
            'title_length', 'title_word_count',
            'has_collection', 'has_cast_info', 'has_production_info',
            'has_keywords', 'has_overview', 'has_tagline', 'has_homepage',
            'has_poster', 'is_holiday_season', 'is_weekend_release',
            'budget_popularity_interaction', 'budget_per_genre', 'cast_crew_ratio'
        ]
        
        # Encoded categorical features
        categorical_features = [
            'original_language_encoded', 'primary_country_encoded',
            'budget_category_encoded'
        ]
        
        # Genre binary features
        genre_features = [col for col in numeric_features if col.startswith('genre_')]
        
        return numeric_features + categorical_features

def main():
    """Example usage"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Initialize preprocessor
    preprocessor = TMDBPreprocessor()
    
    # Preprocess training data
    print("\n" + "="*50)
    print("PREPROCESSING TRAINING DATA")
    print("="*50)
    train_processed = preprocessor.preprocess(train_df, is_train=True, fit=True)
    
    # Preprocess test data (using fitted encoders)
    print("\n" + "="*50)
    print("PREPROCESSING TEST DATA")
    print("="*50)
    test_processed = preprocessor.preprocess(test_df, is_train=False, fit=False)
    
    # Save processed data
    print("\nSaving processed data...")
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
    
    # Display info
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Training data shape: {train_processed.shape}")
    print(f"Test data shape: {test_processed.shape}")
    print(f"\nNew columns created: {len(train_processed.columns) - len(train_df.columns)}")
    
    print("\nSample of processed features:")
    feature_cols = preprocessor.get_feature_columns()
    available_features = [col for col in feature_cols if col in train_processed.columns]
    print(train_processed[available_features[:10]].head())
    
    print("\nTarget variable (revenue_log) statistics:")
    if 'revenue_log' in train_processed.columns:
        print(train_processed['revenue_log'].describe())
    
    print("\nFiles saved:")
    print("  - train_processed.csv")
    print("  - test_processed.csv")

if __name__ == "__main__":
    main()
