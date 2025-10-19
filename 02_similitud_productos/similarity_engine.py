import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional
from unicodedata import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: Optional[int] = None
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True
        )
        self.item_vectors = None
        self.items_df = None
        self.fitted = False
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fit(
        self,
        items_df: pd.DataFrame,
        title_column: str = 'ITE_ITEM_TITLE'
    ) -> 'SimilarityEngine':
        """
        Entrena el vectorizador TF-IDF con un dataset (vocabulario).
        No establece el catálogo de búsqueda.
        """
        if title_column not in items_df.columns:
            raise ValueError(f"Title column '{title_column}' not found")
        
        titles_clean = items_df[title_column].apply(self.preprocess_text)
        self.vectorizer.fit(titles_clean)
        self.fitted = True
        return self
    
    def set_catalog(
        self,
        catalog_df: pd.DataFrame,
        title_column: str = 'ITE_ITEM_TITLE',
        id_column: str = 'ITE_ITEM_ID'
    ) -> 'SimilarityEngine':
        """
        Establece el catálogo de productos sobre el que se buscarán similitudes.
        Requiere que el vectorizador ya esté entrenado (fit).
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        self.items_df = catalog_df.copy()
        
        if id_column not in self.items_df.columns:
            self.items_df[id_column] = self.items_df.index.astype(str)
        
        if title_column not in self.items_df.columns:
            raise ValueError(f"Title column '{title_column}' not found")
        
        self.items_df['title_clean'] = self.items_df[title_column].apply(
            self.preprocess_text
        )
        
        self.item_vectors = self.vectorizer.transform(
            self.items_df['title_clean']
        )
        
        return self
    
    def find_similar(
        self,
        query: str,
        top_k: int = 2,
        return_scores: bool = True
    ) -> List[dict]:
        
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        query_clean = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query_clean])
        
        similarities = cosine_similarity(query_vector, self.item_vectors).ravel()
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            item_data = self.items_df.iloc[idx]
            result = {
                'item_id': str(item_data['ITE_ITEM_ID']),
                'title': item_data['ITE_ITEM_TITLE'],
            }
            if return_scores:
                result['similarity_score'] = float(similarities[idx])
            
            results.append(result)
        
        return results
    
    def find_similar_batch(
        self,
        queries: List[str],
        top_k: int = 2
    ) -> List[List[dict]]:
        return [self.find_similar(q, top_k) for q in queries]
    
    def get_item_similarity(
        self,
        item_id_1: str,
        item_id_2: str
    ) -> float:
        
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        idx1 = self.items_df[self.items_df['ITE_ITEM_ID'] == item_id_1].index
        idx2 = self.items_df[self.items_df['ITE_ITEM_ID'] == item_id_2].index
        
        if len(idx1) == 0 or len(idx2) == 0:
            raise ValueError("Item ID not found")
        
        idx1, idx2 = idx1[0], idx2[0]
        
        sim = cosine_similarity(
            self.item_vectors[idx1:idx1+1],
            self.item_vectors[idx2:idx2+1]
        )[0, 0]
        
        return float(sim)
    
    def save(self, model_path: str):
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'item_vectors': self.item_vectors,
            'items_df': self.items_df
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, model_path: str) -> 'SimilarityEngine':
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        engine = cls()
        engine.vectorizer = model_data['vectorizer']
        engine.item_vectors = model_data['item_vectors']
        engine.items_df = model_data['items_df']
        engine.fitted = True
        
        return engine


def train_and_save_model(
    data_path: str = 'data/items_titles.csv',
    model_path: str = 'models/similarity_model.pkl'
) -> SimilarityEngine:
    
    df = pd.read_csv(data_path)
    
    engine = SimilarityEngine()
    engine.fit(df)
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    engine.save(model_path)
    
    print(f"Model trained and saved to {model_path}")
    print(f"Catalog size: {len(df)} items")
    print(f"Vocabulary size: {len(engine.vectorizer.get_feature_names_out())}")
    
    return engine


if __name__ == "__main__":
    engine = train_and_save_model()
    
    test_query = "tenis nike branco"
    results = engine.find_similar(test_query, top_k=3)
    
    print(f"\n\nTest query: '{test_query}'")
    print("Top 3 similar items:")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['similarity_score']:.4f}] {result['title']}")
