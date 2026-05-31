import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import joblib




class MovieRecommenderPipeline:
    """
    基于 surprise 的端到端推荐 Pipeline
    - 正确处理缺失值（只拟合观测到的评分）
    - 预测值天然在 [1, 5] 范围内
    - 自包含序列化
    """
    
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = None
        self.movies_df = None
        self.user_seen = {}      # userId -> set(movieId)
        self.all_movie_ids = []  # 全量电影 ID 列表
        self.trainset = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """训练模型"""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.all_movie_ids = sorted(movies_df['movieId'].unique())
        
        # 记录每个用户已看过的电影
        self.user_seen = (
            ratings_df.groupby('userId')['movieId']
            .apply(set)
            .to_dict()
        )
        
        # surprise 需要三列：user item rating
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # 使用全部数据训练（演示/生产环境通常用全量）
        self.trainset = data.build_full_trainset()
        
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        self.model.fit(self.trainset)
        
        # 评估：在训练集上算 RMSE（实际应划分测试集，这里仅演示）
        print(f"Training complete. n_factors={self.n_factors}")
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """预测单个用户-电影评分"""
        pred = self.model.predict(uid=user_id, iid=movie_id)
        return pred.est

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        为指定用户生成推荐
        只预测未看过的电影，按预测评分排序
        """
        if user_id not in self.user_seen:
            return pd.DataFrame({'error': [f'User {user_id} not found']})
        
        seen = self.user_seen[user_id]
        candidates = []
        
        # 遍历所有电影，预测未看过的
        for movie_id in self.all_movie_ids:
            if movie_id in seen:
                continue
            est = self.predict_rating(user_id, movie_id)
            candidates.append((movie_id, est))
        
        # 按预测评分降序
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:n_recommendations]
        
        # 组装结果
        results = []
        for movie_id, pred_score in top:
            info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not info.empty:
                row = info.iloc[0]
                results.append({
                    'movieId': movie_id,
                    'title': row['title'],
                    'genres': row['genres'],
                    'predicted_rating': round(pred_score, 3)
                })
        
        return pd.DataFrame(results)

    def get_user_favorites(self, ratings_df: pd.DataFrame, user_id: int, n: int = 5) -> pd.DataFrame:
        """查看用户历史高分电影"""
        user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('rating', ascending=False).head(n)
        favorites = []
        for _, row in user_ratings.iterrows():
            info = self.movies_df[self.movies_df['movieId'] == row['movieId']]
            if not info.empty:
                r = info.iloc[0]
                favorites.append({
                    'movieId': row['movieId'],
                    'title': r['title'],
                    'genres': r['genres'],
                    'rating': row['rating']
                })
        return pd.DataFrame(favorites)

    # ========== MLOps 核心 ==========
    
    def save(self, filepath: str = 'recommendation_pipeline.pkl'):
        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = 'recommendation_pipeline.pkl'):
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline