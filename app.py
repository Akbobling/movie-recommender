from fastapi import FastAPI
from pipeline import MovieRecommenderPipeline

app = FastAPI(title="Movie Recommender API")

# 启动时加载模型（全局单例，避免每次请求重复加载）
pipeline = MovieRecommenderPipeline.load('recommendation_pipeline.pkl')


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    """
    为用户生成电影推荐
    示例：GET /recommend/1?n=10
    """
    recs = pipeline.recommend(user_id, n_recommendations=n)
    return recs.to_dict(orient='records')


@app.get("/health")
def health():
    """
    健康检查接口，用于监控和负载均衡
    """
    return {
        "status": "ok",
        "model": "svd",
        "components": pipeline.n_factors
    }


@app.get("/predict")
def predict_rating(user_id: int, movie_id: int):
    """
    预测单个用户对单部电影的评分
    示例：GET /predict?user_id=1&movie_id=296
    """
    score = pipeline.predict_rating(user_id, movie_id)
    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "predicted_rating": round(score, 3)
    }