use std::sync::Arc;

use anyhow::{anyhow, Result};
use axum::extract::{Query, State};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use entity::QueryRequest;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::ScoredPoint;
use serde_json::json;

use crate::entity::{AppError, PayloadRequest, ResultData, SearchResultItem};
use crate::graceful_shutdown::shutdown_signal;

mod embeddings;
mod entity;
mod graceful_shutdown;
mod qdrant;

const COLLECTION_NAME: &str = "search_test";

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_thread_ids(true)
        .with_thread_names(true)
        .init();

    let client = Arc::new(qdrant::make_client().await?);

    let app = Router::new()
        .route("/", get(index))
        .route("/search", get(handle_search))
        .route("/save", post(handle_save))
        .route("/create", put(handle_create))
        .with_state(client);

    tracing::info!("starting server...");

    axum::Server::bind(&"0.0.0.0:8000".parse().unwrap())
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    Ok(())
}

async fn index() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Hello, sbv!",
    }))
}

async fn handle_search(
    State(client): State<Arc<QdrantClient>>,
    Query(payload): Query<QueryRequest>,
) -> Result<Json<ResultData<Vec<SearchResultItem>>>, AppError> {
    let param = embeddings::extract_key(payload.value).await?;
    let output: Vec<Vec<f32>> = embeddings::encode(param).await?;

    let weights = calculate_weights(output.len(), 0.8)?;
    let vector: Vec<f32> = weighted_average(output, weights)?;
    tracing::info!("Vector size:[{}]", &vector.len());

    let res: Vec<ScoredPoint> =
        qdrant::search(&client, COLLECTION_NAME, &vector, payload.limit).await?;

    let items = res
        .into_iter()
        .map(|p| SearchResultItem::new_from_scored_point(p))
        .collect::<Vec<SearchResultItem>>();
    Ok(Json(ResultData::new(items)))
}

async fn handle_save(
    State(client): State<Arc<QdrantClient>>,
    Json(payload): Json<PayloadRequest>,
) -> Result<Json<ResultData<()>>, AppError> {
    let _result = qdrant::save(&client, COLLECTION_NAME, payload).await?;

    Ok(Json(ResultData::new(())))
}

async fn handle_create(
    State(client): State<Arc<QdrantClient>>,
) -> Result<Json<ResultData<()>>, AppError> {
    let _result = qdrant::create(&client, COLLECTION_NAME).await?;

    Ok(Json(ResultData::new(())))
}

fn weighted_average(vectors: Vec<Vec<f32>>, weights: Vec<f32>) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(anyhow!("Vector is empty."));
    }

    let dim = vectors[0].len();
    let mut result_vector = vec![0.0; dim];

    for (vector, weight) in vectors.iter().zip(weights.iter()) {
        if vector.len() != dim {
            return Err(anyhow!("All vectors must have the same dimension"));
        }
        for (i, &value) in vector.iter().enumerate() {
            result_vector[i] += value * weight;
        }
    }

    let total_weight: f32 = weights.iter().sum();
    if total_weight != 0.0 {
        for value in result_vector.iter_mut() {
            *value /= total_weight;
        }
    } else {
        return Err(anyhow!("Total weight cannot be zero"));
    }

    Ok(result_vector)
}

fn calculate_weights(size: usize, decay_rate: f32) -> Result<Vec<f32>> {
    Ok((0..size).map(|i| decay_rate.powi(i as i32)).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn vec_init() {
        let v = vec![12.; 10];
        assert_eq!(v.len(), 10usize);
    }
}
