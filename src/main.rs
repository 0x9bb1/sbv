use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Query, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::ScoredPoint;
use serde_json::json;
use tracing_subscriber::fmt::time;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::entity::{AppError, PayloadRequest, ResultData, SearchResultItem};
use crate::graceful_shutdown::shutdown_signal;

mod embeddings;
mod entity;
mod graceful_shutdown;
mod qdrant;

const COLLECTION_NAME: &str = "test";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "search_by_vector=trace".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_timer(time::LocalTime::rfc_3339()))
        .init();
    let client = Arc::new(qdrant::make_client().await?);

    let app = Router::new()
        .route("/", get(index))
        .route("/search", get(handle_search))
        .route("/save", post(handle_save))
        .with_state(client);

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    Ok(())
}

async fn index() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Hello, World!",
    }))
}

async fn handle_search(
    State(client): State<Arc<QdrantClient>>,
    Query(payload): Query<PayloadRequest>,
) -> Result<Json<ResultData<Vec<SearchResultItem>>>, AppError> {
    let param = payload.value;
    let params = vec![param];
    let output = embeddings::encode(params).await?;
    let vector = &output[0];

    let res: Vec<ScoredPoint> =
        qdrant::search(&client, COLLECTION_NAME, vector, payload.limit).await?;

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

#[cfg(test)]
mod tests {
    #[test]
    fn vec_init() {
        let v = vec![12.; 10];
        dbg!(&v);
        assert_eq!(v.len(), 10 as usize);
    }
}
