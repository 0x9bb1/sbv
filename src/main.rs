use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use qdrant_client::client::QdrantClient;
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::prelude::Value;

use qdrant_client::qdrant::{ScoredPoint};

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing_subscriber::fmt::time;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::graceful_shutdown::shutdown_signal;

mod embeddings;
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
        .with(
            tracing_subscriber::fmt::layer()
                .with_timer(time::LocalTime::rfc_3339())
        )
        .init();
    let client = Arc::new(qdrant::make_client().await?);

    let app = Router::new()
        .route("/", get(index))
        .route("/search", get(handle_search))
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

#[derive(Deserialize)]
struct Color {
    color: String,
}

async fn handle_search(
    State(client): State<Arc<QdrantClient>>,
    Query(payload): Query<Color>,
) -> Result<SearchResult, AppError> {
    let param = payload.color;
    let params = vec![param];
    let output = embeddings::encode(params).await?;
    let vector = &output[0];

    let res: Vec<ScoredPoint> = qdrant::search(&client, COLLECTION_NAME, vector, None).await?;

    Ok(SearchResult(res))
}

struct SearchResult(Vec<ScoredPoint>);

#[derive(Serialize)]
struct SearchResultItem {
    id: String,
    score: f32,
    payload: HashMap<String, Value>,
}

impl IntoResponse for SearchResult {
    fn into_response(self) -> Response {
        let items: Vec<SearchResultItem> = self
            .0
            .into_iter()
            .map(|point| {
                SearchResultItem {
                    id: match point.id {
                        None => "".to_string(),
                        Some(id) => match id.point_id_options {
                            None => "".to_string(),
                            Some(id) => match id {
                                PointIdOptions::Num(id) => id.to_string(),
                                PointIdOptions::Uuid(id) => id,
                            },
                        },
                    },
                    score: point.score,
                    payload: point.payload,
                }
            })
            .collect::<Vec<SearchResultItem>>();

        let body = Json(json!({
            "result": items,
        }));
        (StatusCode::OK, body).into_response()
    }
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": self.0.to_string(),
        }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
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
