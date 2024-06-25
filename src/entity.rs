use std::collections::HashMap;

use axum::response::IntoResponse;
use axum::response::Response;
use axum::Json;
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::prelude::Value;
use qdrant_client::qdrant::ScoredPoint;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct PayloadRequest {
    pub key: String,
    pub value: Vec<String>,
    pub limit: Option<u64>,
}

#[derive(Deserialize, Debug)]
pub struct QueryRequest {
    pub key: String,
    pub value: String,
    pub limit: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct ResultData<T: Serialize> {
    data: Option<T>,
    msg: String,
    success: bool,
}

impl<T> ResultData<T>
where
    T: Serialize,
{
    pub(crate) fn new(data: T) -> Self {
        Self {
            data: Some(data),
            msg: "".to_string(),
            success: true,
        }
    }

    pub(crate) fn new_with_msg(msg: String) -> Self {
        Self {
            data: None,
            msg,
            success: false,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct SearchResultItem {
    id: String,
    score: f32,
    payload: HashMap<String, Value>,
}

impl SearchResultItem {
    pub(crate) fn new_from_scored_point(point: ScoredPoint) -> Self {
        Self {
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
    }
}

// Make our own error that wraps `anyhow::Error`.
pub(crate) struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let res: ResultData<()> = ResultData::new_with_msg(self.0.to_string());
        Json(res).into_response()
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
