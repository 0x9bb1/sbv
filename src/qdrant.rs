use anyhow::{Ok, Result};
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::prelude::{Payload, PointStruct, SearchPoints};
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::CreateCollection;
use qdrant_client::qdrant::Distance;
use qdrant_client::qdrant::ScoredPoint;
use qdrant_client::qdrant::VectorParams;
use qdrant_client::qdrant::VectorsConfig;

use crate::embeddings;
use crate::entity::PayloadRequest;
use tracing::log::debug;
use tracing::{event, instrument};
use uuid::Uuid;

const AK: &'static str = "QDRANT_API_KEY";

pub async fn make_client() -> Result<QdrantClient> {
    let url = std::env::var("QDRANT_URL")?;
    let mut config = QdrantClientConfig::from_url(url.as_str());
    // using an env variable for the API KEY for example
    let api_key = std::env::var(AK).ok();

    if let Some(api_key) = api_key {
        config.set_api_key(&api_key);
    }
    let client = QdrantClient::new(Some(config))?;
    Ok(client)
}

#[instrument(name = "search", skip(client, vector, limit))]
pub async fn search(
    client: &QdrantClient,
    collection_name: &str,
    vector: &Vec<f32>,
    limit: Option<u64>,
) -> Result<Vec<ScoredPoint>> {
    let start = std::time::Instant::now();

    let search_result = client
        .search_points(&SearchPoints {
            collection_name: collection_name.into(),
            vector: vector.to_vec(),
            filter: None,
            limit: limit.unwrap_or(10),
            with_vectors: None,
            with_payload: Some(true.into()),
            params: None,
            score_threshold: None,
            offset: None,
            ..Default::default()
        })
        .await?;
    debug!("search_result: {:?}", &search_result);

    event!(tracing::Level::INFO, "search took {:?}", start.elapsed());

    Ok(search_result.result)
}

#[instrument(name = "save", skip(client, req))]
pub async fn save(client: &QdrantClient, collection_name: &str, req: PayloadRequest) -> Result<()> {
    let output = embeddings::encode(req.value.clone()).await?;

    let mut points = Vec::new();
    for (idx, vector) in output.iter().enumerate() {
        let id = Uuid::new_v4();
        let mut payload: Payload = Payload::new();
        let value = match req.value.get(idx) {
            Some(value) => value.clone(),
            None => String::from("unknown"),
        };
        payload.insert(req.key.clone(), value);

        let point = PointStruct::new(id.to_string(), vector.clone(), payload);
        points.push(point);
    }

    let response = client
        .upsert_points_blocking(collection_name, None, points, None)
        .await?;
    debug!("response: {:?}", &response);

    Ok(())
}

pub(crate) async fn create(client: &QdrantClient, collection_name: &str) -> Result<()> {
    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 1536,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

    Ok(())
}
