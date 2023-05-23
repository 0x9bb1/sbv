use anyhow::Result;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::prelude::{Payload, PointStruct, SearchPoints};
use qdrant_client::qdrant::ScoredPoint;

use crate::embeddings;
use crate::entity::PayloadRequest;
use tracing::{event, instrument};
use uuid::Uuid;

const AK: &'static str = "qdrant_api_key";

const URL: &'static str =
    "https://f68bcdbc-a90e-4791-8358-b448e0335a86.ap-northeast-1-0.aws.cloud.qdrant.io:6334";

pub async fn make_client() -> Result<QdrantClient> {
    let mut config = QdrantClientConfig::from_url(URL);
    // using an env variable for the API KEY for example
    let api_key = std::env::var(AK).ok();

    if let Some(api_key) = api_key {
        config.set_api_key(&api_key);
    }
    let client = QdrantClient::new(Some(config)).await?;
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
    // dbg!(&search_result);

    event!(tracing::Level::INFO, "search took {:?}", start.elapsed());

    Ok(search_result.result)
}

#[instrument(name = "save", skip(client, req))]
pub async fn save(client: &QdrantClient, collection_name: &str, req: PayloadRequest) -> Result<()> {
    let start = std::time::Instant::now();

    let mut payload: Payload = Payload::new();
    payload.insert(req.key, req.value.clone());
    let id = Uuid::new_v4();
    let params = vec![req.value];
    let output = embeddings::encode(params).await?;

    event!(tracing::Level::INFO, "encode took {:?}", start.elapsed());

    let start = std::time::Instant::now();

    let vector = &output[0];
    let points = vec![PointStruct::new(id.to_string(), vector.clone(), payload)];

    let _response = client.upsert_points(collection_name, points, None).await?;
    // dbg!(response);

    event!(tracing::Level::INFO, "save took {:?}", start.elapsed());

    Ok(())
}
