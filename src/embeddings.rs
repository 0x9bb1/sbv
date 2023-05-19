use anyhow::Result;
use lazy_static::lazy_static;
use rust_bert::pipelines::sentence_embeddings::{
    Embedding, SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use std::sync::{Arc, Mutex};
use tracing::event;

lazy_static! {
    static ref MODEL: Arc<Mutex<SentenceEmbeddingsModel>> = {
        let output = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()
            .unwrap();

        Arc::new(Mutex::new(output))
    };
}

pub async fn encode(texts: Vec<String>) -> Result<Vec<Embedding>> {
    let output = tokio::task::spawn_blocking(move || {
        let span = tracing::info_span!("encode");
        let _enter = span.enter();

        let start = std::time::Instant::now();

        let texts_slice: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let output = MODEL
            .lock()
            .unwrap()
            .encode(texts_slice.as_slice())
            .unwrap();

        event!(tracing::Level::INFO, "encoded in {:?}", start.elapsed());

        output
    })
    .await?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[tokio::test]
    async fn test_create_model() -> Result<()> {
        let _ = tokio::task::spawn_blocking(move || {
            let output = MODEL.lock().unwrap().encode(&["red"]).unwrap();
            dbg!(output);
        })
        .await?;

        Ok(())
    }
}
