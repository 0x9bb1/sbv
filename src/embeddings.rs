use anyhow::{Ok, Result};
use async_openai::types::{
    ChatCompletionRequestSystemMessageArgs, CreateChatCompletionRequestArgs,
    CreateEmbeddingRequestArgs,
};
use async_openai::Client;
use tracing::event;

pub async fn encode(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    tracing::info!("text size:[{}]", &texts.len());

    let start = std::time::Instant::now();

    let client = build_client()?;

    let request = CreateEmbeddingRequestArgs::default()
        // .model("text-embedding-ada-002")
        .model("text-embedding-3-large")
        .dimensions(1536 as u32)
        .input(texts)
        .build()?;

    let response = client.embeddings().create(request).await?;

    event!(tracing::Level::INFO, "encoded in {:?}", start.elapsed());

    let mut vectors = Vec::new();
    for ele in response.data {
        vectors.push(ele.embedding);
    }

    Ok(vectors)
}

pub async fn extract_key(source: String) -> Result<Vec<String>> {
    let client = build_client()?;
    let prompt = formatted_prompt(&source)?;
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages([ChatCompletionRequestSystemMessageArgs::default()
            .content(prompt)
            .build()?
            .into()])
        .max_tokens(512_u32)
        .build()?;

    let response = client.chat().create(request).await?;

    for choice in response.choices {
        let content = choice.message.content;
        let index = choice.index;
        let role = choice.message.role;
        tracing::info!("{}: Role: {}  Content: {:?}", index, role, content);
        if let Some(content) = content {
            let words: Vec<String> = content.split(' ').map(|word| word.to_string()).collect();
            return Ok(words);
        }
    }
    Ok(vec![source.clone()])
}

fn formatted_prompt(action: &str) -> Result<String> {
    Ok(format!(
        r"
    用户输入了一个搜索查询：{}。请识别并输出以下内容：
    1. 主要的商品类别
    2. 查询中的所有修饰词
    不需要过程。比如输入 条纹衬衫 只需要输出:
    衬衫,条纹
    ",
        action
    ))
}

fn build_client() -> Result<Client<async_openai::config::OpenAIConfig>, anyhow::Error> {
    let proxy = reqwest::Proxy::all("http://10.10.54.49:10809")?;
    let http_client = reqwest::Client::builder().proxy(proxy).build()?;
    let mut client = Client::new();
    client = Client::with_http_client(client, http_client);
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::build_client;

    #[tokio::test]
    async fn models() {
        dotenv::dotenv().ok();

        let client = build_client().unwrap();

        let model_list = client.models().list().await.unwrap();

        println!("List of models:\n {:#?}", model_list.data);
    }
}
