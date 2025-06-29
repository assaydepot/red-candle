use magnus::{class, function, method, prelude::*, Error, RModule, Float, RArray};
use candle_transformers::models::bert::{BertModel, Config};
use candle_core::{Device, Tensor, Result as CandleResult, IndexOp};
use candle_nn::{VarBuilder, ops::sigmoid, Linear};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer, EncodeInput};
use std::thread;

#[magnus::wrap(class = "Candle::Reranker", free_immediately, size)]
pub struct Reranker {
    model: BertModel,
    tokenizer: Tokenizer,
    linear_layer: Linear,
    device: Device,
}

impl Reranker {
    pub fn new(model_id: String) -> Result<Self, Error> {
        let device = Device::Cpu;
        let handle = thread::spawn(move || {
            let api = Api::new().unwrap();
            let repo = api.repo(Repo::new(model_id, RepoType::Model));
            let config_filename = repo.get("config.json").unwrap();
            let tokenizer_filename = repo.get("tokenizer.json").unwrap();
            let weights_filename = repo.get("model.safetensors").unwrap();
            let config = std::fs::read_to_string(config_filename).unwrap();
            let config: Config = serde_json::from_str(&config).unwrap();

            let mut tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], candle_core::DType::F32, &device).unwrap()
            };
            let model = BertModel::load(vb.pp("bert"), &config).unwrap();
            let linear_layer = candle_nn::linear(config.hidden_size, 1, vb.pp("cls.predictions.transform.dense")).unwrap();
            (model, tokenizer, linear_layer)
        });
        let (model, tokenizer, linear_layer) = handle.join().unwrap();
        Ok(Self { model, tokenizer, linear_layer, device: Device::Cpu })
    }

    pub fn rerank(&self, query: String, documents: RArray) -> Result<RArray, Error> {
        let documents: Vec<String> = documents.to_vec()?;
        let query_and_docs: Vec<EncodeInput> = documents
            .iter()
            .map(|d| (query.clone(), d.clone()).into())
            .collect();

        let encodings = self.tokenizer.encode_batch(query_and_docs, true).unwrap();
        let token_ids = encodings
            .iter()
            .map(|e| Ok(e.get_ids().to_vec()))
            .collect::<CandleResult<Vec<_>>>().unwrap();

        let token_ids = Tensor::new(token_ids, &self.device).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        let attention_mask = token_ids.ne(0u32).unwrap();
        let logits = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).unwrap();
        let cls_token = logits.i((.., 0)).unwrap();
        let scores = sigmoid(&cls_token.apply(&self.linear_layer).unwrap().squeeze(1).unwrap()).unwrap();
        let scores_vec: Vec<f32> = scores.to_vec1().unwrap();

        let mut ranked_docs: Vec<(String, f32)> = documents.into_iter().zip(scores_vec).collect();
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let result_array = RArray::new();
        for (doc, score) in ranked_docs {
            let pair = RArray::new();
            pair.push(doc).unwrap();
            pair.push(Float::from_f64(score as f64)).unwrap();
            result_array.push(pair).unwrap();
        }
        Ok(result_array)
    }
}

pub fn init(rb_candle: RModule) -> Result<(), Error> {
    let c_reranker = rb_candle.define_class("Reranker", class::object())?;
    c_reranker.define_singleton_method("new", function!(Reranker::new, 1))?;
    c_reranker.define_method("rerank", method!(Reranker::rerank, 2))?;
    Ok(())
}