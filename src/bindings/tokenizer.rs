//! Python bindings for Tokenizer class

use crate::encoding::Encoding;
use crate::huggingface::{HuggingFaceTokenizer, ChatTemplateResult};
use super::encoding::{PyEncoding, PyBatchEncoding};
use super::components::{PyNormalizer, PyPreTokenizer, PyPostProcessor, PyDecoder};
use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;

/// Python-exposed Tokenizer class
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: HuggingFaceTokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = HuggingFaceTokenizer::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (repo_id, revision = None, local_files_only = false))]
    fn from_pretrained(repo_id: &str, revision: Option<&str>, local_files_only: bool) -> PyResult<Self> {
        let inner = HuggingFaceTokenizer::from_pretrained_with_options(repo_id, revision, local_files_only)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (
        text,
        text_pair = None,
        add_special_tokens = true,
        padding = None,
        truncation = false,
        max_length = None,
        stride = 0,
        return_attention_mask = true,
        return_token_type_ids = true,
        return_offsets_mapping = false,
        return_special_tokens_mask = false
    ))]
    fn __call__(
        &self,
        text: pyo3::Bound<'_, pyo3::types::PyAny>,
        text_pair: Option<pyo3::Bound<'_, pyo3::types::PyAny>>,
        add_special_tokens: bool,
        padding: Option<&str>,
        truncation: bool,
        max_length: Option<usize>,
        stride: usize,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        return_offsets_mapping: bool,
        return_special_tokens_mask: bool,
    ) -> PyResult<PyBatchEncoding> {
        if let Ok(list) = text.extract::<Vec<String>>() {
            let pairs: Option<Vec<String>> = text_pair.and_then(|p| p.extract().ok());

            let mut result_encodings: Vec<Encoding> = if let Some(pair_list) = pairs {
                list.iter()
                    .zip(pair_list.iter())
                    .map(|(a, b)| {
                        if add_special_tokens {
                            self.inner.encode_pair_to_encoding(a, b)
                        } else {
                            let ids_a = self.inner.encode(a);
                            let ids_b = self.inner.encode(b);
                            let mut enc = Encoding::from_ids(
                                ids_a.clone(),
                                ids_a.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                            );
                            let enc_b = Encoding::from_ids(
                                ids_b.clone(),
                                ids_b.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                            );
                            enc.merge(enc_b, 1);
                            enc
                        }
                    })
                    .collect()
            } else {
                list.iter()
                    .map(|t| {
                        if add_special_tokens {
                            self.inner.encode_to_encoding(t)
                        } else {
                            let ids = self.inner.encode(t);
                            Encoding::from_ids(
                                ids.clone(),
                                ids.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                            )
                        }
                    })
                    .collect()
            };

            let max_len = max_length.unwrap_or(self.inner.model_max_length());
            if truncation {
                for enc in &mut result_encodings {
                    if enc.len() > max_len {
                        if stride > 0 {
                            enc.truncate_with_stride(max_len, stride);
                        } else {
                            enc.truncate(max_len);
                        }
                    }
                }
            }

            if let Some(pad_strategy) = padding {
                let pad_to_len = if pad_strategy == "max_length" {
                    max_len
                } else {
                    result_encodings.iter().map(|e| e.len()).max().unwrap_or(0)
                };
                let pad_id = self.inner.special_tokens().get("[PAD]")
                    .or_else(|| self.inner.special_tokens().get("<pad>"))
                    .copied()
                    .unwrap_or(0);
                let pad_token = self.inner.id_to_token(pad_id).unwrap_or_else(|| "<pad>".to_string());
                let pad_left = pad_strategy == "left" || self.inner.padding_side() == "left";

                for enc in &mut result_encodings {
                    enc.pad(pad_to_len, pad_id, &pad_token, pad_left);
                }
            }

            Ok(PyBatchEncoding::new(
                result_encodings,
                return_attention_mask,
                return_token_type_ids,
                return_offsets_mapping,
                return_special_tokens_mask,
            ))
        } else if let Ok(single_text) = text.extract::<String>() {
            let pair: Option<String> = text_pair.and_then(|p| p.extract().ok());

            let mut result_encoding = if let Some(ref p) = pair {
                if add_special_tokens {
                    self.inner.encode_pair_to_encoding(&single_text, p)
                } else {
                    let ids_a = self.inner.encode(&single_text);
                    let ids_b = self.inner.encode(p);
                    let mut enc = Encoding::from_ids(
                        ids_a.clone(),
                        ids_a.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                    );
                    let enc_b = Encoding::from_ids(
                        ids_b.clone(),
                        ids_b.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                    );
                    enc.merge(enc_b, 1);
                    enc
                }
            } else if add_special_tokens {
                self.inner.encode_to_encoding(&single_text)
            } else {
                let ids = self.inner.encode(&single_text);
                Encoding::from_ids(
                    ids.clone(),
                    ids.iter().filter_map(|&id| self.inner.id_to_token(id)).collect(),
                )
            };

            let max_len = max_length.unwrap_or(self.inner.model_max_length());
            if truncation && result_encoding.len() > max_len {
                if stride > 0 {
                    result_encoding.truncate_with_stride(max_len, stride);
                } else {
                    result_encoding.truncate(max_len);
                }
            }

            if let Some(pad_strategy) = padding {
                let pad_to_len = if pad_strategy == "max_length" { max_len } else { result_encoding.len() };
                let pad_id = self.inner.special_tokens().get("[PAD]")
                    .or_else(|| self.inner.special_tokens().get("<pad>"))
                    .copied()
                    .unwrap_or(0);
                let pad_token = self.inner.id_to_token(pad_id).unwrap_or_else(|| "<pad>".to_string());
                let pad_left = pad_strategy == "left" || self.inner.padding_side() == "left";
                result_encoding.pad(pad_to_len, pad_id, &pad_token, pad_left);
            }

            Ok(PyBatchEncoding::new(
                vec![result_encoding],
                return_attention_mask,
                return_token_type_ids,
                return_offsets_mapping,
                return_special_tokens_mask,
            ))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected str or List[str]"
            ))
        }
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        Ok(self.inner.encode(text))
    }

    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.inner.encode_batch(&refs))
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Ok(self.inner.decode(&ids))
    }

    #[pyo3(signature = (ids, skip_special_tokens = false, clean_up_tokenization_spaces = true))]
    fn decode_with_options(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> PyResult<String> {
        Ok(self.inner.decode_with_options(&ids, skip_special_tokens, clean_up_tokenization_spaces))
    }

    fn decode_batch(&self, batch: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        Ok(self.inner.decode_batch(&batch))
    }

    #[pyo3(signature = (batch, skip_special_tokens = false, clean_up_tokenization_spaces = true))]
    fn decode_batch_with_options(
        &self,
        batch: Vec<Vec<u32>>,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> PyResult<Vec<String>> {
        Ok(self.inner.decode_batch_with_options(&batch, skip_special_tokens, clean_up_tokenization_spaces))
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        self.inner.convert_tokens_to_string(&tokens)
    }

    #[pyo3(signature = (ids, already_has_special_tokens = true))]
    fn get_special_tokens_mask(&self, ids: Vec<u32>, already_has_special_tokens: bool) -> Vec<u32> {
        self.inner.get_special_tokens_mask(&ids, already_has_special_tokens)
    }

    #[pyo3(signature = (is_pair = false))]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.inner.num_special_tokens_to_add(is_pair)
    }

    #[getter]
    fn is_fast(&self) -> bool {
        self.inner.is_fast()
    }

    fn encode_plus(&self, text: &str) -> PyEncoding {
        PyEncoding { inner: self.inner.encode_to_encoding(text) }
    }

    fn batch_encode_plus(&self, texts: Vec<String>) -> Vec<PyEncoding> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch_to_encoding(&refs)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    #[getter]
    fn special_tokens(&self) -> PyResult<StdHashMap<String, u32>> {
        Ok(self.inner.special_tokens().iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn save_pretrained(&self, dir: &str) -> PyResult<()> {
        self.inner.save_pretrained(dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn encode_to_encoding(&self, text: &str) -> PyEncoding {
        PyEncoding { inner: self.inner.encode_to_encoding(text) }
    }

    fn encode_pair_to_encoding(&self, text: &str, text_pair: &str) -> PyEncoding {
        PyEncoding { inner: self.inner.encode_pair_to_encoding(text, text_pair) }
    }

    #[pyo3(signature = (text, text_pair = None, max_length = 512, stride = 0))]
    fn encode_with_truncation(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: usize,
        stride: usize,
    ) -> PyEncoding {
        PyEncoding {
            inner: self.inner.encode_to_encoding_with_truncation(text, text_pair, max_length, stride),
        }
    }

    fn encode_batch_to_encoding(&self, texts: Vec<String>) -> Vec<PyEncoding> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch_to_encoding(&refs)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    fn encode_batch_pairs_to_encoding(&self, pairs: Vec<(String, String)>) -> Vec<PyEncoding> {
        let refs: Vec<(&str, &str)> = pairs.iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.inner.encode_batch_pairs_to_encoding(&refs)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    #[pyo3(signature = (texts, max_length = None, pad_left = false))]
    fn encode_batch_with_padding(
        &self,
        texts: Vec<String>,
        max_length: Option<usize>,
        pad_left: bool,
    ) -> Vec<PyEncoding> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch_with_padding(&refs, max_length, pad_left)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    #[pyo3(signature = (pairs, max_length = None, pad_left = false))]
    fn encode_batch_pairs_with_padding(
        &self,
        pairs: Vec<(String, String)>,
        max_length: Option<usize>,
        pad_left: bool,
    ) -> Vec<PyEncoding> {
        let refs: Vec<(&str, &str)> = pairs.iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.inner.encode_batch_pairs_with_padding(&refs, max_length, pad_left)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    fn add_token(&mut self, content: &str, id: u32, special: bool) {
        self.inner.add_token(content, id, special);
    }

    fn add_tokens(&mut self, tokens: Vec<(String, u32, bool)>) {
        self.inner.add_tokens(tokens);
    }

    fn set_normalizer(&mut self, normalizer: &PyNormalizer) {
        self.inner.set_normalizer(normalizer.inner.clone());
    }

    fn set_pre_tokenizer(&mut self, pre_tokenizer: &PyPreTokenizer) {
        self.inner.set_pre_tokenizer(pre_tokenizer.inner.clone());
    }

    fn set_post_processor(&mut self, post_processor: &PyPostProcessor) {
        self.inner.set_post_processor(post_processor.inner.clone());
    }

    fn set_decoder(&mut self, decoder: &PyDecoder) {
        self.inner.set_decoder(decoder.inner.clone());
    }

    fn get_vocab(&self) -> StdHashMap<String, u32> {
        self.inner.get_vocab().into_iter().collect()
    }

    #[pyo3(signature = (ids, skip_special_tokens = false))]
    fn convert_ids_to_tokens(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Vec<Option<String>> {
        self.inner.convert_ids_to_tokens(&ids, skip_special_tokens)
    }

    #[getter]
    fn model_max_length(&self) -> usize {
        self.inner.model_max_length()
    }

    #[setter]
    fn set_model_max_length(&mut self, value: usize) {
        self.inner.set_model_max_length(value);
    }

    #[getter]
    fn padding_side(&self) -> String {
        self.inner.padding_side().to_string()
    }

    #[setter]
    fn set_padding_side(&mut self, value: &str) {
        self.inner.set_padding_side(value);
    }

    #[getter]
    fn truncation_side(&self) -> String {
        self.inner.truncation_side().to_string()
    }

    #[setter]
    fn set_truncation_side(&mut self, value: &str) {
        self.inner.set_truncation_side(value);
    }

    #[getter]
    fn chat_template(&self) -> Option<String> {
        self.inner.chat_template().map(|s| s.to_string())
    }

    #[setter]
    fn set_chat_template(&mut self, value: Option<String>) {
        self.inner.set_chat_template(value);
    }

    #[pyo3(signature = (messages, add_generation_prompt = false, tokenize = true))]
    fn apply_chat_template(
        &self,
        messages: Vec<StdHashMap<String, String>>,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        use std::collections::HashMap;

        let msgs: Vec<HashMap<String, String>> = messages.into_iter()
            .map(|m| m.into_iter().collect())
            .collect();

        let result = self.inner.apply_chat_template(&msgs, add_generation_prompt, tokenize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Python::with_gil(|py| {
            match result {
                ChatTemplateResult::Text(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
                ChatTemplateResult::Tokenized(ids) => Ok(ids.into_pyobject(py)?.into_any().unbind()),
            }
        })
    }

    #[pyo3(signature = (
        ids,
        pair_ids = None,
        add_special_tokens = true,
        padding = None,
        truncation = false,
        max_length = None,
        stride = 0,
        return_attention_mask = true
    ))]
    fn prepare_for_model(
        &self,
        ids: Vec<u32>,
        pair_ids: Option<Vec<u32>>,
        add_special_tokens: bool,
        padding: Option<&str>,
        truncation: bool,
        max_length: Option<usize>,
        stride: usize,
        return_attention_mask: bool,
    ) -> PyEncoding {
        PyEncoding {
            inner: self.inner.prepare_for_model(
                ids,
                pair_ids,
                add_special_tokens,
                padding,
                truncation,
                max_length,
                stride,
                return_attention_mask,
            )
        }
    }

    #[pyo3(signature = (repo_id, token = None, private = false))]
    fn push_to_hub(
        &self,
        repo_id: &str,
        token: Option<&str>,
        private: bool,
    ) -> PyResult<String> {
        let temp_dir = std::env::temp_dir().join(format!("tokenizer_upload_{}", std::process::id()));
        self.inner.save_pretrained(&temp_dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let auth_token = token.map(|s| s.to_string())
            .or_else(|| std::env::var("HF_TOKEN").ok());

        let auth_token = auth_token.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No token provided. Set HF_TOKEN environment variable or pass token parameter."
            )
        })?;

        let create_url = "https://huggingface.co/api/repos/create".to_string();
        let create_response = ureq::post(&create_url)
            .set("Authorization", &format!("Bearer {}", auth_token))
            .send_json(ureq::json!({
                "type": "model",
                "name": repo_id,
                "private": private
            }));

        if let Err(e) = create_response {
            let err_str = e.to_string();
            if !err_str.contains("409") && !err_str.contains("already exists") {
                eprintln!("Warning creating repo: {}", err_str);
            }
        }

        let files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"];

        for filename in &files {
            let file_path = temp_dir.join(filename);
            if file_path.exists() {
                let content = std::fs::read_to_string(&file_path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

                let upload_url = format!(
                    "https://huggingface.co/api/{}/upload/main/{}",
                    repo_id, filename
                );

                ureq::put(&upload_url)
                    .set("Authorization", &format!("Bearer {}", auth_token))
                    .set("Content-Type", "application/json")
                    .send_string(&content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("Failed to upload {}: {}", filename, e)
                    ))?;
            }
        }

        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(format!("https://huggingface.co/{}", repo_id))
    }

    // Special Token Properties
    #[getter]
    fn bos_token(&self) -> Option<String> {
        self.inner.bos_token().map(|s| s.to_string())
    }

    #[getter]
    fn eos_token(&self) -> Option<String> {
        self.inner.eos_token().map(|s| s.to_string())
    }

    #[getter]
    fn pad_token(&self) -> Option<String> {
        self.inner.pad_token().map(|s| s.to_string())
    }

    #[getter]
    fn unk_token(&self) -> Option<String> {
        self.inner.unk_token().map(|s| s.to_string())
    }

    #[getter]
    fn sep_token(&self) -> Option<String> {
        self.inner.sep_token().map(|s| s.to_string())
    }

    #[getter]
    fn cls_token(&self) -> Option<String> {
        self.inner.cls_token().map(|s| s.to_string())
    }

    #[getter]
    fn mask_token(&self) -> Option<String> {
        self.inner.mask_token().map(|s| s.to_string())
    }

    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }

    #[getter]
    fn unk_token_id(&self) -> Option<u32> {
        self.inner.unk_token_id()
    }

    #[getter]
    fn sep_token_id(&self) -> Option<u32> {
        self.inner.sep_token_id()
    }

    #[getter]
    fn cls_token_id(&self) -> Option<u32> {
        self.inner.cls_token_id()
    }

    #[getter]
    fn mask_token_id(&self) -> Option<u32> {
        self.inner.mask_token_id()
    }

    #[getter]
    fn all_special_tokens(&self) -> Vec<String> {
        self.inner.all_special_tokens()
    }

    #[getter]
    fn all_special_ids(&self) -> Vec<u32> {
        self.inner.all_special_ids()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize(text)
    }

    fn convert_tokens_to_ids(&self, tokens: Vec<String>) -> Vec<Option<u32>> {
        self.inner.convert_tokens_to_ids(&tokens)
    }

    #[pyo3(signature = (sequences, skip_special_tokens = false, clean_up_tokenization_spaces = true))]
    fn batch_decode(
        &self,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> Vec<String> {
        self.inner.decode_batch_with_options(&sequences, skip_special_tokens, clean_up_tokenization_spaces)
    }

    #[pyo3(signature = (direction = None, pad_to_multiple_of = None, pad_id = None, pad_token = None, length = None))]
    fn enable_padding(
        &mut self,
        direction: Option<&str>,
        pad_to_multiple_of: Option<usize>,
        pad_id: Option<u32>,
        pad_token: Option<&str>,
        length: Option<usize>,
    ) {
        self.inner.enable_padding(direction, pad_to_multiple_of, pad_id, pad_token, length);
    }

    fn no_padding(&mut self) {
        self.inner.no_padding();
    }

    #[pyo3(signature = (max_length, stride = None, strategy = None, direction = None))]
    fn enable_truncation(
        &mut self,
        max_length: usize,
        stride: Option<usize>,
        strategy: Option<&str>,
        direction: Option<&str>,
    ) {
        self.inner.enable_truncation(max_length, stride, strategy, direction);
    }

    fn no_truncation(&mut self) {
        self.inner.no_truncation();
    }

    fn add_special_tokens(&mut self, special_tokens_dict: StdHashMap<String, String>) -> usize {
        use std::collections::HashMap;
        let map: HashMap<String, String> = special_tokens_dict.into_iter().collect();
        self.inner.add_special_tokens_dict(&map)
    }
}
