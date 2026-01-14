//! HuggingFace Hub integration for downloading pretrained tokenizers
//!
//! Supports downloading tokenizer files from the HuggingFace Hub.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// HuggingFace Hub configuration
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Base URL for the Hub API
    pub endpoint: String,
    /// Cache directory for downloaded files
    pub cache_dir: PathBuf,
    /// Whether to use authentication token
    pub token: Option<String>,
    /// Request timeout in seconds
    pub timeout: u64,
}

impl Default for HubConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("huggingface")
            .join("hub");

        Self {
            endpoint: "https://huggingface.co".to_string(),
            cache_dir,
            token: std::env::var("HF_TOKEN").ok(),
            timeout: 60,
        }
    }
}

/// Download a file from the HuggingFace Hub
pub fn download_file(
    repo_id: &str,
    filename: &str,
    config: &HubConfig,
) -> io::Result<PathBuf> {
    // Create cache path
    let repo_cache = config.cache_dir.join(repo_id.replace('/', "--"));
    fs::create_dir_all(&repo_cache)?;

    let file_path = repo_cache.join(filename);

    // Check if file already exists in cache
    if file_path.exists() {
        return Ok(file_path);
    }

    // Build download URL
    let url = format!(
        "{}/{}/resolve/main/{}",
        config.endpoint, repo_id, filename
    );

    // Download file using ureq (blocking HTTP client)
    let response = download_with_ureq(&url, config)?;

    // Write to cache
    let mut file = fs::File::create(&file_path)?;
    file.write_all(&response)?;

    Ok(file_path)
}

/// Download using ureq HTTP client
fn download_with_ureq(url: &str, config: &HubConfig) -> io::Result<Vec<u8>> {
    let mut request = ureq::get(url)
        .timeout(std::time::Duration::from_secs(config.timeout));

    // Add auth token if available
    if let Some(ref token) = config.token {
        request = request.set("Authorization", &format!("Bearer {}", token));
    }

    let response = request
        .call()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    // Handle redirects (ureq follows them automatically)
    if response.status() != 200 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("HTTP {}: {}", response.status(), url),
        ));
    }

    // Read response body
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok(bytes)
}

/// Download tokenizer files from HuggingFace Hub
pub fn download_tokenizer(
    repo_id: &str,
    config: &HubConfig,
) -> io::Result<PathBuf> {
    // Try to download tokenizer.json first
    let result = download_file(repo_id, "tokenizer.json", config);

    if result.is_ok() {
        return result;
    }

    // Fallback: try vocab.json + merges.txt (GPT-2 style)
    let vocab_path = download_file(repo_id, "vocab.json", config)?;
    let _ = download_file(repo_id, "merges.txt", config);

    // Return vocab path, the caller will need to handle this format
    Ok(vocab_path)
}

/// Check if a model exists in the cache
pub fn is_cached(repo_id: &str, filename: &str, config: &HubConfig) -> bool {
    let repo_cache = config.cache_dir.join(repo_id.replace('/', "--"));
    repo_cache.join(filename).exists()
}

/// Get the cache path for a repo
pub fn get_cache_path(repo_id: &str, config: &HubConfig) -> PathBuf {
    config.cache_dir.join(repo_id.replace('/', "--"))
}

/// Clear cache for a specific repo
pub fn clear_cache(repo_id: &str, config: &HubConfig) -> io::Result<()> {
    let repo_cache = config.cache_dir.join(repo_id.replace('/', "--"));
    if repo_cache.exists() {
        fs::remove_dir_all(repo_cache)?;
    }
    Ok(())
}

/// Clear all cached files
pub fn clear_all_cache(config: &HubConfig) -> io::Result<()> {
    if config.cache_dir.exists() {
        fs::remove_dir_all(&config.cache_dir)?;
    }
    Ok(())
}

/// Resolve a model identifier to a local path
///
/// This function handles:
/// - Local file paths (returned as-is)
/// - HuggingFace Hub identifiers (downloaded and cached)
pub fn resolve_model_path(model_id: &str, config: Option<HubConfig>) -> io::Result<PathBuf> {
    // Check if it's a local path
    let local_path = Path::new(model_id);
    if local_path.exists() {
        // It's a local directory or file
        if local_path.is_dir() {
            let tokenizer_json = local_path.join("tokenizer.json");
            if tokenizer_json.exists() {
                return Ok(tokenizer_json);
            }
        }
        return Ok(local_path.to_path_buf());
    }

    // It's a Hub identifier, download it
    let config = config.unwrap_or_default();
    download_tokenizer(model_id, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_config_default() {
        let config = HubConfig::default();
        assert!(config.endpoint.contains("huggingface.co"));
        assert_eq!(config.timeout, 60);
    }

    #[test]
    fn test_cache_path() {
        let config = HubConfig::default();
        let path = get_cache_path("gpt2", &config);
        assert!(path.to_string_lossy().contains("gpt2"));
    }

    #[test]
    fn test_resolve_local_path() {
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_tokenizer.json");
        fs::write(&test_file, "{}").unwrap();

        let result = resolve_model_path(test_file.to_str().unwrap(), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file);

        // Cleanup
        fs::remove_file(test_file).ok();
    }
}
