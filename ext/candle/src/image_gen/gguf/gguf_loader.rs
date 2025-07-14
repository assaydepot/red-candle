use candle_core::{Device, Result as CandleResult, quantized::gguf_file};
use std::collections::HashMap;
use std::path::Path;

/// Information about a component within a GGUF file
#[derive(Debug, Clone)]
pub struct GGUFComponentInfo {
    pub component_type: ComponentType,
    pub tensor_count: usize,
    pub size_bytes: u64,
    pub tensor_prefixes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComponentType {
    MMDiT,           // Multimodal Diffusion Transformer
    VAE,             // Variational Auto-Encoder  
    CLIPTextG,       // CLIP-G text encoder
    CLIPTextL,       // CLIP-L text encoder
    T5Text,          // T5-XXL text encoder
    Unknown(String), // Unknown component
}

/// Metadata extracted from a GGUF file
#[derive(Debug)]
pub struct GGUFMetadata {
    pub architecture: String,
    pub quantization_version: u32,
    pub parameter_count: Option<u64>,
    pub components: HashMap<ComponentType, GGUFComponentInfo>,
    pub tensor_count: usize,
    pub model_name: Option<String>,
    pub model_author: Option<String>,
}

impl GGUFMetadata {
    /// Parse metadata from a GGUF file
    pub fn from_file(path: &Path) -> CandleResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {}", e)))?;
        
        let gguf_file = gguf_file::Content::read(&mut file)?;
        
        // Extract basic metadata
        let metadata = &gguf_file.metadata;
        let architecture = metadata.get("general.architecture")
            .and_then(|v| match v {
                gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "sd3".to_string());
        
        let quantization_version = metadata.get("general.quantization_version")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(2);
        
        let parameter_count = metadata.get("general.parameter_count")
            .and_then(|v| match v {
                gguf_file::Value::U64(n) => Some(*n),
                _ => None,
            });
        
        let model_name = metadata.get("general.name")
            .and_then(|v| match v {
                gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            });
        
        let model_author = metadata.get("general.author")
            .and_then(|v| match v {
                gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            });
        
        // Analyze tensor structure to identify components
        let components = Self::analyze_components(&gguf_file)?;
        let tensor_count = gguf_file.tensor_infos.len();
        
        Ok(Self {
            architecture,
            quantization_version,
            parameter_count,
            components,
            tensor_count,
            model_name,
            model_author,
        })
    }
    
    /// Analyze tensor names to identify model components
    fn analyze_components(gguf_file: &gguf_file::Content) -> CandleResult<HashMap<ComponentType, GGUFComponentInfo>> {
        let mut components = HashMap::new();
        let tensor_infos = &gguf_file.tensor_infos;
        
        // Group tensors by component based on naming patterns
        let mut mmdit_tensors = Vec::new();
        let mut vae_tensors = Vec::new();
        let mut clip_g_tensors = Vec::new();
        let mut clip_l_tensors = Vec::new();
        let mut t5_tensors = Vec::new();
        let mut unknown_tensors = Vec::new();
        
        for (name, info) in tensor_infos {
            let tensor_name = name.as_str();
            
            // Classify tensors based on naming patterns
            if Self::is_mmdit_tensor(tensor_name) {
                mmdit_tensors.push((name.clone(), info));
            } else if Self::is_vae_tensor(tensor_name) {
                vae_tensors.push((name.clone(), info));
            } else if Self::is_clip_g_tensor(tensor_name) {
                clip_g_tensors.push((name.clone(), info));
            } else if Self::is_clip_l_tensor(tensor_name) {
                clip_l_tensors.push((name.clone(), info));
            } else if Self::is_t5_tensor(tensor_name) {
                t5_tensors.push((name.clone(), info));
            } else {
                unknown_tensors.push((name.clone(), info));
            }
        }
        
        // Create component info for each identified component
        if !mmdit_tensors.is_empty() {
            components.insert(ComponentType::MMDiT, Self::create_component_info(
                ComponentType::MMDiT,
                &mmdit_tensors
            ));
        }
        
        if !vae_tensors.is_empty() {
            components.insert(ComponentType::VAE, Self::create_component_info(
                ComponentType::VAE,
                &vae_tensors
            ));
        }
        
        if !clip_g_tensors.is_empty() {
            components.insert(ComponentType::CLIPTextG, Self::create_component_info(
                ComponentType::CLIPTextG,
                &clip_g_tensors
            ));
        }
        
        if !clip_l_tensors.is_empty() {
            components.insert(ComponentType::CLIPTextL, Self::create_component_info(
                ComponentType::CLIPTextL,
                &clip_l_tensors
            ));
        }
        
        if !t5_tensors.is_empty() {
            components.insert(ComponentType::T5Text, Self::create_component_info(
                ComponentType::T5Text,
                &t5_tensors
            ));
        }
        
        // Log unknown tensors for debugging
        if !unknown_tensors.is_empty() {
            eprintln!("Found {} unknown tensors:", unknown_tensors.len());
            for (name, _) in unknown_tensors.iter().take(5) {
                eprintln!("  - {}", name);
            }
            if unknown_tensors.len() > 5 {
                eprintln!("  ... and {} more", unknown_tensors.len() - 5);
            }
        }
        
        Ok(components)
    }
    
    /// Check if tensor name belongs to MMDiT component
    fn is_mmdit_tensor(name: &str) -> bool {
        // SD3 MMDiT tensor patterns
        name.starts_with("model.diffusion_model.") ||
        name.starts_with("diffusion_model.") ||
        name.starts_with("dit.") ||
        name.contains(".joint_blocks.") ||
        name.contains(".final_layer.") ||
        name.contains(".x_embedder.") ||
        name.contains(".t_embedder.") ||
        name.contains(".y_embedder.") ||
        name.contains(".context_embedder.")
    }
    
    /// Check if tensor name belongs to VAE component
    fn is_vae_tensor(name: &str) -> bool {
        name.starts_with("first_stage_model.") ||
        name.starts_with("vae.") ||
        (name.starts_with("encoder.") && !name.contains("text_encoder")) ||
        (name.starts_with("decoder.") && !name.contains("text_encoder"))
    }
    
    /// Check if tensor name belongs to CLIP-G text encoder
    fn is_clip_g_tensor(name: &str) -> bool {
        name.starts_with("conditioner.embedders.0.") ||
        name.starts_with("text_encoder.") ||
        name.starts_with("text_encoders.clip_g.") ||
        name.starts_with("cond_stage_model.0.") ||
        (name.contains("clip") && name.contains("_g.") && !name.contains("clip_l"))
    }
    
    /// Check if tensor name belongs to CLIP-L text encoder
    fn is_clip_l_tensor(name: &str) -> bool {
        name.starts_with("conditioner.embedders.1.") ||
        name.starts_with("text_encoder_2.") ||
        name.starts_with("text_encoders.clip_l.") ||
        name.starts_with("cond_stage_model.1.") ||
        (name.contains("clip") && name.contains("_l.") && !name.contains("clip_g"))
    }
    
    /// Check if tensor name belongs to T5 text encoder
    fn is_t5_tensor(name: &str) -> bool {
        name.starts_with("conditioner.embedders.2.") ||
        name.starts_with("text_encoder_3.") ||
        name.starts_with("text_encoders.t5xxl.") ||
        name.starts_with("t5.") ||
        name.starts_with("t5xxl.") ||
        name.contains("t5_xxl")
    }
    
    /// Create component info from tensor list
    fn create_component_info(
        component_type: ComponentType,
        tensors: &[(String, &gguf_file::TensorInfo)]
    ) -> GGUFComponentInfo {
        let tensor_count = tensors.len();
        let size_bytes = tensors.iter()
            .map(|(_, info)| {
                let n_elements: usize = info.shape.elem_count();
                info.ggml_dtype.type_size() * n_elements
            })
            .sum::<usize>() as u64;
        
        // Extract common prefixes
        let mut prefixes = std::collections::HashSet::new();
        for (name, _) in tensors {
            if let Some(prefix) = name.split('.').next() {
                prefixes.insert(prefix.to_string());
            }
        }
        
        GGUFComponentInfo {
            component_type,
            tensor_count,
            size_bytes,
            tensor_prefixes: prefixes.into_iter().collect(),
        }
    }
    
    /// Check if this GGUF file contains a specific component
    pub fn has_component(&self, component_type: &ComponentType) -> bool {
        self.components.contains_key(component_type)
    }
    
    /// Get information about a specific component
    pub fn get_component(&self, component_type: &ComponentType) -> Option<&GGUFComponentInfo> {
        self.components.get(component_type)
    }
    
    /// Get total model size in bytes
    pub fn total_size_bytes(&self) -> u64 {
        self.components.values()
            .map(|info| info.size_bytes)
            .sum()
    }
    
    /// Get human-readable size
    pub fn total_size_human(&self) -> String {
        let bytes = self.total_size_bytes() as f64;
        if bytes < 1024.0 {
            format!("{:.1} B", bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.1} KB", bytes / 1024.0)
        } else if bytes < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.1} MB", bytes / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Load a specific component from a GGUF file
/// Returns the GGUF content and file handle for component loading
pub fn load_component_tensors(
    path: &Path,
    component_type: &ComponentType,
    device: &Device,
) -> CandleResult<(gguf_file::Content, std::fs::File, Device)> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {}", e)))?;
    
    let gguf_content = gguf_file::Content::read(&mut file)?;
    
    // Filter tensors for the specific component
    let tensor_infos = &gguf_content.tensor_infos;
    let mut found_tensors = 0;
    
    for (name, _info) in tensor_infos {
        let belongs_to_component = match component_type {
            ComponentType::MMDiT => GGUFMetadata::is_mmdit_tensor(name),
            ComponentType::VAE => GGUFMetadata::is_vae_tensor(name),
            ComponentType::CLIPTextG => GGUFMetadata::is_clip_g_tensor(name),
            ComponentType::CLIPTextL => GGUFMetadata::is_clip_l_tensor(name),
            ComponentType::T5Text => GGUFMetadata::is_t5_tensor(name),
            ComponentType::Unknown(_) => false,
        };
        
        if belongs_to_component {
            found_tensors += 1;
        }
    }
    
    if found_tensors == 0 {
        return Err(candle_core::Error::Msg(
            format!("No tensors found for component {:?}", component_type)
        ));
    }
    
    // Return the GGUF content and file handle for component-specific loading
    Ok((gguf_content, file, device.clone()))
}