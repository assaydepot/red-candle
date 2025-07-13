use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <safetensors_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    println!("Inspecting: {}", file_path);

    // Load the safetensors file
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[Path::new(file_path)], DType::F32, &device)? };

    // Get tensor info from the underlying data
    let tensors = candle_core::safetensors::load(file_path, &device)?;
    
    // Group tensors by prefix
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    
    for (name, tensor) in &tensors {
        let parts: Vec<&str> = name.split('.').collect();
        let prefix = if parts.len() > 2 {
            parts[0..2].join(".")
        } else {
            parts[0].to_string()
        };
        
        groups.entry(prefix.clone()).or_insert_with(Vec::new).push(name.clone());
    }
    
    // Print summary
    println!("\nTotal tensors: {}", tensors.len());
    println!("\nTensor groups:");
    
    let mut sorted_groups: Vec<_> = groups.iter().collect();
    sorted_groups.sort_by_key(|&(k, _)| k);
    
    for (prefix, names) in sorted_groups {
        println!("\n{}: {} tensors", prefix, names.len());
        
        // Show first few tensor names
        let mut sorted_names = names.clone();
        sorted_names.sort();
        
        for (i, name) in sorted_names.iter().take(5).enumerate() {
            if let Some(tensor) = tensors.get(name) {
                println!("  {} -> {:?}", name, tensor.shape());
            }
        }
        
        if names.len() > 5 {
            println!("  ... and {} more", names.len() - 5);
        }
    }
    
    // Check for specific paths we need
    println!("\nChecking for SD3 VAE paths:");
    let vae_paths = vec![
        "first_stage_model.decoder.conv_in.weight",
        "first_stage_model.decoder.mid_block.resnets.0.norm1.weight",
        "vae.decoder.conv_in.weight",
        "vae.decoder.mid_block.resnets.0.norm1.weight",
    ];
    
    for path in vae_paths {
        if tensors.contains_key(path) {
            println!("  ✓ Found: {}", path);
        } else {
            println!("  ✗ Missing: {}", path);
        }
    }

    Ok(())
}