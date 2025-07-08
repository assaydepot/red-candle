use std::env;
use std::path::Path;

fn main() {
    // Register our custom cfg flags with rustc
    println!("cargo::rustc-check-cfg=cfg(force_cpu)");
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_mkl)");
    println!("cargo::rustc-check-cfg=cfg(has_accelerate)");
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CANDLE_FORCE_CPU");
    println!("cargo:rerun-if-env-changed=CANDLE_CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CANDLE_FEATURES");

    // Check if we should force CPU only
    if env::var("CANDLE_FORCE_CPU").is_ok() {
        println!("cargo:rustc-cfg=force_cpu");
        println!("cargo:warning=CANDLE_FORCE_CPU is set, disabling all acceleration");
        return;
    }

    // Detect CUDA availability
    let cuda_available = detect_cuda();
    if cuda_available {
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:warning=CUDA detected, CUDA acceleration will be available");
    }

    // Detect Metal availability (macOS only)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cfg=has_metal");
        println!("cargo:warning=Metal detected (macOS), Metal acceleration will be available");
    }

    // Detect MKL availability
    if detect_mkl() {
        println!("cargo:rustc-cfg=has_mkl");
        println!("cargo:warning=Intel MKL detected, MKL acceleration will be available");
    }

    // Detect Accelerate framework (macOS)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cfg=has_accelerate");
        println!("cargo:warning=Accelerate framework detected (macOS)");
    }
}

fn detect_cuda() -> bool {
    // Check environment variables first
    if env::var("CANDLE_CUDA_PATH").is_ok() {
        return true;
    }

    if env::var("CUDA_ROOT").is_ok() || env::var("CUDA_PATH").is_ok() {
        return true;
    }

    // Check common CUDA installation paths
    let cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\CUDA",
    ];

    for path in &cuda_paths {
        if Path::new(path).exists() {
            return true;
        }
    }

    // Check if nvcc is in PATH
    if let Ok(path_var) = env::var("PATH") {
        for path in env::split_paths(&path_var) {
            if path.join("nvcc").exists() || path.join("nvcc.exe").exists() {
                return true;
            }
        }
    }

    false
}

fn detect_mkl() -> bool {
    // Check environment variables
    if env::var("MKLROOT").is_ok() || env::var("MKL_ROOT").is_ok() {
        return true;
    }

    // Check common MKL installation paths
    let mkl_paths = [
        "/opt/intel/mkl",
        "/opt/intel/oneapi/mkl/latest",
        "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest",
        "C:\\Program Files\\Intel\\oneAPI\\mkl\\latest",
    ];

    for path in &mkl_paths {
        if Path::new(path).exists() {
            return true;
        }
    }

    false
}