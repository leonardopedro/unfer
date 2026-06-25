fn main() {
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");

    // Phase 11.1: Auto-detection of CUDA library paths to prevent CUBLAS mismatch
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
    }
}
