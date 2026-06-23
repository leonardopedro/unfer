use candle_core::Device;

#[cfg(feature = "cuda")]
fn main() {
    println!("Checking CUDA availability...");
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("SUCCESS: Found CUDA device: {:?}", device);
        }
        Err(e) => {
            println!("FAILURE: Could not initialize CUDA: {:?}", e);
        }
    }

    println!("cuda_if_available(0) returns: {:?}", Device::cuda_if_available(0));
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let _ = Device::Cpu;
    println!("Built without the `cuda` feature; rebuild with `--features cuda` to probe a GPU.");
}
