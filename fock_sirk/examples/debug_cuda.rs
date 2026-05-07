use candle_core::Device;

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
