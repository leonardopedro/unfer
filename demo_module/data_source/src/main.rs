use serde_json::json;
use unfer_ffi::{uk_get_result, uk_init, uk_model_create, uk_model_free, uk_observe};

fn main() {
    uk_init(std::ptr::null(), 0);

    let spec = json!({
        "hamiltonian": { "kind": "builtin", "name": "harmonic_chain", "params": { "n_modes": 1, "omega": 1.0 } },
        "prior": { "kind": "vacuum" },
        "solver": { "krylov_dim": 4, "prune_eps": 1e-12, "max_components": null, "restarts": 1, "device": { "kind": "cpu" } }
    })
    .to_string();
    let spec_bytes = spec.as_bytes();
    let handle = uk_model_create(spec_bytes.as_ptr(), spec_bytes.len() as i64);

    if handle < 0 {
        panic!("Failed to create model: {}", handle);
    }
    println!("Created model with handle: {}", handle);

    let observations: Vec<serde_json::Value> = vec![
        json!({ "kind": "vacuum" }),
        json!({ "kind": "boson_mode_total", "mode": 0, "cmp": "eq", "value": 0 }),
        json!({ "kind": "boson_mode_total", "mode": 0, "cmp": "ge", "value": 1 }),
        json!({ "kind": "boson_mode_total", "mode": 0, "cmp": "eq", "value": 1 }),
    ];
    let labels = ["vacuum", "mode0==0", "mode0>=1", "mode0==1"];
    for (i, obs) in observations.iter().enumerate() {
        let obs_json = obs.to_string();
        let obs_bytes = obs_json.as_bytes();

        if uk_observe(handle, obs_bytes.as_ptr(), obs_bytes.len() as i64) == 0 {
            let mut buf = vec![0u8; 1024];
            let len = uk_get_result(handle, buf.as_mut_ptr(), 1024);
            if len > 0 {
                let result = String::from_utf8_lossy(&buf[..len as usize]);
                println!("Observation {} ({}): result = {}", i, labels[i], result);
            }
        } else {
            println!("Observation {} ({}) failed", i, labels[i]);
        }
    }

    uk_model_free(handle);
    println!("Model freed successfully.");
}
