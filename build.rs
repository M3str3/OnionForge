use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Get the output directory for build artifacts
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    
    // Path to the CUDA source file
    let cuda_source = "kernels.cu";
    let ptx_output = out_path.join("kernels.ptx");
    
    // Tell Cargo to re-run this build script if the CUDA source changes
    println!("cargo:rerun-if-changed={}", cuda_source);
    
    // Verify that the source file exists
    if !Path::new(cuda_source).exists() {
        panic!("CUDA source file '{}' not found", cuda_source);
    }
    
    // Attempt to compile with nvc (NVIDIA HPC SDK) first
    let nvc_result = Command::new("nvc")
        .arg("-cuda")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_output)
        .arg(cuda_source)
        .output();
    
    match nvc_result {
        Ok(output) => {
            if !output.status.success() {
                eprintln!("Error compiling with nvc:");
                eprintln!("{}", String::from_utf8_lossy(&output.stderr));
                
                // Try nvcc as fallback
                println!("Attempting to compile with nvcc as fallback...");
                let nvcc_result = Command::new("nvcc")
                    .arg("-ptx")
                    .arg("-o")
                    .arg(&ptx_output)
                    .arg(cuda_source)
                    .output();
                
                match nvcc_result {
                    Ok(nvcc_output) => {
                        if !nvcc_output.status.success() {
                            eprintln!("Error compiling with nvcc:");
                            eprintln!("{}", String::from_utf8_lossy(&nvcc_output.stderr));
                            panic!("Failed to compile CUDA kernel with both nvc and nvcc");
                        } else {
                            println!("Successful compilation with nvcc");
                        }
                    }
                    Err(_) => {
                        panic!("nvcc not found. Make sure CUDA toolkit is installed");
                    }
                }
            } else {
                println!("Successful compilation with nvc");
            }
        }
        Err(_) => {
            // nvc is not available, try nvcc
            println!("nvc not found, attempting with nvcc...");
            let nvcc_result = Command::new("nvcc")
                .arg("-ptx")
                .arg("-o")
                .arg(&ptx_output)
                .arg(cuda_source)
                .output();
            
            match nvcc_result {
                Ok(nvcc_output) => {
                    if !nvcc_output.status.success() {
                        eprintln!("Error compiling with nvcc:");
                        eprintln!("{}", String::from_utf8_lossy(&nvcc_output.stderr));
                        panic!("Failed to compile CUDA kernel with nvcc");
                    } else {
                        println!("Successful compilation with nvcc");
                    }
                }
                Err(_) => {
                    panic!("nvcc not found. Make sure CUDA toolkit is installed");
                }
            }
        }
    }
    
    // Verify that the PTX file was generated successfully
    if !ptx_output.exists() {
        panic!("PTX file was not generated at {:?}", ptx_output);
    }
    
    // Link against the CUDA library
    println!("cargo:rustc-link-lib=cuda");
}
