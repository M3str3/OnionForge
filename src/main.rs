mod gpu;

use gpu::nvidia::{initialize_nvidia_engine, NvidiaGpuEngine};
use regex::Regex;
use std::{
    env,
    fs::{self, File},
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};
use torut::onion::TorSecretKeyV3;

#[derive(Clone)]
enum KeyGenerationBackend {
    Cpu,
    NvidiaGpu(Arc<NvidiaGpuEngine>),
}

static GPU_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);

impl KeyGenerationBackend {
    fn generate_key(&self) -> TorSecretKeyV3 {
        match self {
            KeyGenerationBackend::Cpu => TorSecretKeyV3::generate(),
            KeyGenerationBackend::NvidiaGpu(engine) => match engine.generate_single() {
                Ok(key) => key,
                Err(err) => {
                    if GPU_ERROR_LOGGED
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        eprintln!(
                            "[#] GPU backend error: {}. Falling back to CPU generation for subsequent attempts.",
                            err
                        );
                    }
                    TorSecretKeyV3::generate()
                }
            },
        }
    }

    fn description(&self) -> String {
        match self {
            KeyGenerationBackend::Cpu => "CPU".to_string(),
            KeyGenerationBackend::NvidiaGpu(engine) => {
                format!("NVIDIA GPU [{}]", engine.label())
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <prefix> [--no-header] [--simulate] [--no-stop] [--gpu] [--gpu-worker <path>] [--gpu-worker-arg <arg>]",
            args[0]
        );
        return;
    }

    let prefix = args[1].clone();
    let mut simulate = false;
    let mut no_stop = false;
    let mut no_header = false;
    let mut use_gpu = false;
    let mut gpu_worker_path = env::var("ONIONFORGE_GPU_WORKER").ok();
    let mut gpu_worker_args: Vec<String> = env::var("ONIONFORGE_GPU_WORKER_ARGS")
        .ok()
        .map(|value| {
            value
                .split_whitespace()
                .map(|token| token.to_string())
                .collect()
        })
        .unwrap_or_default();

    let mut index = 2;
    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "--simulate" => simulate = true,
            "--no-stop" => no_stop = true,
            "--no-header" => no_header = true,
            "--gpu" => use_gpu = true,
            "--gpu-worker" => {
                if index + 1 >= args.len() {
                    eprintln!("[!] --gpu-worker requires a command path argument.");
                    return;
                }
                index += 1;
                gpu_worker_path = Some(args[index].clone());
                use_gpu = true;
            }
            "--gpu-worker-arg" => {
                if index + 1 >= args.len() {
                    eprintln!("[!] --gpu-worker-arg requires a value.");
                    return;
                }
                index += 1;
                gpu_worker_args.push(args[index].clone());
                use_gpu = true;
            }
            _ if arg.starts_with("--gpu-worker=") => {
                let value = arg
                    .split_once('=')
                    .map(|(_, value)| value.to_string())
                    .unwrap_or_default();
                if value.is_empty() {
                    eprintln!("[!] --gpu-worker expects a non-empty command path.");
                    return;
                }
                gpu_worker_path = Some(value);
                use_gpu = true;
            }
            _ if arg.starts_with("--gpu-worker-arg=") => {
                let value = arg
                    .split_once('=')
                    .map(|(_, value)| value.to_string())
                    .unwrap_or_default();
                if value.is_empty() {
                    eprintln!("[!] --gpu-worker-arg expects a non-empty value.");
                    return;
                }
                gpu_worker_args.push(value);
                use_gpu = true;
            }
            _ => {
                eprintln!("[!] Unknown option: {}", arg);
                return;
            }
        }
        index += 1;
    }

    let is_regex = prefix.starts_with("r/");

    if no_stop {
        println!("[*] No-stop mode enabled, continuing to generate and save keys...");
    }

    let pattern = if is_regex {
        Regex::new(&prefix[2..]).expect("Invalid regex pattern")
    } else {
        Regex::new(&format!("^{}", regex::escape(&prefix)))
            .expect("Failed to create regex from prefix")
    };

    let backend = if use_gpu {
        println!("[*] GPU acceleration requested. Initializing NVIDIA backend...");
        if let Some(path) = &gpu_worker_path {
            if gpu_worker_args.is_empty() {
                println!("[*] GPU worker command configured: {}", path);
            } else {
                println!(
                    "[*] GPU worker command configured: {} {}",
                    path,
                    gpu_worker_args.join(" ")
                );
            }
        } else if !gpu_worker_args.is_empty() {
            eprintln!(
                "[#] GPU worker arguments were provided but no worker command is configured. Arguments will be ignored until a worker is set."
            );
        }

        match initialize_nvidia_engine(gpu_worker_path, gpu_worker_args) {
            Ok(engine) => {
                println!("[*] GPU backend ready: {}", engine.label());
                println!("[*] GPU optimizations: batch_size=16384, cache_size=65536, threads_per_block=512");
                KeyGenerationBackend::NvidiaGpu(Arc::new(engine))
            }
            Err(err) => {
                eprintln!(
                    "[#] Failed to initialise GPU backend ({}). Falling back to CPU generation.",
                    err
                );
                KeyGenerationBackend::Cpu
            }
        }
    } else {
        KeyGenerationBackend::Cpu
    };

    println!("[*] Using backend: {}", backend.description());

    let num_threads = num_cpus::get();

    if simulate {
        println!(
            "[#] Simulation mode activated. Running for 10 seconds to estimate generation time..."
        );
        run_simulation(&pattern, &prefix, is_regex, backend.clone());
        return;
    }

    if no_header {
        println!("[#] WARNING: No header mode activated. The key will not have the header required by Tor service.");
    } else {
        println!("[*] Header mode activated. The key will be ready to use.");
    }

    let (tx, rx) = mpsc::channel();
    let attempt_counter = Arc::new(AtomicU64::new(0));
    let time_counter = Arc::new(AtomicU64::new(0));

    let display_attempt = attempt_counter.clone();
    let display_time = time_counter.clone();

    println!(
        "[*] Searching for .onion addresses starting with '{}'",
        prefix
    );

    thread::spawn(move || loop {
        print!(
            "\rAttempts: {:>10}, Elapsed Time: {:>5} s",
            display_attempt.load(Ordering::SeqCst),
            display_time.load(Ordering::SeqCst)
        );
        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_secs(1));
        display_time.fetch_add(1, Ordering::SeqCst);
    });

    let key_dir = "keys";
    fs::create_dir_all(key_dir).expect("[!] Failed to create directory for keys");

    for _ in 0..num_threads {
        let tx_clone = tx.clone();
        let regex_clone = pattern.clone();
        let counter_clone = attempt_counter.clone();
        let backend_clone = backend.clone();

        thread::spawn(move || {
            loop {
                let sk = backend_clone.generate_key();
                let address = sk.public().get_onion_address();
                let address_str = address.to_string();

                counter_clone.fetch_add(1, Ordering::SeqCst);

                if regex_clone.is_match(&address_str) {
                    let file_name = format!("{}/{}.bin", key_dir, address_str);

                    let mut file = File::create(&file_name).expect("[!] Failed to create file");

                    if !no_header {
                        // To be accepted by Tor service, the key file must have the following header:
                        // -------------------------------------------------------------------------------
                        // 00000000  3d 3d 20 65 64 32 35 35  31 39 76 31 2d 73 65 63  |== ed25519v1-sec|
                        // 00000010  72 65 74 3a 20 74 79 70  65 30 20 3d 3d 00 00 00  |ret: type0 ==...|
                        let header: [u8; 32] = [
                            0x3d, 0x3d, 0x20, 0x65, 0x64, 0x32, 0x35, 0x35, 0x31, 0x39, 0x76, 0x31,
                            0x2d, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x3a, 0x20, 0x74, 0x79, 0x70,
                            0x65, 0x30, 0x20, 0x3d, 0x3d, 0x00, 0x00, 0x00,
                        ];
                        file.write_all(&header).expect("[!] Failed to write header");
                    }

                    file.write_all(&sk.as_bytes())
                        .expect("[!] Failed to write to file");
                    println!("\n[*] Saved private key in {}", &file_name);

                    if !no_stop {
                        tx_clone
                            .send((address_str, sk))
                            .expect("[!] Failed to send through channel");
                        break;
                    }
                }
            }
        });
    }

    if !no_stop {
        let (found_address, _) = rx.recv().unwrap();
        println!("\n[*] First found address: {}", found_address);
    } else {
        loop {
            thread::sleep(Duration::from_secs(100));
        }
    }
}

fn run_simulation(pattern: &Regex, prefix: &str, is_regex: bool, backend: KeyGenerationBackend) {
    let num_threads = num_cpus::get();
    let attempt_counter = Arc::new(AtomicU64::new(0));
    let match_counter = Arc::new(AtomicU64::new(0));
    let simulation_duration = Duration::from_secs(10);

    for _ in 0..num_threads {
        let counter_clone = attempt_counter.clone();
        let match_clone = match_counter.clone();
        let regex_clone = pattern.clone();
        let backend_clone = backend.clone();

        thread::spawn(move || {
            let end_time = std::time::Instant::now() + simulation_duration;
            while std::time::Instant::now() < end_time {
                let sk = backend_clone.generate_key();
                let address = sk.public().get_onion_address();
                let address_str = address.to_string();
                counter_clone.fetch_add(1, Ordering::SeqCst);
                if regex_clone.is_match(&address_str) {
                    match_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
    }

    thread::sleep(simulation_duration);
    let total_attempts = attempt_counter.load(Ordering::SeqCst);
    let total_matches = match_counter.load(Ordering::SeqCst);
    let attempts_per_second = total_attempts as f64 / simulation_duration.as_secs() as f64;
    let match_rate = total_matches as f64 / total_attempts as f64;
    let estimated_time_to_match = if match_rate > 0.0 {
        1.0 / (attempts_per_second * match_rate)
    } else {
        f64::INFINITY
    };

    let probability = calculate_probability(prefix, is_regex);
    let estimated_seconds = if probability > 0.0 {
        1.0 / (attempts_per_second * probability)
    } else {
        f64::INFINITY
    };

    println!("\n[*] Simulation results:");
    println!(
        "  Total attempts in {} seconds: {}",
        simulation_duration.as_secs(),
        total_attempts
    );
    println!("  Total matches found: {}", total_matches);
    println!("  Attempts per second: {:.2}", attempts_per_second);
    println!("  Match rate: {:.6}", match_rate);
    println!(
        "  Estimated time to find a match based on simulation: {:.2} seconds",
        estimated_time_to_match
    );
    println!(
        "  Estimated time to find a match based on probability: {:.2} seconds",
        estimated_seconds
    );
    println!("  Threads used: {}", num_threads);
    println!("  Prefix length: {}", prefix.len());
}

fn calculate_probability(prefix: &str, is_regex: bool) -> f64 {
    const ONION_ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz234567";
    let alphabet_size = ONION_ALPHABET.len() as f64;

    if is_regex {
        println!(
            "Probability checking from regex is not yet implemented... using standard checking..."
        );
        let approx_pattern_length = prefix.len() as f64;
        1.0 / alphabet_size.powf(approx_pattern_length)
    } else {
        let prefix_length = prefix.len();
        1.0 / alphabet_size.powi(prefix_length as i32)
    }
}
