use torut::onion::TorSecretKeyV3;
use std::{env, fs::{self, File}, io::{self, Write}, sync::{mpsc, atomic::{AtomicU64, Ordering}, Arc}, thread, time::Duration};
use num_cpus;
use regex::Regex;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <prefix> [--simulate] [--no-stop]", args[0]);
        return;
    }
    
    let simulate = args.contains(&"--simulate".to_string());
    let no_stop = args.contains(&"--no-stop".to_string());
    let prefix = args[1].clone();
    let is_regex = prefix.starts_with("r/");

    if no_stop {
        println!("[*] No-stop mode enabled, continuing to generate and save keys...");
    }
    
    let pattern = if is_regex {
        Regex::new(&prefix[2..]).expect("Invalid regex pattern")
    } else {
        Regex::new(&format!("^{}", regex::escape(&prefix))).expect("Failed to create regex from prefix")
    };

    let num_threads = num_cpus::get();

    if simulate {
        println!("[#] Simulation mode activated. Running for 10 seconds to estimate generation time...");
        run_simulation(&pattern, &prefix, is_regex);
        return;
    }

    let (tx, rx) = mpsc::channel();
    let attempt_counter = Arc::new(AtomicU64::new(0));
    let time_counter = Arc::new(AtomicU64::new(0));

    let display_attempt = attempt_counter.clone();
    let display_time = time_counter.clone();
    
    println!("[*] Searching for .onion addresses starting with '{}'", prefix);
    
    thread::spawn(move || {
        loop {
            print!("\rAttempts: {:>10}, Elapsed Time: {:>5} s", display_attempt.load(Ordering::SeqCst), display_time.load(Ordering::SeqCst));
            io::stdout().flush().unwrap();
            thread::sleep(Duration::from_secs(1));
            display_time.fetch_add(1, Ordering::SeqCst);
        }
    });

    let key_dir = "keys";
    fs::create_dir_all(key_dir).expect("[!] Failed to create directory for keys");

    for _ in 0..num_threads {
        let tx_clone = tx.clone();
        let regex_clone = pattern.clone();
        let counter_clone = attempt_counter.clone();

        thread::spawn(move || {
            loop {
                let sk = TorSecretKeyV3::generate();
                let address = sk.public().get_onion_address();
                let address_str = address.to_string();

                counter_clone.fetch_add(1, Ordering::SeqCst);

                if regex_clone.is_match(&address_str) {
                    let file_name = format!("{}/{}.bin", key_dir, address_str);

                    let mut file = File::create(&file_name).expect("[!] Failed to create file");
                    file.write_all(&sk.as_bytes()).expect("[!] Failed to write to file");
                    println!("\n[*] Saved private key in {}", &file_name);

                    if !no_stop {
                        tx_clone.send((address_str, sk)).expect("[!] Failed to send through channel");
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
        loop { thread::sleep(Duration::from_secs(100)); }
    }
}

fn run_simulation(pattern: &Regex, prefix: &str, is_regex: bool) {
    let num_threads = num_cpus::get();
    let attempt_counter = Arc::new(AtomicU64::new(0));
    let match_counter = Arc::new(AtomicU64::new(0));
    let simulation_duration = Duration::from_secs(10);

    for _ in 0..num_threads {
        let counter_clone = attempt_counter.clone();
        let match_clone = match_counter.clone();
        let regex_clone = pattern.clone();

        thread::spawn(move || {
            let end_time = std::time::Instant::now() + simulation_duration;
            while std::time::Instant::now() < end_time {
                let sk = TorSecretKeyV3::generate();
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
    println!("  Total attempts in {} seconds: {}", simulation_duration.as_secs(), total_attempts);
    println!("  Total matches found: {}", total_matches);
    println!("  Attempts per second: {:.2}", attempts_per_second);
    println!("  Match rate: {:.6}", match_rate);
    println!("  Estimated time to find a match based on simulation: {:.2} seconds", estimated_time_to_match);
    println!("  Estimated time to find a match based on probability: {:.2} seconds", estimated_seconds);
    println!("  Threads used: {}", num_threads);
    println!("  Prefix length: {}", prefix.len());
}

fn calculate_probability(prefix: &str, is_regex: bool) -> f64 {
    const ONION_ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz234567";
    let alphabet_size = ONION_ALPHABET.len() as f64;
    
    if is_regex {
        todo!();
        let approx_pattern_length = prefix.len() as f64;
        return 1.0 / alphabet_size.powf(approx_pattern_length);
    } else {
        let prefix_length = prefix.len();
        return 1.0 / alphabet_size.powi(prefix_length as i32);
    }
}
