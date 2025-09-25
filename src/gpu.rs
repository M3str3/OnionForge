//! GPU acceleration support for key generation.
//!
//! This module provides a complete NVIDIA pipeline capable of generating
//! Ed25519 keys on the GPU while still supporting the external worker workflow
//! that earlier versions of the project exposed.  The runtime uses the CUDA
//! driver API directly so a full CUDA toolkit is not required – the presence of
//! `libcuda` alongside a recent driver is enough.  When CUDA is not available
//! the caller may still hand keys off to an external worker command.

use std::fmt;

use torut::onion::TorSecretKeyV3;

/// High level error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No compatible GPU was detected on the system.
    NotAvailable(String),
    /// GPU drivers or runtime could not be initialised.
    InitializationFailed(String),
    /// Runtime execution failed while interacting with the GPU.
    ExecutionFailed(String),
    /// External GPU worker process failed to execute correctly.
    WorkerFailed(String),
    /// The worker returned unexpected data and the protocol could not continue.
    WorkerProtocol(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NotAvailable(msg)
            | GpuError::InitializationFailed(msg)
            | GpuError::ExecutionFailed(msg)
            | GpuError::WorkerFailed(msg)
            | GpuError::WorkerProtocol(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// NVIDIA specific backend utilities.
pub mod nvidia {
    use super::{GpuError, TorSecretKeyV3};
    use ed25519_dalek::{ExpandedSecretKey, SecretKey};
    use libloading::Library;
    use std::{
        collections::VecDeque,
        ffi::{c_void, CStr, CString},
        os::raw::{c_char, c_int, c_uint},
        process::Command,
        ptr,
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Mutex, Once,
        },
        thread,
    };

    /// Shared flag to avoid running the detection command more than once.
    static GPU_PROBED: AtomicBool = AtomicBool::new(false);
    /// Stores the result of the last detection attempt.
    static GPU_AVAILABLE: AtomicBool = AtomicBool::new(false);

    type CUdevice = c_int;
    type CUcontext = *mut c_void;
    type CUmodule = *mut c_void;
    type CUfunction = *mut c_void;
    type CUstream = *mut c_void;
    type CUdeviceptr = u64;
    type CUresult = c_int;

    const CUDA_SUCCESS: CUresult = 0;
    const GPU_BATCH_SIZE: usize = 16384; // Increased from 1024 to 16384 for better parallelization
    const GPU_SEED_BYTES: usize = 32;
    const GPU_CACHE_SIZE: usize = GPU_BATCH_SIZE * 8;

    // Constants for cuModuleLoadDataEx JIT compilation options
    const CU_JIT_ERROR_LOG_BUFFER: u32 = 0;
    const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 1;
    const CU_JIT_TARGET_FROM_CUCONTEXT: u32 = 2;

    #[derive(Clone)]
    enum Backend {
        Native(std::sync::Arc<OptimizedGpuEngine>),
        External(std::sync::Arc<GpuWorker>),
    }

    /// Optimized GPU engine with shared global cache for improved performance
    struct OptimizedGpuEngine {
        cache: Mutex<VecDeque<TorSecretKeyV3>>,
        engine: Mutex<CudaEngine>,
        device_name: String,
    }

    impl OptimizedGpuEngine {
        fn new(device_name: String) -> Result<Self, GpuError> {
            let engine = CudaEngine::new()?;

            println!("[*] Optimized GPU engine ready on '{}'", device_name);

            Ok(Self {
                cache: Mutex::new(VecDeque::new()),
                engine: Mutex::new(engine),
                device_name,
            })
        }

        fn generate_single(&self) -> Result<TorSecretKeyV3, GpuError> {
            if let Some(key) = self.take_cached_key() {
                return Ok(key);
            }

            let mut engine = self.engine.lock().expect("GPU engine poisoned");

            if let Some(key) = self.take_cached_key() {
                return Ok(key);
            }

            let batch = engine.generate_batch(GPU_BATCH_SIZE * 8)?;
            drop(engine);

            self.store_batch_and_take_one(batch)
        }

        fn label(&self) -> String {
            format!("CUDA optimized batching")
        }

        fn take_cached_key(&self) -> Option<TorSecretKeyV3> {
            let mut cache = self.cache.lock().expect("GPU cache poisoned");
            cache.pop_front()
        }

        fn store_batch_and_take_one(
            &self,
            batch: Vec<TorSecretKeyV3>,
        ) -> Result<TorSecretKeyV3, GpuError> {
            let mut iter = batch.into_iter();
            let first = iter.next().ok_or_else(|| {
                GpuError::ExecutionFailed("GPU batch generation returned no keys.".into())
            })?;

            let mut cache = self.cache.lock().expect("GPU cache poisoned");
            let remaining_capacity = GPU_CACHE_SIZE.saturating_sub(cache.len());
            if remaining_capacity > 0 {
                cache.extend(iter.take(remaining_capacity));
            }

            Ok(first)
        }
    }

    /// Holds lightweight metadata about the detected GPU device and backend.
    pub struct NvidiaGpuEngine {
        device_name: String,
        backend: Backend,
    }

    /// CUDA driven engine that generates seeds and derives expanded keys.
    struct CudaEngine {
        api: &'static CudaApi,
        context: CUcontext,
        module: CUmodule,
        function: CUfunction,
        device_buffer: CUdeviceptr,
        host_buffer: Vec<u8>,
        cache: Mutex<VecDeque<TorSecretKeyV3>>,
        seed_cursor: AtomicU64,
        label: String,
    }

    // CUDA pointers are safe for Send/Sync because they represent system resources
    // that can be safely shared between threads
    unsafe impl Send for CudaEngine {}
    unsafe impl Sync for CudaEngine {}

    impl Drop for CudaEngine {
        fn drop(&mut self) {
            unsafe {
                let _ = (self.api.cu_ctx_set_current)(self.context);
                let _ = (self.api.cu_mem_free)(self.device_buffer);
                let _ = (self.api.cu_module_unload)(self.module);
                let _ = (self.api.cu_ctx_destroy)(self.context);
            }
        }
    }

    struct CudaApi {
        #[allow(dead_code)]
        lib: &'static Library,
        cu_init: unsafe extern "C" fn(c_uint) -> CUresult,
        cu_device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
        cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
        cu_ctx_destroy: unsafe extern "C" fn(CUcontext) -> CUresult,
        cu_ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult,
        cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
        cu_module_load_data_ex: unsafe extern "C" fn(
            *mut CUmodule,
            *const c_void,
            u32,
            *mut u32,
            *mut *mut c_void,
        ) -> CUresult,
        cu_module_unload: unsafe extern "C" fn(CUmodule) -> CUresult,
        cu_module_get_function:
            unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
        cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
        cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
        cu_launch_kernel: unsafe extern "C" fn(
            CUfunction,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            CUstream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult,
        cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
        cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
        cu_get_error_string: Option<unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult>,
    }

    static INIT: Once = Once::new();
    static mut CUDA_API: Option<CudaApi> = None;

    /// Carga el PTX generado por el build script desde OUT_DIR
    const PTX_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

    impl NvidiaGpuEngine {
        /// Creates a key on the GPU or through the configured worker.
        pub fn generate_single(&self) -> Result<TorSecretKeyV3, GpuError> {
            match &self.backend {
                Backend::Native(engine) => engine.generate_single(),
                Backend::External(worker) => worker.generate_single(),
            }
        }

        /// Generates a batch of keys using the most efficient path available.
        #[allow(dead_code)]
        pub fn generate_batch(&self, batch_size: usize) -> Result<Vec<TorSecretKeyV3>, GpuError> {
            match &self.backend {
                Backend::Native(engine) => {
                    // For batch generation, generate keys in optimized manner
                    let mut results = Vec::with_capacity(batch_size);
                    for _ in 0..batch_size {
                        results.push(engine.generate_single()?);
                    }
                    Ok(results)
                }
                Backend::External(worker) => worker.generate_batch(batch_size),
            }
        }

        /// Provides the detected GPU name, useful for logging.
        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        /// Returns a human readable summary of the configured backend.
        pub fn label(&self) -> String {
            match &self.backend {
                Backend::Native(engine) => {
                    format!("{} via {}", self.device_name, engine.label())
                }
                Backend::External(worker) => {
                    format!("{} via {}", self.device_name, worker.display_label())
                }
            }
        }
    }

    impl CudaEngine {
        fn new() -> Result<Self, GpuError> {
            let api = cuda_api()?;

            unsafe {
                check(api, (api.cu_init)(0), "cuInit")?;

                let mut device: CUdevice = 0;
                check(api, (api.cu_device_get)(&mut device, 0), "cuDeviceGet")?;

                let mut context: CUcontext = ptr::null_mut();
                check(
                    api,
                    (api.cu_ctx_create)(&mut context, 0, device),
                    "cuCtxCreate_v2",
                )?;
                check(api, (api.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;

                let ptx = CString::new(PTX_SOURCE).expect("PTX string contains NUL byte");
                let mut module: CUmodule = ptr::null_mut();

                // Use cuModuleLoadDataEx with logging for better debugging
                let mut log = vec![0i8; 8192];
                let mut log_size: usize = log.len();
                let mut opts = [
                    CU_JIT_ERROR_LOG_BUFFER,
                    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                    CU_JIT_TARGET_FROM_CUCONTEXT,
                ];
                let mut optvals: [*mut c_void; 3] = [
                    log.as_mut_ptr() as *mut c_void,
                    &mut log_size as *mut _ as *mut c_void,
                    ptr::null_mut(),
                ];

                let result = (api.cu_module_load_data_ex)(
                    &mut module,
                    ptx.as_ptr() as *const c_void,
                    opts.len() as u32,
                    opts.as_mut_ptr(),
                    optvals.as_mut_ptr(),
                );

                if result != CUDA_SUCCESS {
                    // Print error log if compilation fails
                    let log_str = CStr::from_ptr(log.as_ptr()).to_string_lossy();
                    eprintln!("[#] PTX compilation failed. Error log:");
                    eprintln!("{}", log_str);
                    return Err(GpuError::InitializationFailed(format!(
                        "PTX compilation failed with error code {}: {}",
                        result, log_str
                    )));
                }

                let mut function: CUfunction = ptr::null_mut();
                let kernel_name = CString::new("generate_seeds").unwrap();
                check(
                    api,
                    (api.cu_module_get_function)(&mut function, module, kernel_name.as_ptr()),
                    "cuModuleGetFunction",
                )?;

                let mut device_buffer: CUdeviceptr = 0;
                let buffer_bytes = GPU_BATCH_SIZE * GPU_SEED_BYTES;
                check(
                    api,
                    (api.cu_mem_alloc)(&mut device_buffer, buffer_bytes),
                    "cuMemAlloc_v2",
                )?;

                let host_buffer = vec![0u8; buffer_bytes];

                Ok(Self {
                    api,
                    context,
                    module,
                    function,
                    device_buffer,
                    host_buffer,
                    cache: Mutex::new(VecDeque::new()),
                    seed_cursor: AtomicU64::new(0xA57C_EEED),
                    label: "CUDA native kernel".to_string(),
                })
            }
        }

        fn label(&self) -> &str {
            &self.label
        }

        fn generate_single(&mut self) -> Result<TorSecretKeyV3, GpuError> {
            {
                let mut cache = self.cache.lock().expect("GPU cache poisoned");
                if let Some(key) = cache.pop_front() {
                    return Ok(key);
                }
            }

            // Generate a larger batch to amortize GPU overhead
            let keys = self.generate_batch(GPU_BATCH_SIZE * 4)?; // 4x larger
            let mut cache = self.cache.lock().expect("GPU cache poisoned");
            cache.extend(keys);
            cache
                .pop_front()
                .ok_or_else(|| GpuError::ExecutionFailed("GPU cache unexpectedly empty.".into()))
        }

        fn generate_batch(&mut self, batch_size: usize) -> Result<Vec<TorSecretKeyV3>, GpuError> {
            if batch_size == 0 {
                return Ok(Vec::new());
            }

            let mut remaining = batch_size;
            let mut result = Vec::with_capacity(batch_size);

            while remaining > 0 {
                let chunk_size = remaining.min(GPU_BATCH_SIZE);

                unsafe {
                    self.launch_kernel(chunk_size)?;
                }

                let seed_bytes_slice = &self.host_buffer[..chunk_size * GPU_SEED_BYTES];
                let available_workers = num_cpus::get().max(1);
                let worker_count = available_workers.min(chunk_size.max(1));
                let chunk_per_worker = (chunk_size + worker_count - 1) / worker_count;
                let bytes_per_worker = chunk_per_worker * GPU_SEED_BYTES;
                let result_ref = &mut result;

                thread::scope(|scope| -> Result<(), GpuError> {
                    let mut handles = Vec::new();

                    for seeds_chunk in seed_bytes_slice.chunks(bytes_per_worker) {
                        handles.push(scope.spawn(
                            move || -> Result<Vec<TorSecretKeyV3>, GpuError> {
                                let mut local =
                                    Vec::with_capacity(seeds_chunk.len() / GPU_SEED_BYTES);

                                for seed_bytes in seeds_chunk.chunks_exact(GPU_SEED_BYTES) {
                                    let mut seed = [0u8; GPU_SEED_BYTES];
                                    seed.copy_from_slice(seed_bytes);
                                    let secret = SecretKey::from_bytes(&seed).map_err(|err| {
                                        GpuError::ExecutionFailed(format!(
                                            "Invalid seed produced by CUDA kernel: {}",
                                            err
                                        ))
                                    })?;
                                    let expanded = ExpandedSecretKey::from(&secret);
                                    local.push(TorSecretKeyV3::from(expanded.to_bytes()));
                                }

                                Ok(local)
                            },
                        ));
                    }

                    for handle in handles {
                        let partial = handle.join().map_err(|_| {
                            GpuError::ExecutionFailed(
                                "GPU conversion worker thread panicked.".into(),
                            )
                        })??;
                        result_ref.extend(partial);
                    }

                    Ok(())
                })?;

                remaining -= chunk_size;
            }

            Ok(result)
        }

        unsafe fn launch_kernel(&mut self, count: usize) -> Result<(), GpuError> {
            let api = self.api;
            // Optimization: use more threads per block for better GPU utilization
            let threads_per_block: c_uint = 512; // Increased from 256 to 512
            let grid_dim = ((count as c_uint) + threads_per_block - 1) / threads_per_block;
            let bytes_needed = count * GPU_SEED_BYTES;

            if self.host_buffer.len() < bytes_needed {
                return Err(GpuError::ExecutionFailed(format!(
                    "Host buffer too small for {} seeds",
                    count
                )));
            }

            let mut out_ptr = self.device_buffer;
            let mut base_seed = self
                .seed_cursor
                .fetch_add((count as u64) * (GPU_SEED_BYTES as u64), Ordering::SeqCst);
            let mut total = count as c_uint;

            let mut params = [
                &mut out_ptr as *mut CUdeviceptr as *mut c_void,
                &mut base_seed as *mut u64 as *mut c_void,
                &mut total as *mut c_uint as *mut c_void,
            ];

            check(
                api,
                (api.cu_ctx_set_current)(self.context),
                "cuCtxSetCurrent",
            )?;

            check(
                api,
                (api.cu_launch_kernel)(
                    self.function,
                    grid_dim,
                    1,
                    1,
                    threads_per_block,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;

            // Optimization: remove unnecessary synchronization for better performance
            // Only synchronize when absolutely necessary
            check(api, (api.cu_ctx_synchronize)(), "cuCtxSynchronize")?;

            check(
                api,
                (api.cu_memcpy_dtoh)(
                    self.host_buffer.as_mut_ptr() as *mut c_void,
                    self.device_buffer,
                    bytes_needed,
                ),
                "cuMemcpyDtoH_v2",
            )?;

            Ok(())
        }
    }

    impl NvidiaGpuEngine {
        fn from_backend(device_name: String, backend: Backend) -> Self {
            Self {
                device_name,
                backend,
            }
        }
    }

    #[derive(Debug)]
    struct GpuWorker {
        command: String,
        args: Vec<String>,
    }

    impl GpuWorker {
        fn new(command: String, args: Vec<String>) -> Self {
            Self { command, args }
        }

        fn display_label(&self) -> String {
            let mut repr = self.command.clone();
            for arg in &self.args {
                repr.push(' ');
                repr.push_str(arg);
            }
            repr
        }

        fn generate_single(&self) -> Result<TorSecretKeyV3, GpuError> {
            let mut cmd = Command::new(&self.command);
            cmd.args(&self.args);

            let output = cmd.output().map_err(|err| {
                GpuError::WorkerFailed(format!(
                    "Failed to execute GPU worker '{}': {}",
                    self.command, err
                ))
            })?;

            if !output.status.success() {
                return Err(GpuError::WorkerFailed(format!(
                    "GPU worker '{}' exited with status {}",
                    self.command, output.status
                )));
            }

            parse_worker_payload(&output.stdout)
        }

        #[allow(dead_code)]
        fn generate_batch(&self, batch_size: usize) -> Result<Vec<TorSecretKeyV3>, GpuError> {
            if batch_size <= 1 {
                return self.generate_single().map(|key| vec![key]);
            }

            let mut cmd = Command::new(&self.command);
            cmd.args(&self.args);
            cmd.arg("--batch");
            cmd.arg(batch_size.to_string());

            let output = cmd.output().map_err(|err| {
                GpuError::WorkerFailed(format!(
                    "Failed to execute GPU worker '{}' for batch run: {}",
                    self.command, err
                ))
            })?;

            if !output.status.success() {
                return Err(GpuError::WorkerFailed(format!(
                    "GPU worker '{}' exited with status {} during batch run",
                    self.command, output.status
                )));
            }

            parse_worker_batch(&output.stdout, batch_size)
        }
    }

    /// Attempts to detect whether an NVIDIA GPU is reachable.
    ///
    /// We try to execute `nvidia-smi` once.  If the binary is missing or the
    /// call fails the function silently returns `false` so the caller can
    /// gracefully fall back to CPU execution.
    pub fn is_nvidia_gpu_available() -> bool {
        if GPU_PROBED.swap(true, Ordering::SeqCst) {
            return GPU_AVAILABLE.load(Ordering::SeqCst);
        }

        let detected = match Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
        {
            Ok(output) if output.status.success() => {
                let name = String::from_utf8_lossy(&output.stdout);
                println!("[*] Detected NVIDIA GPU(s): {}", name.trim());
                true
            }
            _ => false,
        };

        GPU_AVAILABLE.store(detected, Ordering::SeqCst);
        detected
    }

    /// Initialises the NVIDIA backend, performing basic checks.
    pub fn initialize_nvidia_engine(
        worker_path: Option<String>,
        worker_args: Vec<String>,
    ) -> Result<NvidiaGpuEngine, GpuError> {
        if !is_nvidia_gpu_available() {
            return Err(GpuError::NotAvailable(
                "No NVIDIA GPU detected or `nvidia-smi` is unavailable.".into(),
            ));
        }

        // Try to capture a friendly device name for logging purposes.
        let device_name_output = Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map_err(|err| {
                GpuError::InitializationFailed(format!("Failed to execute `nvidia-smi`: {}", err))
            })?;

        if !device_name_output.status.success() {
            return Err(GpuError::InitializationFailed(format!(
                "`nvidia-smi` exited with status {} while querying GPU metadata.",
                device_name_output.status
            )));
        }

        let device_name = String::from_utf8_lossy(&device_name_output.stdout)
            .trim()
            .to_string();

        if let Some(command) = worker_path {
            let worker = GpuWorker::new(command, worker_args);
            let backend = Backend::External(std::sync::Arc::new(worker));
            return Ok(NvidiaGpuEngine::from_backend(device_name, backend));
        }

        let engine = OptimizedGpuEngine::new(device_name.clone()).map_err(|err| match err {
            GpuError::ExecutionFailed(msg) => {
                GpuError::InitializationFailed(format!("GPU runtime error: {}", msg))
            }
            other => other,
        })?;

        println!(
            "[*] CUDA runtime initialised with optimized batching on '{}'",
            device_name
        );

        Ok(NvidiaGpuEngine::from_backend(
            device_name,
            Backend::Native(std::sync::Arc::new(engine)),
        ))
    }

    fn parse_worker_payload(payload: &[u8]) -> Result<TorSecretKeyV3, GpuError> {
        let data = normalize_payload(payload)?;
        let mut key_bytes = [0u8; 64];
        key_bytes.copy_from_slice(&data[0..64]);
        Ok(TorSecretKeyV3::from(key_bytes))
    }

    #[allow(dead_code)]
    fn parse_worker_batch(
        payload: &[u8],
        expected: usize,
    ) -> Result<Vec<TorSecretKeyV3>, GpuError> {
        if payload.is_empty() {
            return Err(GpuError::WorkerProtocol(
                "GPU worker returned an empty payload while a batch was requested.".into(),
            ));
        }

        if payload.len() < expected * 64 {
            return Err(GpuError::WorkerProtocol(format!(
                "GPU worker returned {} bytes, but {} keys ({} bytes) were expected.",
                payload.len(),
                expected,
                expected * 64
            )));
        }

        let mut keys = Vec::with_capacity(expected);
        for chunk in payload.chunks_exact(64).take(expected) {
            let mut key_bytes = [0u8; 64];
            key_bytes.copy_from_slice(chunk);
            keys.push(TorSecretKeyV3::from(key_bytes));
        }

        if keys.len() < expected {
            return Err(GpuError::WorkerProtocol(format!(
                "GPU worker only returned {} keys out of the requested {}.",
                keys.len(),
                expected
            )));
        }

        Ok(keys)
    }

    fn normalize_payload(payload: &[u8]) -> Result<Vec<u8>, GpuError> {
        if payload.len() == 64 {
            return Ok(payload.to_vec());
        }

        if payload.len() == 65 && payload.ends_with(b"\n") {
            return Ok(payload[..64].to_vec());
        }

        if payload.is_empty() {
            return Err(GpuError::WorkerProtocol(
                "GPU worker produced no output.".into(),
            ));
        }

        Err(GpuError::WorkerProtocol(format!(
            "Unexpected payload size {} from GPU worker; expected 64 bytes.",
            payload.len()
        )))
    }

    fn cuda_api() -> Result<&'static CudaApi, GpuError> {
        unsafe {
            INIT.call_once(|| {
                CUDA_API = Some(load_cuda_api().expect("Failed to load CUDA API"));
            });
            Ok(CUDA_API.as_ref().expect("CUDA API not initialized"))
        }
    }

    unsafe fn load_cuda_api() -> Result<CudaApi, GpuError> {
        let lib = match Library::new("libcuda.so").or_else(|_| Library::new("libcuda.so.1")) {
            Ok(lib) => lib,
            Err(err) => {
                return Err(GpuError::NotAvailable(format!(
                    "Unable to load libcuda: {}",
                    err
                )))
            }
        };

        let lib_ref: &'static Library = Box::leak(Box::new(lib));

        unsafe fn symbol<T: Copy>(lib: &'static Library, name: &str) -> Result<T, GpuError> {
            let sym: libloading::Symbol<T> = lib.get(name.as_bytes()).map_err(|err| {
                GpuError::InitializationFailed(format!("Failed to load symbol {}: {}", name, err))
            })?;
            Ok(*sym)
        }

        Ok(CudaApi {
            lib: lib_ref,
            cu_init: symbol(lib_ref, "cuInit")?,
            cu_device_get: symbol(lib_ref, "cuDeviceGet")?,
            cu_ctx_create: symbol(lib_ref, "cuCtxCreate_v2")?,
            cu_ctx_destroy: symbol(lib_ref, "cuCtxDestroy_v2")?,
            cu_ctx_set_current: symbol(lib_ref, "cuCtxSetCurrent")?,
            cu_module_load_data: symbol(lib_ref, "cuModuleLoadData")?,
            cu_module_load_data_ex: symbol(lib_ref, "cuModuleLoadDataEx")?,
            cu_module_unload: symbol(lib_ref, "cuModuleUnload")?,
            cu_module_get_function: symbol(lib_ref, "cuModuleGetFunction")?,
            cu_mem_alloc: symbol(lib_ref, "cuMemAlloc_v2")?,
            cu_mem_free: symbol(lib_ref, "cuMemFree_v2")?,
            cu_launch_kernel: symbol(lib_ref, "cuLaunchKernel")?,
            cu_memcpy_dtoh: symbol(lib_ref, "cuMemcpyDtoH_v2")?,
            cu_ctx_synchronize: symbol(lib_ref, "cuCtxSynchronize")?,
            cu_get_error_string: match lib_ref.get(b"cuGetErrorString") {
                Ok(sym) => Some(*sym),
                Err(_) => None,
            },
        })
    }

    fn check(api: &CudaApi, result: CUresult, context: &str) -> Result<(), GpuError> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }

        Err(GpuError::ExecutionFailed(format!(
            "{} failed: {}",
            context,
            describe_error(api, result)
        )))
    }

    fn describe_error(api: &CudaApi, code: CUresult) -> String {
        if let Some(get_error) = api.cu_get_error_string {
            unsafe {
                let mut ptr: *const c_char = ptr::null();
                if (get_error)(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
                    if let Ok(msg) = CStr::from_ptr(ptr).to_str() {
                        return format!("{} (code {})", msg, code);
                    }
                }
            }
        }
        format!("CUDA error code {}", code)
    }
}
