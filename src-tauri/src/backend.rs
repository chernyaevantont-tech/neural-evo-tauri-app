// NOTE:
// Tauri CLI may merge features in dev mode and pass both `cuda-workers` and
// `wgpu-backend` to Cargo. In that case we intentionally prioritize CUDA below.

#[cfg(not(feature = "cuda-workers"))]
use burn::backend::wgpu::WgpuDevice;
#[cfg(not(feature = "cuda-workers"))]
use burn::backend::{Autodiff, Wgpu};

#[cfg(feature = "cuda-workers")]
use burn::backend::cuda::CudaDevice;
#[cfg(feature = "cuda-workers")]
use burn::backend::{Autodiff, Cuda};

#[cfg(not(feature = "cuda-workers"))]
pub type TrainBackend = Autodiff<Wgpu>;
#[cfg(feature = "cuda-workers")]
pub type TrainBackend = Autodiff<Cuda>;

#[cfg(not(feature = "cuda-workers"))]
pub type TrainDevice = WgpuDevice;
#[cfg(feature = "cuda-workers")]
pub type TrainDevice = CudaDevice;

pub fn create_device() -> TrainDevice {
    TrainDevice::default()
}

pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda-workers")]
    {
        "cuda"
    }

    #[cfg(not(feature = "cuda-workers"))]
    {
        "wgpu"
    }
}
