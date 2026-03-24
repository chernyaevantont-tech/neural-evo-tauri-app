use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::dtos::TrainingProfiler;

#[derive(Debug, Clone)]
pub struct ProfilerCollector {
    train_start_at: Option<Instant>,
    train_end_at: Option<Instant>,
    val_start_at: Option<Instant>,
    val_end_at: Option<Instant>,
    test_start_at: Option<Instant>,
    test_end_at: Option<Instant>,

    train_start_ms: Option<u64>,
    first_batch_ms: Option<u64>,
    train_end_ms: Option<u64>,
    val_start_ms: Option<u64>,
    val_end_ms: Option<u64>,
    test_start_ms: Option<u64>,
    test_end_ms: Option<u64>,

    peak_active_memory_mb: f32,
    batch_count: u32,
    train_samples: u64,
    inference_samples: u64,
    early_stop_epoch: Option<u32>,
}

impl ProfilerCollector {
    pub fn new() -> Self {
        Self {
            train_start_at: None,
            train_end_at: None,
            val_start_at: None,
            val_end_at: None,
            test_start_at: None,
            test_end_at: None,
            train_start_ms: None,
            first_batch_ms: None,
            train_end_ms: None,
            val_start_ms: None,
            val_end_ms: None,
            test_start_ms: None,
            test_end_ms: None,
            peak_active_memory_mb: 0.0,
            batch_count: 0,
            train_samples: 0,
            inference_samples: 0,
            early_stop_epoch: None,
        }
    }

    pub fn mark_train_start(&mut self) {
        self.train_start_at = Some(Instant::now());
        self.train_start_ms = Some(now_ms());
    }

    pub fn mark_first_batch(&mut self) {
        if self.first_batch_ms.is_none() {
            self.first_batch_ms = Some(now_ms());
        }
    }

    pub fn mark_train_end(&mut self) {
        self.train_end_at = Some(Instant::now());
        self.train_end_ms = Some(now_ms());
    }

    pub fn mark_val_start(&mut self) {
        self.val_start_at = Some(Instant::now());
        self.val_start_ms = Some(now_ms());
    }

    pub fn mark_val_end(&mut self) {
        self.val_end_at = Some(Instant::now());
        self.val_end_ms = Some(now_ms());
    }

    pub fn mark_test_start(&mut self) {
        self.test_start_at = Some(Instant::now());
        self.test_start_ms = Some(now_ms());
    }

    pub fn mark_test_end(&mut self) {
        self.test_end_at = Some(Instant::now());
        self.test_end_ms = Some(now_ms());
    }

    pub fn record_batch(&mut self, batch_size: usize) {
        self.batch_count = self.batch_count.saturating_add(1);
        self.train_samples = self.train_samples.saturating_add(batch_size as u64);
    }

    pub fn record_inference_samples(&mut self, sample_count: usize) {
        self.inference_samples = self.inference_samples.saturating_add(sample_count as u64);
    }

    pub fn update_peak_memory(&mut self, current_mb: f32) {
        if current_mb > self.peak_active_memory_mb {
            self.peak_active_memory_mb = current_mb;
        }
    }

    pub fn mark_early_stop_epoch(&mut self, epoch: usize) {
        self.early_stop_epoch = Some(epoch as u32);
    }

    pub fn finalize(&self) -> TrainingProfiler {
        let total_train_duration_ms = elapsed_ms(self.train_start_at, self.train_end_at);
        let val_duration_ms = elapsed_ms(self.val_start_at, self.val_end_at);
        let test_duration_ms = elapsed_ms(self.test_start_at, self.test_end_at);

        let samples_per_sec = if total_train_duration_ms > 0 {
            self.train_samples as f32 / (total_train_duration_ms as f32 / 1000.0)
        } else {
            0.0
        };

        let inference_msec_per_sample = if self.inference_samples > 0 {
            val_duration_ms as f32 / self.inference_samples as f32
        } else {
            0.0
        };

        TrainingProfiler {
            train_start_ms: self.train_start_ms.unwrap_or(0),
            first_batch_ms: self.first_batch_ms.unwrap_or(0),
            train_end_ms: self.train_end_ms.unwrap_or(0),
            total_train_duration_ms,
            val_start_ms: self.val_start_ms.unwrap_or(0),
            val_end_ms: self.val_end_ms.unwrap_or(0),
            val_duration_ms,
            test_start_ms: self.test_start_ms.unwrap_or(0),
            test_end_ms: self.test_end_ms.unwrap_or(0),
            test_duration_ms,
            peak_active_memory_mb: self.peak_active_memory_mb,
            peak_model_params_mb: 0.0,
            peak_gradient_mb: 0.0,
            peak_optim_state_mb: 0.0,
            peak_activation_mb: 0.0,
            samples_per_sec,
            inference_msec_per_sample,
            batch_count: self.batch_count,
            early_stop_epoch: self.early_stop_epoch,
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn elapsed_ms(start: Option<Instant>, end: Option<Instant>) -> u64 {
    match (start, end) {
        (Some(s), Some(e)) => e.duration_since(s).as_millis() as u64,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = ProfilerCollector::new();
        let result = profiler.finalize();
        assert_eq!(result.batch_count, 0);
    }

    #[test]
    fn test_batch_counting() {
        let mut profiler = ProfilerCollector::new();
        profiler.mark_train_start();
        for _ in 0..50 {
            profiler.record_batch(32);
        }
        profiler.mark_train_end();

        let result = profiler.finalize();
        assert_eq!(result.batch_count, 50);
    }

    #[test]
    fn test_throughput_calculation() {
        let mut profiler = ProfilerCollector::new();
        profiler.mark_train_start();
        for _ in 0..100 {
            profiler.record_batch(16);
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
        profiler.mark_train_end();

        let result = profiler.finalize();
        assert!(result.samples_per_sec > 0.0);
        assert!(result.total_train_duration_ms >= 150);
    }
}
