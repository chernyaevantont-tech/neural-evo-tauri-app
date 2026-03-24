use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct Scheduler {
    total_budget_mb: Arc<AtomicU64>,
    reserved_vram_mb: Arc<AtomicU64>,
    safety_margin_mb: u64,
}

impl Scheduler {
    pub fn new(total_budget_mb: u64, safety_margin_mb: u64) -> Self {
        Self {
            total_budget_mb: Arc::new(AtomicU64::new(total_budget_mb)),
            reserved_vram_mb: Arc::new(AtomicU64::new(0)),
            safety_margin_mb,
        }
    }

    pub fn available_vram_mb(&self) -> u64 {
        let total = self.total_budget_mb.load(Ordering::Relaxed);
        let reserved = self.reserved_vram_mb.load(Ordering::Relaxed);
        let usable = total.saturating_sub(self.safety_margin_mb);
        usable.saturating_sub(reserved)
    }

    pub fn try_reserve(&self, needed_mb: u64) -> bool {
        if self.available_vram_mb() >= needed_mb {
            let current = self.reserved_vram_mb.load(Ordering::Relaxed);
            let new_reserved = current.saturating_add(needed_mb);

            if new_reserved <= self.total_budget_mb.load(Ordering::Relaxed) {
                match self.reserved_vram_mb.compare_exchange(
                    current,
                    new_reserved,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(_) => return self.try_reserve(needed_mb), // Retry on contention
                }
            }
        }
        false
    }

    pub fn release(&self, freed_mb: u64) {
        self.reserved_vram_mb
            .fetch_sub(freed_mb, Ordering::Release);
    }

    pub async fn attempt_schedule(&self, _run_id: &str) -> Result<(), String> {
        // Placeholder: actual scheduling logic in later tasks
        Ok(())
    }

    pub async fn cancel_active_jobs(&self, _run_id: &str) -> Result<(), String> {
        // Placeholder: actual cancellation logic in later tasks
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let sched = Scheduler::new(8000, 500);
        assert_eq!(sched.available_vram_mb(), 7500);
    }

    #[test]
    fn test_reserve_and_release() {
        let sched = Scheduler::new(8000, 500);

        assert!(sched.try_reserve(1000));
        assert_eq!(sched.available_vram_mb(), 6500);

        sched.release(1000);
        assert_eq!(sched.available_vram_mb(), 7500);
    }

    #[test]
    fn test_reserve_overflow_protection() {
        let sched = Scheduler::new(8000, 500);
        let available = sched.available_vram_mb();
        
        // Try to reserve more than available
        assert!(!sched.try_reserve(available + 1000));
    }

    #[test]
    fn test_multiple_reserves() {
        let sched = Scheduler::new(8000, 500);

        assert!(sched.try_reserve(2000));
        assert!(sched.try_reserve(2000));
        assert!(sched.try_reserve(2000));
        // After 6000 reserved, only 1500 available
        assert!(!sched.try_reserve(2000));

        sched.release(2000);
        assert!(sched.try_reserve(1000));
    }
}
