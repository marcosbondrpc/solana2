use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::hint::spin_loop;

#[inline]
fn is_pow2(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// A minimal MPMC ring buffer optimized for batch operations.
/// Capacity must be a power of two.
pub struct MpmcRing<T> {
    mask: usize,
    buf: Vec<UnsafeCell<Option<T>>>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Send> Send for MpmcRing<T> {}
unsafe impl<T: Send> Sync for MpmcRing<T> {}

impl<T> MpmcRing<T> {
    pub fn with_capacity_pow2(cap: usize) -> Self {
        assert!(is_pow2(cap), "capacity must be a power of two");
        let mut buf = Vec::with_capacity(cap);
        for _ in 0..cap {
            buf.push(UnsafeCell::new(None));
        }
        Self {
            mask: cap - 1,
            buf,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        h.wrapping_sub(t)
    }

    /// Try to push a batch; returns number actually pushed.
    pub fn push_many<I: IntoIterator<Item = T>>(&self, items: I) -> usize {
        let mut n = 0;
        for it in items {
            let h = self.head.load(Ordering::Relaxed);
            let t = self.tail.load(Ordering::Acquire);
            if h.wrapping_sub(t) == self.capacity() {
                break;
            }
            let idx = h & self.mask;
            unsafe {
                *self.buf[idx].get() = Some(it);
            }
            self.head.store(h.wrapping_add(1), Ordering::Release);
            n += 1;
        }
        n
    }

    /// Pop up to `max` items, passing each to `out`. Returns number popped.
    pub fn pop_many(&self, max: usize, mut out: impl FnMut(T)) -> usize {
        let mut n = 0;
        while n < max {
            let t = self.tail.load(Ordering::Relaxed);
            let h = self.head.load(Ordering::Acquire);
            if t == h {
                break;
            }
            let idx = t & self.mask;
            let val = unsafe { (*self.buf[idx].get()).take() };
            self.tail.store(t.wrapping_add(1), Ordering::Release);
            if let Some(v) = val {
                out(v);
                n += 1;
            } else {
                spin_loop();
            }
        }
        n
    }
}


