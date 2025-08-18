use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::marker::PhantomData;

/// Lock-free single-producer single-consumer ring buffer
/// Optimized for MEV hot path with zero allocations
pub struct SpscRing<T> {
    buffer: Arc<RingBuffer<T>>,
    capacity_mask: usize,
}

struct RingBuffer<T> {
    data: Vec<UnsafeCell<Option<T>>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    cached_head: UnsafeCell<usize>,
    cached_tail: UnsafeCell<usize>,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> SpscRing<T> {
    /// Create new ring buffer with power-of-2 capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");
        
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(UnsafeCell::new(None));
        }
        
        Self {
            buffer: Arc::new(RingBuffer {
                data,
                head: AtomicUsize::new(0),
                tail: AtomicUsize::new(0),
                cached_head: UnsafeCell::new(0),
                cached_tail: UnsafeCell::new(0),
            }),
            capacity_mask: capacity - 1,
        }
    }
    
    /// Create producer handle
    pub fn producer(&self) -> Producer<T> {
        Producer {
            buffer: Arc::clone(&self.buffer),
            capacity_mask: self.capacity_mask,
            _phantom: PhantomData,
        }
    }
    
    /// Create consumer handle
    pub fn consumer(&self) -> Consumer<T> {
        Consumer {
            buffer: Arc::clone(&self.buffer),
            capacity_mask: self.capacity_mask,
            _phantom: PhantomData,
        }
    }
}

pub struct Producer<T> {
    buffer: Arc<RingBuffer<T>>,
    capacity_mask: usize,
    _phantom: PhantomData<*mut T>, // !Send + !Sync
}

pub struct Consumer<T> {
    buffer: Arc<RingBuffer<T>>,
    capacity_mask: usize,
    _phantom: PhantomData<*mut T>, // !Send + !Sync
}

impl<T> Producer<T> {
    /// Try to push value, returns false if full
    #[inline(always)]
    pub fn try_push(&self, value: T) -> bool {
        let tail = self.buffer.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.capacity_mask;
        
        // Check cached head first to avoid atomic load
        let cached_head = unsafe { *self.buffer.cached_head.get() };
        if next_tail == cached_head {
            // Update cache and check again
            let head = self.buffer.head.load(Ordering::Acquire);
            unsafe { *self.buffer.cached_head.get() = head; }
            if next_tail == head {
                return false; // Buffer full
            }
        }
        
        // Write value
        unsafe {
            let slot = &*self.buffer.data[tail].get();
            *slot.as_ptr() = Some(value);
        }
        
        // Update tail with release semantics
        self.buffer.tail.store(next_tail, Ordering::Release);
        true
    }
    
    /// Push value, spinning if full (use carefully in hot path)
    #[inline(always)]
    pub fn push(&self, value: T) {
        while !self.try_push(value) {
            std::hint::spin_loop();
        }
    }
    
    /// Check if buffer is full
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let tail = self.buffer.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.capacity_mask;
        let head = self.buffer.head.load(Ordering::Acquire);
        next_tail == head
    }
}

impl<T> Consumer<T> {
    /// Try to pop value, returns None if empty
    #[inline(always)]
    pub fn try_pop(&self) -> Option<T> {
        let head = self.buffer.head.load(Ordering::Relaxed);
        
        // Check cached tail first
        let cached_tail = unsafe { *self.buffer.cached_tail.get() };
        if head == cached_tail {
            // Update cache and check again
            let tail = self.buffer.tail.load(Ordering::Acquire);
            unsafe { *self.buffer.cached_tail.get() = tail; }
            if head == tail {
                return None; // Buffer empty
            }
        }
        
        // Read value
        let value = unsafe {
            let slot = &*self.buffer.data[head].get();
            (*slot.as_ptr()).take()
        };
        
        // Update head with release semantics
        let next_head = (head + 1) & self.capacity_mask;
        self.buffer.head.store(next_head, Ordering::Release);
        
        value
    }
    
    /// Pop value, spinning if empty (use carefully)
    #[inline(always)]
    pub fn pop(&self) -> T {
        loop {
            if let Some(value) = self.try_pop() {
                return value;
            }
            std::hint::spin_loop();
        }
    }
    
    /// Check if buffer is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let head = self.buffer.head.load(Ordering::Relaxed);
        let tail = self.buffer.tail.load(Ordering::Acquire);
        head == tail
    }
    
    /// Drain all available items
    pub fn drain(&self) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = self.try_pop() {
            items.push(item);
        }
        items
    }
}

/// High-performance multi-producer single-consumer queue
pub struct MpscQueue<T> {
    inner: crossbeam_queue::ArrayQueue<T>,
}

impl<T> MpscQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: crossbeam_queue::ArrayQueue::new(capacity),
        }
    }
    
    #[inline(always)]
    pub fn push(&self, value: T) -> Result<(), T> {
        self.inner.push(value)
    }
    
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        self.inner.pop()
    }
    
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_spsc_basic() {
        let ring = SpscRing::new(256);
        let producer = ring.producer();
        let consumer = ring.consumer();
        
        assert!(producer.try_push(42));
        assert_eq!(consumer.try_pop(), Some(42));
        assert_eq!(consumer.try_pop(), None);
    }
    
    #[test]
    fn test_spsc_concurrent() {
        let ring = SpscRing::new(1024);
        let producer = ring.producer();
        let consumer = ring.consumer();
        
        let producer_thread = thread::spawn(move || {
            for i in 0..10000 {
                producer.push(i);
            }
        });
        
        let consumer_thread = thread::spawn(move || {
            let mut sum = 0u64;
            for _ in 0..10000 {
                sum += consumer.pop() as u64;
            }
            sum
        });
        
        producer_thread.join().unwrap();
        let sum = consumer_thread.join().unwrap();
        
        // Sum of 0..10000
        assert_eq!(sum, 49995000);
    }
}