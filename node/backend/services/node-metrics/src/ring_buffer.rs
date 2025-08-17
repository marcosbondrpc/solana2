use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::RwLock;

/// High-performance ring buffer for storing latency samples
/// Uses power-of-2 sizing for efficient modulo operations via bit masking
pub struct RingBuffer<T: Clone> {
    buffer: RwLock<Vec<Option<T>>>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    size: AtomicUsize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        // Round up to nearest power of 2 for efficient modulo
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;
        
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        
        Self {
            buffer: RwLock::new(buffer),
            capacity,
            mask,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
        }
    }
    
    /// Push a value to the ring buffer
    /// Returns the old value if the buffer was full
    pub fn push(&self, value: T) -> Option<T> {
        let mut buffer = self.buffer.write();
        
        let current_size = self.size.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        
        let old_value = if current_size >= self.capacity {
            // Buffer is full, overwrite oldest value
            let tail = self.tail.load(Ordering::Acquire);
            let old = buffer[tail].take();
            self.tail.store((tail + 1) & self.mask, Ordering::Release);
            old
        } else {
            // Buffer not full yet
            self.size.fetch_add(1, Ordering::AcqRel);
            None
        };
        
        buffer[head] = Some(value);
        self.head.store((head + 1) & self.mask, Ordering::Release);
        
        old_value
    }
    
    /// Get all values in the buffer
    pub fn get_all(&self) -> Vec<T> {
        let buffer = self.buffer.read();
        let size = self.size.load(Ordering::Acquire);
        
        if size == 0 {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(size);
        let tail = self.tail.load(Ordering::Acquire);
        
        for i in 0..size {
            let idx = (tail + i) & self.mask;
            if let Some(ref value) = buffer[idx] {
                result.push(value.clone());
            }
        }
        
        result
    }
    
    /// Get the most recent N values
    pub fn get_recent(&self, n: usize) -> Vec<T> {
        let buffer = self.buffer.read();
        let size = self.size.load(Ordering::Acquire);
        
        if size == 0 {
            return Vec::new();
        }
        
        let count = n.min(size);
        let mut result = Vec::with_capacity(count);
        let head = self.head.load(Ordering::Acquire);
        
        for i in 0..count {
            let idx = (head + self.capacity - 1 - i) & self.mask;
            if let Some(ref value) = buffer[idx] {
                result.push(value.clone());
            }
        }
        
        result.reverse();
        result
    }
    
    pub fn clear(&self) {
        let mut buffer = self.buffer.write();
        for item in buffer.iter_mut() {
            *item = None;
        }
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        self.size.store(0, Ordering::Release);
    }
    
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Acquire)
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_basic() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4);
        
        assert_eq!(buffer.capacity(), 4);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        
        // Push values
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.push(3), None);
        assert_eq!(buffer.push(4), None);
        
        assert_eq!(buffer.len(), 4);
        
        // Buffer full, should return old values
        assert_eq!(buffer.push(5), Some(1));
        assert_eq!(buffer.push(6), Some(2));
        
        let all = buffer.get_all();
        assert_eq!(all, vec![3, 4, 5, 6]);
    }
    
    #[test]
    fn test_ring_buffer_recent() {
        let buffer: RingBuffer<u64> = RingBuffer::new(8);
        
        for i in 1..=10 {
            buffer.push(i);
        }
        
        let recent = buffer.get_recent(3);
        assert_eq!(recent, vec![8, 9, 10]);
    }
}