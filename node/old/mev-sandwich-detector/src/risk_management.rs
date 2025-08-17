use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::VecDeque;
use parking_lot::RwLock;
use tracing::{info, warn, error};

pub struct RiskManager {
    // Kill switches
    global_kill_switch: AtomicBool,
    auto_throttle_enabled: AtomicBool,
    
    // Risk limits
    max_position_size: AtomicU64,
    max_daily_loss: AtomicU64,
    max_consecutive_losses: AtomicU64,
    
    // Tracking
    daily_pnl: Arc<RwLock<i64>>,
    consecutive_losses: AtomicU64,
    rolling_trades: Arc<RwLock<VecDeque<TradeResult>>>,
    
    // Throttling
    throttle_factor: Arc<RwLock<f64>>,
}

#[derive(Clone)]
struct TradeResult {
    profit: i64,
    timestamp: std::time::Instant,
    confidence: f32,
}

impl RiskManager {
    pub fn new() -> Self {
        info!("Initializing Risk Manager with PnL watchdog");
        
        Self {
            global_kill_switch: AtomicBool::new(false),
            auto_throttle_enabled: AtomicBool::new(true),
            
            max_position_size: AtomicU64::new(10_000_000_000), // 10 SOL
            max_daily_loss: AtomicU64::new(5_000_000_000),     // 5 SOL
            max_consecutive_losses: AtomicU64::new(10),
            
            daily_pnl: Arc::new(RwLock::new(0)),
            consecutive_losses: AtomicU64::new(0),
            rolling_trades: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            
            throttle_factor: Arc::new(RwLock::new(1.0)),
        }
    }
    
    pub fn approve_trade(&self, expected_profit: u64, confidence: f32) -> bool {
        // Check kill switch
        if self.global_kill_switch.load(Ordering::Relaxed) {
            warn!("Trade rejected: Global kill switch activated");
            return false;
        }
        
        // Check position size
        if expected_profit > self.max_position_size.load(Ordering::Relaxed) {
            warn!("Trade rejected: Position size {} exceeds limit", expected_profit);
            return false;
        }
        
        // Check daily loss limit
        let daily_pnl = *self.daily_pnl.read();
        if daily_pnl < -(self.max_daily_loss.load(Ordering::Relaxed) as i64) {
            error!("Trade rejected: Daily loss limit exceeded");
            self.activate_kill_switch();
            return false;
        }
        
        // Check consecutive losses
        if self.consecutive_losses.load(Ordering::Relaxed) >= self.max_consecutive_losses.load(Ordering::Relaxed) {
            warn!("Trade rejected: Too many consecutive losses");
            return false;
        }
        
        // Apply throttling
        let throttle = *self.throttle_factor.read();
        if throttle < 1.0 {
            let random = rand::random::<f64>();
            if random > throttle {
                return false; // Randomly reject based on throttle
            }
        }
        
        // Confidence threshold with throttling
        let min_confidence = 0.75 * throttle as f32;
        if confidence < min_confidence {
            return false;
        }
        
        true
    }
    
    pub fn record_trade_result(&self, profit: i64) {
        // Update daily PnL
        {
            let mut daily = self.daily_pnl.write();
            *daily += profit;
        }
        
        // Update consecutive losses
        if profit < 0 {
            self.consecutive_losses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.consecutive_losses.store(0, Ordering::Relaxed);
        }
        
        // Add to rolling window
        {
            let mut trades = self.rolling_trades.write();
            trades.push_back(TradeResult {
                profit,
                timestamp: std::time::Instant::now(),
                confidence: 0.0, // Would be passed in
            });
            
            // Keep only last 10k trades
            if trades.len() > 10000 {
                trades.pop_front();
            }
        }
        
        // Update throttling
        self.update_throttle();
    }
    
    fn update_throttle(&self) {
        if !self.auto_throttle_enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let trades = self.rolling_trades.read();
        if trades.len() < 100 {
            return; // Not enough data
        }
        
        // Calculate recent win rate
        let recent: Vec<_> = trades.iter().rev().take(100).collect();
        let wins = recent.iter().filter(|t| t.profit > 0).count();
        let win_rate = wins as f64 / recent.len() as f64;
        
        // Calculate profit factor
        let total_wins: i64 = recent.iter().filter(|t| t.profit > 0).map(|t| t.profit).sum();
        let total_losses: i64 = recent.iter().filter(|t| t.profit < 0).map(|t| t.profit.abs()).sum();
        
        let profit_factor = if total_losses > 0 {
            total_wins as f64 / total_losses as f64
        } else {
            f64::MAX
        };
        
        // Update throttle based on performance
        let mut throttle = self.throttle_factor.write();
        
        if win_rate < 0.4 || profit_factor < 1.0 {
            // Poor performance - throttle down
            *throttle = (*throttle * 0.9).max(0.1);
            warn!("Throttling down to {:.2}x due to poor performance", *throttle);
        } else if win_rate > 0.6 && profit_factor > 1.5 {
            // Good performance - throttle up
            *throttle = (*throttle * 1.1).min(1.0);
            info!("Throttling up to {:.2}x due to good performance", *throttle);
        }
    }
    
    pub fn is_healthy(&self) -> bool {
        !self.global_kill_switch.load(Ordering::Relaxed) &&
        *self.throttle_factor.read() > 0.2
    }
    
    pub fn activate_kill_switch(&self) {
        error!("ACTIVATING GLOBAL KILL SWITCH");
        self.global_kill_switch.store(true, Ordering::Relaxed);
    }
    
    pub fn deactivate_kill_switch(&self) {
        info!("Deactivating global kill switch");
        self.global_kill_switch.store(false, Ordering::Relaxed);
    }
    
    pub fn reset_daily_stats(&self) {
        *self.daily_pnl.write() = 0;
        self.consecutive_losses.store(0, Ordering::Relaxed);
        info!("Daily risk stats reset");
    }
    
    pub fn get_stats(&self) -> RiskStats {
        let trades = self.rolling_trades.read();
        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.profit > 0).count();
        
        RiskStats {
            daily_pnl: *self.daily_pnl.read(),
            consecutive_losses: self.consecutive_losses.load(Ordering::Relaxed),
            throttle_factor: *self.throttle_factor.read(),
            total_trades,
            winning_trades,
            win_rate: if total_trades > 0 {
                winning_trades as f64 / total_trades as f64
            } else {
                0.0
            },
        }
    }
}

pub struct RiskStats {
    pub daily_pnl: i64,
    pub consecutive_losses: u64,
    pub throttle_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub win_rate: f64,
}