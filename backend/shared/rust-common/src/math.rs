use ethnum::U256;
use rust_decimal::Decimal;

pub fn calculate_profit(amount_in: U256, amount_out: U256) -> U256 {
    if amount_out > amount_in {
        amount_out - amount_in
    } else {
        U256::ZERO
    }
}