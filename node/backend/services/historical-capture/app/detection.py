"""
Advanced MEV detection algorithms
Arbitrage and sandwich attack identification with high precision
"""

import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import asyncio
from dataclasses import dataclass
import uuid


@dataclass
class SwapInfo:
    """Normalized swap information"""
    signature: str
    slot: int
    instruction_index: int
    program_id: str
    pool_address: str
    user: str
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    price: float
    timestamp: Optional[int] = None


class MEVDetector:
    """
    Production-grade MEV detection engine
    Identifies arbitrage and sandwich attacks with high accuracy
    """
    
    # Known DEX program IDs on Solana
    DEX_PROGRAMS = {
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium",
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca Whirlpool",
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca V2",
        "PHoeNiX582Ywqzb2x9B5pSkgCDPoKEpDNk2Q5UVEXx": "Phoenix",
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter V6",
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "Jupiter V4",
        "SSwapUtytfBdBn1b9NUGG6foMVPtcWgpRU32HToDUZr": "Saros",
        "Dooar9JkhdZ7J3LHN3A7YCuoGRUggXhQaG4kijfLGU2": "Stepn",
        "MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky": "Mercurial"
    }
    
    # Token price feeds (in production, use real oracle data)
    TOKEN_PRICES = {
        "So11111111111111111111111111111111111111112": 100.0,  # SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": 1.0,   # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": 1.0,   # USDT
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": 1700.0, # stSOL
        "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": 95.0,    # mSOL
    }
    
    def __init__(self, min_profit_usd: float = 1.0):
        self.min_profit_usd = min_profit_usd
        
    def extract_swaps_from_logs(
        self,
        logs: List[str],
        signature: str,
        slot: int,
        program_id: str
    ) -> List[SwapInfo]:
        """Extract normalized swap events from transaction logs"""
        swaps = []
        
        for idx, log in enumerate(logs):
            swap = self._parse_swap_log(log, signature, slot, idx, program_id)
            if swap:
                swaps.append(swap)
                
        return swaps
        
    def _parse_swap_log(
        self,
        log: str,
        signature: str,
        slot: int,
        idx: int,
        program_id: str
    ) -> Optional[SwapInfo]:
        """Parse swap information from log message"""
        # Raydium pattern
        if "ray_log" in log.lower():
            try:
                parts = log.split()
                if len(parts) >= 8:
                    return SwapInfo(
                        signature=signature,
                        slot=slot,
                        instruction_index=idx,
                        program_id=program_id,
                        pool_address=parts[1],
                        user=parts[2],
                        token_in=parts[3],
                        token_out=parts[4],
                        amount_in=int(parts[5]),
                        amount_out=int(parts[6]),
                        price=float(parts[6]) / float(parts[5]) if float(parts[5]) > 0 else 0
                    )
            except:
                pass
                
        # Orca pattern
        elif "Swap executed" in log:
            try:
                # Parse Orca-specific format
                import re
                amounts = re.findall(r'\d+', log)
                if len(amounts) >= 2:
                    return SwapInfo(
                        signature=signature,
                        slot=slot,
                        instruction_index=idx,
                        program_id=program_id,
                        pool_address="",  # Extract from instruction data
                        user="",
                        token_in="",
                        token_out="",
                        amount_in=int(amounts[0]),
                        amount_out=int(amounts[1]),
                        price=float(amounts[1]) / float(amounts[0]) if float(amounts[0]) > 0 else 0
                    )
            except:
                pass
                
        return None
        
    def detect_arbitrage(
        self,
        transactions: List[Dict[str, Any]],
        max_slot_gap: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect arbitrage opportunities across multiple DEXes
        Uses graph-based cycle detection for optimal path finding
        """
        opportunities = []
        
        # Group transactions by slot window
        slot_windows = defaultdict(list)
        for tx in transactions:
            slot = tx.get("slot", 0)
            window = slot // max_slot_gap
            slot_windows[window].append(tx)
            
        # Analyze each window for arbitrage
        for window, window_txs in slot_windows.items():
            # Extract all swaps in window
            swaps_by_user = defaultdict(list)
            
            for tx in window_txs:
                if not tx.get("meta", {}).get("logMessages"):
                    continue
                    
                user = self._get_tx_signer(tx)
                if not user:
                    continue
                    
                # Extract swaps from transaction
                logs = tx["meta"]["logMessages"]
                for instruction in tx.get("transaction", {}).get("message", {}).get("instructions", []):
                    program_id = instruction.get("programId", "")
                    if program_id in self.DEX_PROGRAMS:
                        swaps = self.extract_swaps_from_logs(
                            logs,
                            tx.get("transaction", {}).get("signatures", [""])[0],
                            tx["slot"],
                            program_id
                        )
                        swaps_by_user[user].extend(swaps)
                        
            # Check each user's swaps for arbitrage
            for user, user_swaps in swaps_by_user.items():
                if len(user_swaps) < 2:
                    continue
                    
                # Build token flow graph
                arb = self._find_arbitrage_cycle(user_swaps)
                if arb and arb["profit_usd"] >= self.min_profit_usd:
                    opportunities.append(arb)
                    
        return opportunities
        
    def _find_arbitrage_cycle(
        self,
        swaps: List[SwapInfo]
    ) -> Optional[Dict[str, Any]]:
        """Find profitable arbitrage cycle in swap sequence"""
        if len(swaps) < 2:
            return None
            
        # Build directed graph of token flows
        graph = defaultdict(list)
        swap_map = {}
        
        for swap in swaps:
            graph[swap.token_in].append((swap.token_out, swap))
            swap_map[f"{swap.token_in}->{swap.token_out}"] = swap
            
        # Find cycles using DFS
        visited = set()
        cycles = []
        
        def dfs(token, path, start_token):
            if token == start_token and len(path) > 1:
                cycles.append(path[:])
                return
                
            if len(path) > 5:  # Max cycle length
                return
                
            for next_token, swap in graph.get(token, []):
                edge = f"{token}->{next_token}"
                if edge not in visited or (next_token == start_token and len(path) > 1):
                    visited.add(edge)
                    path.append(swap)
                    dfs(next_token, path, start_token)
                    path.pop()
                    visited.remove(edge)
                    
        # Start DFS from each token
        for start_token in graph.keys():
            dfs(start_token, [], start_token)
            
        # Evaluate each cycle for profitability
        best_cycle = None
        best_profit = 0
        
        for cycle in cycles:
            profit = self._calculate_cycle_profit(cycle)
            if profit > best_profit:
                best_profit = profit
                best_cycle = cycle
                
        if best_cycle and best_profit >= self.min_profit_usd:
            return {
                "id": str(uuid.uuid4()),
                "start_slot": best_cycle[0].slot,
                "end_slot": best_cycle[-1].slot,
                "transactions": [s.signature for s in best_cycle],
                "path": [s.pool_address for s in best_cycle],
                "tokens": [s.token_in for s in best_cycle] + [best_cycle[-1].token_out],
                "profit_token": best_cycle[0].token_in,
                "profit_amount": int(best_profit * 1e6),  # Convert to lamports
                "profit_usd": best_profit,
                "gas_used": len(best_cycle) * 5000,  # Estimate
                "net_profit_usd": best_profit - (len(best_cycle) * 0.00025 * 100),  # Gas cost
                "detected_at": datetime.utcnow()
            }
            
        return None
        
    def _calculate_cycle_profit(self, cycle: List[SwapInfo]) -> float:
        """Calculate USD profit from arbitrage cycle"""
        if not cycle:
            return 0
            
        # Start with 1 unit of starting token
        amount = 1.0
        start_token = cycle[0].token_in
        
        # Follow the cycle
        for swap in cycle:
            # Apply swap ratio
            amount = amount * (swap.amount_out / swap.amount_in if swap.amount_in > 0 else 0)
            
        # Calculate profit if we end with same token
        if cycle[-1].token_out == start_token:
            profit_ratio = amount - 1.0
            
            # Convert to USD
            token_price = self.TOKEN_PRICES.get(start_token, 1.0)
            return profit_ratio * token_price
            
        return 0
        
    def detect_sandwich_attacks(
        self,
        transactions: List[Dict[str, Any]],
        max_slot_gap: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Detect sandwich attacks using A-V-B pattern matching
        High precision detection with slippage analysis
        """
        attacks = []
        
        # Group by slot for efficient searching
        txs_by_slot = defaultdict(list)
        for tx in transactions:
            txs_by_slot[tx.get("slot", 0)].append(tx)
            
        # Look for sandwich patterns
        for slot, slot_txs in txs_by_slot.items():
            if len(slot_txs) < 3:
                continue
                
            # Extract swaps from each transaction
            swaps_in_slot = []
            for tx in slot_txs:
                if not tx.get("meta", {}).get("logMessages"):
                    continue
                    
                sig = tx.get("transaction", {}).get("signatures", [""])[0]
                logs = tx["meta"]["logMessages"]
                
                for instruction in tx.get("transaction", {}).get("message", {}).get("instructions", []):
                    program_id = instruction.get("programId", "")
                    if program_id in self.DEX_PROGRAMS:
                        swaps = self.extract_swaps_from_logs(logs, sig, slot, program_id)
                        for swap in swaps:
                            swap.tx = tx  # Attach transaction for analysis
                            swaps_in_slot.append(swap)
                            
            # Check for sandwich pattern
            sandwiches = self._find_sandwich_pattern(swaps_in_slot)
            attacks.extend(sandwiches)
            
        return attacks
        
    def _find_sandwich_pattern(
        self,
        swaps: List[SwapInfo]
    ) -> List[Dict[str, Any]]:
        """Identify A-V-B sandwich pattern in swaps"""
        sandwiches = []
        
        # Group swaps by pool
        swaps_by_pool = defaultdict(list)
        for swap in swaps:
            swaps_by_pool[swap.pool_address].append(swap)
            
        # Check each pool for sandwich pattern
        for pool, pool_swaps in swaps_by_pool.items():
            if len(pool_swaps) < 3:
                continue
                
            # Sort by instruction index (execution order)
            pool_swaps.sort(key=lambda x: (x.slot, x.instruction_index))
            
            # Look for A-V-B pattern
            for i in range(len(pool_swaps) - 2):
                front = pool_swaps[i]
                victim = pool_swaps[i + 1]
                back = pool_swaps[i + 2]
                
                # Check if same attacker and opposite trades
                if (front.user == back.user and 
                    front.user != victim.user and
                    front.token_in == back.token_out and
                    front.token_out == back.token_in):
                    
                    # Calculate profits
                    attacker_profit = self._calculate_sandwich_profit(front, victim, back)
                    victim_loss = self._calculate_victim_loss(victim, front)
                    
                    if attacker_profit >= self.min_profit_usd:
                        sandwiches.append({
                            "id": str(uuid.uuid4()),
                            "front_tx": front.signature,
                            "victim_tx": victim.signature,
                            "back_tx": back.signature,
                            "slot": front.slot,
                            "pool_address": pool,
                            "attacker": front.user,
                            "victim": victim.user,
                            "profit_token": front.token_in,
                            "profit_amount": int(attacker_profit * 1e6),
                            "profit_usd": attacker_profit,
                            "victim_loss_usd": victim_loss,
                            "detected_at": datetime.utcnow()
                        })
                        
        return sandwiches
        
    def _calculate_sandwich_profit(
        self,
        front: SwapInfo,
        victim: SwapInfo,
        back: SwapInfo
    ) -> float:
        """Calculate attacker profit from sandwich"""
        # Front-run: Buy token before victim
        tokens_bought = front.amount_out
        cost = front.amount_in
        
        # Back-run: Sell token after victim at higher price
        tokens_sold = back.amount_in
        revenue = back.amount_out
        
        # Profit = revenue - cost (in input token)
        if tokens_bought == tokens_sold:
            profit = revenue - cost
            
            # Convert to USD
            token_price = self.TOKEN_PRICES.get(front.token_in, 1.0)
            return (profit / 1e9) * token_price  # Convert from lamports
            
        return 0
        
    def _calculate_victim_loss(
        self,
        victim: SwapInfo,
        front: SwapInfo
    ) -> float:
        """Calculate victim's loss due to sandwich"""
        # Estimate price impact from front-run
        price_before = front.price
        price_victim = victim.price
        
        if price_before > 0:
            slippage = (price_victim - price_before) / price_before
            
            # Calculate loss in USD
            victim_amount_usd = (victim.amount_in / 1e9) * self.TOKEN_PRICES.get(victim.token_in, 1.0)
            return abs(slippage * victim_amount_usd)
            
        return 0
        
    def _get_tx_signer(self, tx: Dict[str, Any]) -> Optional[str]:
        """Extract transaction signer (fee payer)"""
        try:
            accounts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
            if accounts:
                return accounts[0].get("pubkey", "")
        except:
            pass
        return None