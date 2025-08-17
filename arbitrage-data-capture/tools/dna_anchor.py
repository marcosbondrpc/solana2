#!/usr/bin/env python3
"""
Daily Merkle Anchor for Decision DNA
Anchors decision lineage to Solana blockchain for immutable audit trail
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import base58
import base64

import clickhouse_connect
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.transaction import Transaction
from solana.system_program import create_account, CreateAccountParams
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash as SolanaHash
from solders.message import MessageV0
from solders.transaction import VersionedTransaction

# For Merkle tree construction
from hashlib import sha256


@dataclass
class DecisionBatch:
    """Batch of decisions to anchor"""
    start_time: datetime
    end_time: datetime
    decision_count: int
    decision_hashes: List[str]
    merkle_root: str
    total_value: float
    success_rate: float


class MerkleTree:
    """Merkle tree builder for decision DNA"""
    
    @staticmethod
    def compute_root(leaves: List[bytes]) -> bytes:
        """Compute merkle root from leaf hashes"""
        if not leaves:
            return b'\x00' * 32
        
        # Copy leaves to working array
        level = [leaf for leaf in leaves]
        
        # Build tree bottom-up
        while len(level) > 1:
            next_level = []
            
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Hash pair
                    combined = level[i] + level[i + 1]
                else:
                    # Odd number, duplicate last
                    combined = level[i] + level[i]
                
                next_level.append(sha256(combined).digest())
            
            level = next_level
        
        return level[0]
    
    @staticmethod
    def generate_proof(leaves: List[bytes], index: int) -> List[bytes]:
        """Generate merkle proof for leaf at index"""
        if not leaves or index >= len(leaves):
            return []
        
        proof = []
        level = [leaf for leaf in leaves]
        current_index = index
        
        while len(level) > 1:
            next_level = []
            
            for i in range(0, len(level), 2):
                if i == current_index or i + 1 == current_index:
                    # This pair contains our target
                    if i == current_index and i + 1 < len(level):
                        # Target is left, add right to proof
                        proof.append(level[i + 1])
                    elif i + 1 == current_index:
                        # Target is right, add left to proof
                        proof.append(level[i])
                    
                    # Update index for next level
                    current_index = i // 2
                
                # Compute parent hash
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                
                next_level.append(sha256(combined).digest())
            
            level = next_level
        
        return proof
    
    @staticmethod
    def verify_proof(leaf: bytes, proof: List[bytes], root: bytes) -> bool:
        """Verify merkle proof"""
        current = leaf
        
        for sibling in proof:
            # Order matters - smaller value goes first
            if current < sibling:
                combined = current + sibling
            else:
                combined = sibling + current
            
            current = sha256(combined).digest()
        
        return current == root


class DNAAnchor:
    """Anchors decision DNA to Solana blockchain"""
    
    # Anchor program ID (would be your deployed program)
    ANCHOR_PROGRAM_ID = Pubkey.from_string("DNAnchorXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    def __init__(
        self,
        rpc_url: str,
        clickhouse_url: str,
        keypair_path: str,
        dry_run: bool = False
    ):
        self.rpc_url = rpc_url
        self.clickhouse_url = clickhouse_url
        self.keypair_path = keypair_path
        self.dry_run = dry_run
        self.client: Optional[AsyncClient] = None
        self.ch_client: Optional[Any] = None
        self.keypair: Optional[Keypair] = None
    
    async def setup(self):
        """Initialize connections"""
        # Connect to Solana
        self.client = AsyncClient(self.rpc_url)
        
        # Load keypair
        with open(self.keypair_path, 'r') as f:
            secret_key = json.load(f)
        self.keypair = Keypair.from_secret_key(bytes(secret_key))
        
        # Connect to ClickHouse
        self.ch_client = clickhouse_connect.get_client(
            host=self.clickhouse_url.replace("http://", "").split(":")[0],
            port=8123,
            database="legendary_mev"
        )
        
        print(f"[✓] Connected to Solana: {self.rpc_url}")
        print(f"[✓] Anchor account: {self.keypair.public_key}")
    
    async def fetch_decisions(self, start_time: datetime, end_time: datetime) -> DecisionBatch:
        """Fetch decisions from ClickHouse for anchoring"""
        print(f"[*] Fetching decisions from {start_time} to {end_time}...")
        
        # Query decision lineage
        query = """
        SELECT 
            decision_id,
            decision_hash,
            confidence,
            expected_value,
            actual_value
        FROM decision_lineage
        WHERE created_at >= %(start_time)s 
          AND created_at < %(end_time)s
          AND anchored_at IS NULL
        ORDER BY created_at
        LIMIT 10000
        """
        
        result = self.ch_client.query(
            query,
            parameters={
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        )
        
        if not result.result_rows:
            print("[!] No decisions to anchor")
            return None
        
        decision_hashes = []
        total_value = 0.0
        successful = 0
        
        for row in result.result_rows:
            decision_id, decision_hash, confidence, expected_value, actual_value = row
            decision_hashes.append(decision_hash)
            
            if actual_value:
                total_value += float(actual_value)
                if float(actual_value) > 0:
                    successful += 1
        
        # Compute merkle root
        leaves = [bytes.fromhex(h) for h in decision_hashes]
        merkle_root = MerkleTree.compute_root(leaves)
        
        success_rate = successful / len(decision_hashes) if decision_hashes else 0.0
        
        batch = DecisionBatch(
            start_time=start_time,
            end_time=end_time,
            decision_count=len(decision_hashes),
            decision_hashes=decision_hashes,
            merkle_root=merkle_root.hex(),
            total_value=total_value,
            success_rate=success_rate
        )
        
        print(f"[✓] Found {batch.decision_count} decisions")
        print(f"    Total value: ${batch.total_value:,.2f}")
        print(f"    Success rate: {batch.success_rate:.2%}")
        print(f"    Merkle root: {batch.merkle_root[:16]}...")
        
        return batch
    
    async def anchor_to_solana(self, batch: DecisionBatch) -> Optional[str]:
        """Anchor merkle root to Solana blockchain"""
        if self.dry_run:
            print("[*] DRY RUN - Would anchor to Solana:")
            print(f"    Merkle root: {batch.merkle_root}")
            print(f"    Decisions: {batch.decision_count}")
            return "dry_run_signature"
        
        print("[*] Anchoring to Solana...")
        
        try:
            # Create anchor data
            anchor_data = {
                "version": 1,
                "timestamp": int(time.time()),
                "merkle_root": batch.merkle_root,
                "decision_count": batch.decision_count,
                "total_value_usd": int(batch.total_value * 100),  # Store as cents
                "success_rate_bps": int(batch.success_rate * 10000),  # Basis points
                "start_time": int(batch.start_time.timestamp()),
                "end_time": int(batch.end_time.timestamp()),
            }
            
            # Serialize to bytes
            anchor_bytes = json.dumps(anchor_data).encode('utf-8')
            
            # Create memo instruction (simplified - in production use proper anchor program)
            memo_instruction = self.create_memo_instruction(anchor_bytes)
            
            # Get recent blockhash
            response = await self.client.get_latest_blockhash()
            blockhash = response['result']['value']['blockhash']
            
            # Build transaction
            tx = Transaction(recent_blockhash=blockhash)
            
            # Add compute budget instructions for priority
            tx.add(set_compute_unit_limit(100_000))
            tx.add(set_compute_unit_price(50_000))  # 0.00005 SOL per CU
            
            # Add memo instruction
            tx.add(memo_instruction)
            
            # Sign and send
            tx.sign(self.keypair)
            
            response = await self.client.send_transaction(
                tx,
                self.keypair,
                opts={"skip_preflight": False, "preflight_commitment": "confirmed"}
            )
            
            signature = response['result']
            print(f"[✓] Anchored to Solana: {signature}")
            
            # Wait for confirmation
            await self.wait_for_confirmation(signature)
            
            return signature
            
        except Exception as e:
            print(f"[✗] Failed to anchor: {e}")
            return None
    
    def create_memo_instruction(self, data: bytes) -> Instruction:
        """Create memo instruction for anchoring data"""
        # Memo program ID
        memo_program = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
        
        # Truncate data if too long (max ~900 bytes for memo)
        if len(data) > 900:
            # Hash it instead
            data_hash = sha256(data).digest()
            data = b"DNA_ANCHOR:" + base64.b64encode(data_hash)
        
        return Instruction(
            program_id=memo_program,
            accounts=[],
            data=data
        )
    
    async def wait_for_confirmation(self, signature: str, max_attempts: int = 30):
        """Wait for transaction confirmation"""
        print(f"[*] Waiting for confirmation...")
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.get_signature_statuses([signature])
                status = response['result']['value'][0]
                
                if status and status['confirmationStatus'] in ['confirmed', 'finalized']:
                    print(f"[✓] Transaction confirmed: {status['confirmationStatus']}")
                    return
                
            except Exception as e:
                print(f"[!] Error checking status: {e}")
            
            await asyncio.sleep(1)
        
        print("[!] Transaction not confirmed in time")
    
    async def update_clickhouse(self, batch: DecisionBatch, signature: str):
        """Update ClickHouse with anchor information"""
        print("[*] Updating ClickHouse records...")
        
        update_query = """
        ALTER TABLE decision_lineage
        UPDATE 
            merkle_root = %(merkle_root)s,
            anchor_tx = %(anchor_tx)s,
            anchored_at = now()
        WHERE decision_hash IN %(decision_hashes)s
        """
        
        self.ch_client.command(
            update_query,
            parameters={
                'merkle_root': batch.merkle_root,
                'anchor_tx': signature,
                'decision_hashes': batch.decision_hashes
            }
        )
        
        print(f"[✓] Updated {batch.decision_count} decisions")
    
    async def run_daily_anchor(self):
        """Run daily anchoring process"""
        print("\n" + "="*60)
        print("DECISION DNA DAILY ANCHOR")
        print("="*60)
        
        # Get yesterday's decisions
        end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=1)
        
        # Fetch decisions
        batch = await self.fetch_decisions(start_time, end_time)
        if not batch:
            print("[!] No decisions to anchor")
            return
        
        # Anchor to Solana
        signature = await self.anchor_to_solana(batch)
        if not signature:
            print("[✗] Anchoring failed")
            return
        
        # Update ClickHouse
        if not self.dry_run:
            await self.update_clickhouse(batch, signature)
        
        # Generate report
        report = self.generate_report(batch, signature)
        
        print("\n" + "="*60)
        print("ANCHOR COMPLETE")
        print("="*60)
        print(f"Decisions anchored: {batch.decision_count}")
        print(f"Merkle root: {batch.merkle_root[:32]}...")
        print(f"Solana signature: {signature}")
        print(f"Explorer: https://solscan.io/tx/{signature}")
    
    def generate_report(self, batch: DecisionBatch, signature: str) -> Dict[str, Any]:
        """Generate anchor report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch": asdict(batch),
            "anchor": {
                "signature": signature,
                "explorer_url": f"https://solscan.io/tx/{signature}",
                "merkle_root": batch.merkle_root,
            },
            "statistics": {
                "decisions_anchored": batch.decision_count,
                "total_value_usd": batch.total_value,
                "success_rate": batch.success_rate,
                "period": {
                    "start": batch.start_time.isoformat(),
                    "end": batch.end_time.isoformat()
                }
            }
        }
        
        # Save report
        report_path = f"/tmp/dna_anchor_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[✓] Report saved: {report_path}")
        return report
    
    async def verify_anchor(self, merkle_root: str, signature: str) -> bool:
        """Verify an anchor on-chain"""
        print(f"[*] Verifying anchor {signature}...")
        
        try:
            # Get transaction
            response = await self.client.get_transaction(
                signature,
                encoding="jsonParsed",
                max_supported_transaction_version=0
            )
            
            if not response['result']:
                print("[✗] Transaction not found")
                return False
            
            # Extract memo data
            tx = response['result']
            for instruction in tx['transaction']['message']['instructions']:
                if instruction.get('program') == 'spl-memo':
                    memo_data = instruction.get('parsed', '')
                    if merkle_root in memo_data:
                        print("[✓] Merkle root verified on-chain")
                        return True
            
            print("[✗] Merkle root not found in transaction")
            return False
            
        except Exception as e:
            print(f"[✗] Verification failed: {e}")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Decision DNA Daily Anchor")
    parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com", help="Solana RPC URL")
    parser.add_argument("--clickhouse", default="http://localhost:8123", help="ClickHouse URL")
    parser.add_argument("--keypair", default="~/.config/solana/id.json", help="Keypair path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--verify", help="Verify a previous anchor by signature")
    parser.add_argument("--date", help="Anchor specific date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Expand keypair path
    keypair_path = os.path.expanduser(args.keypair)
    
    anchor = DNAAnchor(
        rpc_url=args.rpc,
        clickhouse_url=args.clickhouse,
        keypair_path=keypair_path,
        dry_run=args.dry_run
    )
    
    await anchor.setup()
    
    if args.verify:
        # Verify mode
        # Would need merkle root from DB or args
        success = await anchor.verify_anchor("merkle_root_here", args.verify)
        sys.exit(0 if success else 1)
    
    elif args.date:
        # Anchor specific date
        date = datetime.strptime(args.date, "%Y-%m-%d")
        end_time = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        batch = await anchor.fetch_decisions(start_time, end_time)
        if batch:
            signature = await anchor.anchor_to_solana(batch)
            if signature and not args.dry_run:
                await anchor.update_clickhouse(batch, signature)
    
    else:
        # Default: anchor yesterday's decisions
        await anchor.run_daily_anchor()
    
    await anchor.client.close()


if __name__ == "__main__":
    asyncio.run(main())