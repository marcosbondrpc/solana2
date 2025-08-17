use solana_sdk::{signature::Signature, transaction::VersionedTransaction};

#[inline]
pub fn first_sig(tx: &VersionedTransaction) -> Option<Signature> {
    tx.signatures.first().cloned()
}


