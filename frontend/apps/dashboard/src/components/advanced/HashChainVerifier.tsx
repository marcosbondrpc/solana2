import { useEffect, useState, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';

const hashState = proxy({
  blocks: [] as Array<{
    height: number;
    hash: string;
    previousHash: string;
    timestamp: number;
    decisions: number;
    verified: boolean;
    merkleRoot: string;
  }>,
  
  verificationStatus: {
    isVerifying: false,
    lastVerified: 0,
    brokenAt: null as number | null,
    totalVerified: 0,
    totalBlocks: 0
  },
  
  currentBlock: {
    height: 0,
    hash: '',
    previousHash: '',
    timestamp: 0
  }
});

async function verifyHash(data: string, expectedHash: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const dataBuffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const computedHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return computedHash === expectedHash;
}

export default function HashChainVerifier() {
  const state = useSnapshot(hashState);
  const [autoVerify, setAutoVerify] = useState(true);
  
  // Fetch latest blocks
  useEffect(() => {
    const fetchBlocks = async () => {
      try {
        const response = await fetch('/api/hashchain/blocks?limit=100');
        const blocks = await response.json();
        hashState.blocks = blocks;
        hashState.verificationStatus.totalBlocks = blocks.length;
      } catch (error) {
        console.error('Failed to fetch blocks:', error);
      }
    };
    
    fetchBlocks();
    const interval = setInterval(fetchBlocks, 5000);
    return () => clearInterval(interval);
  }, []);
  
  // Auto-verify chain
  useEffect(() => {
    if (!autoVerify) return;
    
    const verifyChain = async () => {
      hashState.verificationStatus.isVerifying = true;
      let verified = 0;
      
      for (let i = 1; i < hashState.blocks.length; i++) {
        const block = hashState.blocks[i]!;
        const prevBlock = hashState.blocks[i - 1]!;
        
        if (block.previousHash !== prevBlock.hash) {
          hashState.verificationStatus.brokenAt = block.height;
          block.verified = false;
          break;
        }
        
        // Verify hash integrity
        const blockData = `${block.height}|${block.previousHash}|${block.merkleRoot}|${block.timestamp}`;
        const isValid = await verifyHash(blockData, block.hash);
        
        block.verified = isValid;
        if (!isValid) {
          hashState.verificationStatus.brokenAt = block.height;
          break;
        }
        
        verified++;
      }
      
      hashState.verificationStatus.totalVerified = verified;
      hashState.verificationStatus.lastVerified = Date.now();
      hashState.verificationStatus.isVerifying = false;
    };
    
    const interval = setInterval(verifyChain, 10000);
    verifyChain();
    
    return () => clearInterval(interval);
  }, [autoVerify]);
  
  const exportProof = useCallback(() => {
    const proof = {
      timestamp: Date.now(),
      blocks: hashState.blocks.map(b => ({
        height: b.height,
        hash: b.hash,
        previousHash: b.previousHash,
        merkleRoot: b.merkleRoot,
        timestamp: b.timestamp
      })),
      verification: hashState.verificationStatus
    };
    
    const blob = new Blob([JSON.stringify(proof, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hashchain_proof_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);
  
  return (
    <div style={{
      padding: '24px',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '30px', textShadow: '0 0 15px #00ff00' }}>
        Hash Chain Verifier
      </h1>
      
      {/* Status */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '20px',
        marginBottom: '30px'
      }}>
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Chain Status</h3>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 'bold',
            color: state.verificationStatus.brokenAt ? '#ff0000' : '#00ff00'
          }}>
            {state.verificationStatus.brokenAt ? 'BROKEN' : 'VALID'}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Verified Blocks</h3>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.verificationStatus.totalVerified} / {state.verificationStatus.totalBlocks}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Current Height</h3>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.blocks[0]?.height || 0}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Last Verified</h3>
          <div style={{ fontSize: '1rem' }}>
            {state.verificationStatus.lastVerified ? 
              new Date(state.verificationStatus.lastVerified).toLocaleTimeString() : 'Never'}
          </div>
        </div>
      </div>
      
      {/* Controls */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)',
        display: 'flex',
        gap: '20px',
        alignItems: 'center'
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input
            type="checkbox"
            checked={autoVerify}
            onChange={(e) => setAutoVerify(e.target.checked)}
          />
          Auto-Verify
        </label>
        
        <button
          onClick={exportProof}
          style={{
            padding: '10px 20px',
            background: '#00ff00',
            color: '#000',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          Export Proof
        </button>
        
        {state.verificationStatus.isVerifying && (
          <span style={{ color: '#ffff00' }}>Verifying...</span>
        )}
      </div>
      
      {/* Block Chain Visualization */}
      <div style={{
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Block Chain</h2>
        
        <div style={{
          display: 'flex',
          gap: '10px',
          overflowX: 'auto',
          padding: '20px 0'
        }}>
          {state.blocks.slice(0, 20).map((block, index) => (
            <div
              key={block.height}
              style={{
                minWidth: '150px',
                padding: '15px',
                border: `2px solid ${
                  !block.verified ? '#ff0000' :
                  block.height === state.verificationStatus.brokenAt ? '#ff0000' :
                  '#00ff00'
                }`,
                borderRadius: '8px',
                background: block.verified ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)',
                position: 'relative'
              }}
            >
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '10px' }}>
                Block #{block.height}
              </div>
              
              <div style={{ fontSize: '10px', marginBottom: '5px' }}>
                Hash: {block.hash.slice(0, 8)}...
              </div>
              
              <div style={{ fontSize: '10px', marginBottom: '5px' }}>
                Prev: {block.previousHash.slice(0, 8)}...
              </div>
              
              <div style={{ fontSize: '10px', marginBottom: '5px' }}>
                Decisions: {block.decisions}
              </div>
              
              <div style={{ fontSize: '10px' }}>
                {new Date(block.timestamp).toLocaleTimeString()}
              </div>
              
              {index < state.blocks.length - 1 && (
                <div style={{
                  position: 'absolute',
                  right: '-12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '24px',
                  height: '2px',
                  background: block.verified ? '#00ff00' : '#ff0000'
                }} />
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* Detailed Block List */}
      <div style={{
        marginTop: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Block Details</h2>
        
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #00ff00' }}>
              <th style={{ padding: '10px', textAlign: 'left' }}>Height</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Hash</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Merkle Root</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Decisions</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Timestamp</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {state.blocks.slice(0, 10).map(block => (
              <tr key={block.height} style={{ borderBottom: '1px solid #00ff0044' }}>
                <td style={{ padding: '10px' }}>{block.height}</td>
                <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '11px' }}>
                  {block.hash.slice(0, 16)}...
                </td>
                <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '11px' }}>
                  {block.merkleRoot.slice(0, 16)}...
                </td>
                <td style={{ padding: '10px' }}>{block.decisions}</td>
                <td style={{ padding: '10px', fontSize: '12px' }}>
                  {new Date(block.timestamp).toLocaleString()}
                </td>
                <td style={{ padding: '10px' }}>
                  <span style={{
                    color: block.verified ? '#00ff00' : '#ff0000',
                    fontWeight: 'bold'
                  }}>
                    {block.verified ? '✓ VALID' : '✗ INVALID'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}