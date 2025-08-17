import * as Comlink from 'comlink';

interface ComputeWorkerAPI {
  processDetectionBatch(events: any[]): Promise<any[]>;
  calculateBehavioralEmbedding(entities: any[]): Promise<any[]>;
  computeModelMetrics(data: any): Promise<any>;
  generateMerkleProof(nodes: any[]): Promise<string>;
}

class ComputeWorker implements ComputeWorkerAPI {
  async processDetectionBatch(events: any[]): Promise<any[]> {
    // Heavy computation for processing detection events
    return events.map(event => ({
      ...event,
      processed: true,
      risk_score: this.calculateRiskScore(event),
      pattern_match: this.detectPattern(event),
      cluster_id: this.assignCluster(event),
    }));
  }

  async calculateBehavioralEmbedding(entities: any[]): Promise<any[]> {
    // t-SNE/UMAP-like projection simulation
    return entities.map(entity => {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.sqrt(entity.riskAppetite) * 10;
      return {
        ...entity,
        embedding: {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: entity.successRate / 100 * 5,
        },
      };
    });
  }

  async computeModelMetrics(data: any): Promise<any> {
    const { predictions, actuals } = data;
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === 1 && actuals[i] === 1) tp++;
      else if (predictions[i] === 1 && actuals[i] === 0) fp++;
      else if (predictions[i] === 0 && actuals[i] === 0) tn++;
      else if (predictions[i] === 0 && actuals[i] === 1) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1,
      confusion_matrix: { tp, fp, tn, fn },
    };
  }

  async generateMerkleProof(nodes: any[]): Promise<string> {
    // Simplified Merkle tree generation
    if (nodes.length === 0) return '';
    
    let hashes = nodes.map(n => this.hash(JSON.stringify(n)));
    
    while (hashes.length > 1) {
      const newHashes = [];
      for (let i = 0; i < hashes.length; i += 2) {
        const left = hashes[i];
        const right = hashes[i + 1] || hashes[i];
        newHashes.push(this.hash(left + right));
      }
      hashes = newHashes;
    }
    
    return hashes[0];
  }

  private calculateRiskScore(event: any): number {
    const severityWeight = {
      LOW: 0.25,
      MEDIUM: 0.5,
      HIGH: 0.75,
      CRITICAL: 1.0,
    };
    
    const baseScore = severityWeight[event.severity as keyof typeof severityWeight] || 0.5;
    const profitFactor = Math.min(event.metrics.profitEstimate / 10000, 1);
    const confidenceFactor = event.metrics.confidence;
    
    return baseScore * 0.4 + profitFactor * 0.3 + confidenceFactor * 0.3;
  }

  private detectPattern(event: any): string {
    const patterns = ['SANDWICH', 'FRONTRUN', 'ARBITRAGE', 'LIQUIDATION'];
    const typeIndex = patterns.indexOf(event.type);
    
    if (typeIndex >= 0 && Math.random() > 0.3) {
      return patterns[typeIndex];
    }
    
    return 'UNKNOWN';
  }

  private assignCluster(event: any): number {
    // Simple clustering based on venue and type
    const venueHash = this.simpleHash(event.venue);
    const typeHash = this.simpleHash(event.type);
    return (venueHash + typeHash) % 10;
  }

  private hash(str: string): string {
    // Simple hash function for demonstration
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }
}

Comlink.expose(new ComputeWorker());