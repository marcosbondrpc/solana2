#!/usr/bin/env python3
"""
MEV Detection Inference Service API
100% DETECTION-ONLY - No execution capabilities
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import os
import asyncio
from datetime import datetime, timedelta

# Import our inference modules
from landing_ratio import LandingRatioAnalyzer
from latency_skew import LatencySkewAnalyzer
from fleet_cluster import FleetClusterAnalyzer
from ordering_quirks import OrderingQuirksAnalyzer
from economic_impact import EconomicImpactAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="MEV Detection Inference Service",
    description="Statistical analysis and economic impact measurement for MEV detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")

landing_analyzer = LandingRatioAnalyzer(CLICKHOUSE_URL)
latency_analyzer = LatencySkewAnalyzer(CLICKHOUSE_URL)
fleet_analyzer = FleetClusterAnalyzer(CLICKHOUSE_URL)
ordering_analyzer = OrderingQuirksAnalyzer(CLICKHOUSE_URL)
economic_analyzer = EconomicImpactAnalyzer(CLICKHOUSE_URL)


@app.get("/")
async def root():
    """Health check and service info"""
    return {
        "service": "MEV Detection Inference",
        "status": "healthy",
        "mode": "DETECTION_ONLY",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "landing_ratio": "/api/landing-ratio",
            "latency_analysis": "/api/latency-skew",
            "fleet_detection": "/api/fleet-cluster",
            "ordering_quirks": "/api/ordering-quirks",
            "economic_impact": "/api/economic-impact"
        }
    }


@app.get("/api/landing-ratio")
async def analyze_landing_ratio(
    lookback_hours: int = Query(24, description="Hours to look back"),
    target_entity: Optional[str] = Query(None, description="Specific entity to analyze")
):
    """
    Analyze bundle landing rates with statistical hypothesis testing
    """
    try:
        results = landing_analyzer.detect_anomalies(
            lookback_hours=lookback_hours,
            target_entity=target_entity
        )
        
        # Convert to dict for JSON response
        response = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": lookback_hours,
            "total_entities": len(results),
            "anomalous_entities": sum(1 for r in results if r.is_anomalous),
            "results": [
                {
                    "entity_id": r.entity_id,
                    "landing_rate": r.landing_rate,
                    "peer_mean_rate": r.peer_mean_rate,
                    "t_statistic": r.t_statistic,
                    "p_value": r.p_value,
                    "adjusted_p_value": r.adjusted_p_value,
                    "is_anomalous": r.is_anomalous,
                    "effect_size": r.effect_size,
                    "confidence_interval": r.confidence_interval,
                    "decision_dna": r.decision_dna
                }
                for r in results[:100]  # Limit response size
            ]
        }
        
        # Export to ClickHouse
        landing_analyzer.export_to_clickhouse(results)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/latency-skew")
async def analyze_latency_skew(
    lookback_hours: int = Query(24, description="Hours to look back")
):
    """
    Detect ultra-optimized operators through latency distribution analysis
    """
    try:
        results = latency_analyzer.detect_ultra_optimized(
            lookback_hours=lookback_hours
        )
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": lookback_hours,
            "total_entities": len(results),
            "ultra_optimized": sum(1 for r in results if r.is_ultra_optimized),
            "results": [
                {
                    "entity_id": r.entity_id,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "p95_p50_spread": r.p95_p50_spread,
                    "levene_p_value": r.levene_p_value,
                    "is_ultra_optimized": r.is_ultra_optimized,
                    "decision_dna": r.decision_dna,
                    "optimization_techniques": latency_analyzer.analyze_optimization_techniques(r)
                }
                for r in results[:50]
            ]
        }
        
        # Export to ClickHouse
        latency_analyzer.export_to_clickhouse(results)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fleet-cluster")
async def analyze_fleet_clusters(
    lookback_hours: int = Query(168, description="Hours to look back (default 7 days)")
):
    """
    Detect coordinated wallet fleets using clustering algorithms
    """
    try:
        profiles = fleet_analyzer.analyze_entity_fleets(
            lookback_hours=lookback_hours
        )
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": lookback_hours,
            "total_fleets": len(profiles),
            "total_clusters": sum(len(p.clusters) for p in profiles),
            "fleets": [
                {
                    "fleet_id": p.fleet_id,
                    "entity": p.primary_entity,
                    "wallet_count": p.wallet_count,
                    "active_wallets": p.active_wallets,
                    "coordination_score": p.coordination_score,
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "size": c.cluster_size,
                            "similarity": c.behavioral_similarity,
                            "rotation_pattern": c.rotation_pattern,
                            "confidence": c.cluster_confidence,
                            "decision_dna": c.decision_dna
                        }
                        for c in p.clusters
                    ]
                }
                for p in profiles[:20]
            ]
        }
        
        # Export to ClickHouse
        fleet_analyzer.export_to_clickhouse(profiles)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ordering-quirks")
async def analyze_ordering_quirks(
    lookback_hours: int = Query(72, description="Hours to look back")
):
    """
    Detect statistically significant transaction ordering patterns
    """
    try:
        profiles = ordering_analyzer.analyze_entity_quirks(
            lookback_hours=lookback_hours
        )
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": lookback_hours,
            "total_entities": len(profiles),
            "entities_with_quirks": sum(1 for p in profiles if p.quirk_score > 0),
            "profiles": [
                {
                    "entity_id": p.entity_id,
                    "quirk_score": p.quirk_score,
                    "total_transactions": p.total_transactions,
                    "sandwich_patterns": len(p.sandwich_patterns),
                    "backrun_patterns": len(p.backrun_patterns),
                    "atomic_patterns": len(p.atomic_patterns),
                    "most_targeted_pools": p.most_targeted_pools[:5],
                    "top_patterns": [
                        {
                            "type": pat.pattern_type,
                            "pool": pat.pool_address[:16] + "...",
                            "p_value": pat.bootstrap_p_value,
                            "effect_size": pat.effect_size,
                            "decision_dna": pat.decision_dna
                        }
                        for pat in (p.sandwich_patterns + p.backrun_patterns + p.atomic_patterns)[:3]
                    ]
                }
                for p in profiles[:20]
            ]
        }
        
        # Export to ClickHouse
        ordering_analyzer.export_to_clickhouse(profiles)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/economic-impact")
async def analyze_economic_impact(
    lookback_days: int = Query(90, description="Days to look back")
):
    """
    Measure economic impact of MEV activities
    """
    try:
        profiles = economic_analyzer.build_economic_profiles(
            lookback_days=lookback_days
        )
        
        # Calculate totals
        total_extraction = sum(p.gross_profit_sol for p in profiles)
        total_victims = sum(p.total_victims for p in profiles)
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "market_overview": {
                "total_extraction_sol": total_extraction,
                "total_unique_victims": total_victims,
                "active_entities": len(profiles),
                "reference": {
                    "b91_monthly": 7800,
                    "note": "B91 extracts ~7,800 SOL/month"
                }
            },
            "top_extractors": [
                {
                    "entity_id": p.entity_id,
                    "sol_extracted_90d": p.sol_extracted_90d,
                    "sol_extracted_30d": p.sol_extracted_30d,
                    "sol_extracted_7d": p.sol_extracted_7d,
                    "net_profit_sol": p.net_profit_sol,
                    "roi_percentage": p.roi_percentage,
                    "market_share": p.market_share,
                    "total_victims": p.total_victims,
                    "avg_victim_loss": p.avg_victim_loss_sol,
                    "profit_per_sandwich": p.profit_per_sandwich,
                    "success_rate": p.success_rate,
                    "decision_dna": p.decision_dna
                }
                for p in profiles[:10]
            ]
        }
        
        # Export to ClickHouse
        economic_analyzer.export_to_clickhouse(profiles)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entity/{entity_id}")
async def get_entity_profile(entity_id: str):
    """
    Get comprehensive profile for a specific entity
    """
    try:
        # Run all analyses for this entity
        landing_results = landing_analyzer.detect_anomalies(
            lookback_hours=24,
            target_entity=entity_id
        )
        
        latency_results = latency_analyzer.detect_ultra_optimized(
            lookback_hours=24
        )
        
        economic_profiles = economic_analyzer.build_economic_profiles(
            lookback_days=30
        )
        
        # Find entity-specific results
        entity_landing = next((r for r in landing_results if r.entity_id == entity_id), None)
        entity_latency = next((r for r in latency_results if r.entity_id == entity_id), None)
        entity_economic = next((p for p in economic_profiles if p.entity_id == entity_id), None)
        
        response = {
            "entity_id": entity_id,
            "timestamp": datetime.now().isoformat(),
            "landing_metrics": {
                "landing_rate": entity_landing.landing_rate if entity_landing else None,
                "is_anomalous": entity_landing.is_anomalous if entity_landing else None,
                "p_value": entity_landing.p_value if entity_landing else None
            } if entity_landing else None,
            "latency_metrics": {
                "p50_ms": entity_latency.p50_latency_ms if entity_latency else None,
                "p99_ms": entity_latency.p99_latency_ms if entity_latency else None,
                "is_ultra_optimized": entity_latency.is_ultra_optimized if entity_latency else None
            } if entity_latency else None,
            "economic_metrics": {
                "sol_extracted_30d": entity_economic.sol_extracted_30d if entity_economic else None,
                "market_share": entity_economic.market_share if entity_economic else None,
                "roi_percentage": entity_economic.roi_percentage if entity_economic else None,
                "total_victims": entity_economic.total_victims if entity_economic else None
            } if entity_economic else None
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-all-analyses")
async def run_all_analyses():
    """
    Run all analyses and generate comprehensive reports
    """
    try:
        # Run all analyses in parallel
        results = await asyncio.gather(
            asyncio.to_thread(landing_analyzer.detect_anomalies, 24),
            asyncio.to_thread(latency_analyzer.detect_ultra_optimized, 24),
            asyncio.to_thread(fleet_analyzer.analyze_entity_fleets, 168),
            asyncio.to_thread(ordering_analyzer.analyze_entity_quirks, 72),
            asyncio.to_thread(economic_analyzer.build_economic_profiles, 90)
        )
        
        landing_results, latency_results, fleet_profiles, quirk_profiles, economic_profiles = results
        
        # Generate reports
        reports = {
            "landing_rate": landing_analyzer.generate_report(landing_results),
            "latency": latency_analyzer.generate_report(latency_results),
            "fleets": fleet_analyzer.generate_report(fleet_profiles),
            "quirks": ordering_analyzer.generate_report(quirk_profiles),
            "economic": economic_analyzer.generate_report(economic_profiles)
        }
        
        # Save reports to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, report in reports.items():
            with open(f"/app/reports/{name}_analysis_{timestamp}.md", "w") as f:
                f.write(report)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "reports_generated": list(reports.keys()),
            "summary": {
                "anomalous_landing_rates": sum(1 for r in landing_results if r.is_anomalous),
                "ultra_optimized_operators": sum(1 for r in latency_results if r.is_ultra_optimized),
                "coordinated_fleets": len(fleet_profiles),
                "entities_with_quirks": sum(1 for p in quirk_profiles if p.quirk_score > 0),
                "total_sol_extracted_90d": sum(p.gross_profit_sol for p in economic_profiles)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)