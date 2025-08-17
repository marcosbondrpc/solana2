-- Economic Aggregates Query
-- Comprehensive economic impact metrics

WITH daily_extraction AS (
    SELECT 
        entity_id,
        toDate(timestamp) as date,
        -- Gross extraction
        SUM(profit_sol) as daily_gross_sol,
        COUNT(*) as daily_transactions,
        countIf(is_sandwich = 1) as daily_sandwiches,
        countIf(is_backrun = 1) as daily_backruns,
        countIf(is_atomic_arb = 1) as daily_atomic_arbs,
        
        -- Costs
        SUM(tip_lamports) / 1e9 as daily_tips_sol,
        SUM(priority_fee_lamports) / 1e9 as daily_fees_sol,
        
        -- Bundle metrics
        COUNT(DISTINCT bundle_hash) as daily_bundles,
        countIf(landed = 1) as daily_landed,
        
        -- Victim metrics
        COUNT(DISTINCT victim_address) as daily_unique_victims,
        SUM(victim_loss_sol) as daily_victim_losses,
        AVG(victim_loss_sol) as avg_victim_loss,
        MAX(victim_loss_sol) as max_victim_loss
        
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 90 DAY
    GROUP BY entity_id, date
),

rolling_metrics AS (
    SELECT 
        entity_id,
        date,
        daily_gross_sol,
        daily_tips_sol,
        daily_fees_sol,
        
        -- 7-day rolling windows
        SUM(daily_gross_sol) OVER (
            PARTITION BY entity_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as sol_7d,
        
        SUM(daily_tips_sol) OVER (
            PARTITION BY entity_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as tips_7d,
        
        -- 30-day rolling windows
        SUM(daily_gross_sol) OVER (
            PARTITION BY entity_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as sol_30d,
        
        SUM(daily_sandwiches) OVER (
            PARTITION BY entity_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as sandwiches_30d,
        
        SUM(daily_unique_victims) OVER (
            PARTITION BY entity_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as victims_30d,
        
        -- Success rate
        daily_landed / NULLIF(daily_transactions, 0) as daily_success_rate,
        
        -- Net profit
        daily_gross_sol - daily_tips_sol - daily_fees_sol as daily_net_sol
        
    FROM daily_extraction
),

entity_aggregates AS (
    SELECT 
        entity_id,
        
        -- Total metrics (90 days)
        SUM(daily_gross_sol) as total_gross_sol_90d,
        SUM(daily_net_sol) as total_net_sol_90d,
        SUM(daily_tips_sol) as total_tips_sol_90d,
        SUM(daily_fees_sol) as total_fees_sol_90d,
        
        -- Average daily metrics
        AVG(daily_gross_sol) as avg_daily_gross,
        AVG(daily_net_sol) as avg_daily_net,
        
        -- Efficiency metrics
        SUM(daily_gross_sol) / NULLIF(SUM(daily_tips_sol), 0) as tip_efficiency,
        SUM(daily_net_sol) / NULLIF(SUM(daily_gross_sol), 0) as net_margin,
        
        -- Activity metrics
        COUNT(DISTINCT date) as active_days,
        SUM(daily_transactions) as total_transactions,
        SUM(daily_sandwiches) as total_sandwiches,
        SUM(daily_landed) as total_landed,
        
        -- Success metrics
        AVG(daily_success_rate) as avg_success_rate,
        
        -- Victim impact
        SUM(daily_unique_victims) as total_unique_victims,
        SUM(daily_victim_losses) as total_victim_losses,
        AVG(avg_victim_loss) as overall_avg_victim_loss,
        
        -- Latest metrics
        argMax(sol_7d, date) as latest_sol_7d,
        argMax(sol_30d, date) as latest_sol_30d,
        MAX(date) as last_active_date
        
    FROM rolling_metrics
    GROUP BY entity_id
),

market_share AS (
    SELECT 
        entity_id,
        total_gross_sol_90d,
        total_gross_sol_90d / SUM(total_gross_sol_90d) OVER () * 100 as market_share_pct,
        ROW_NUMBER() OVER (ORDER BY total_gross_sol_90d DESC) as market_rank,
        SUM(total_gross_sol_90d) OVER (ORDER BY total_gross_sol_90d DESC) / 
            SUM(total_gross_sol_90d) OVER () * 100 as cumulative_market_share
    FROM entity_aggregates
)

SELECT 
    a.*,
    m.market_share_pct,
    m.market_rank,
    m.cumulative_market_share,
    
    -- ROI calculation
    (a.total_net_sol_90d / NULLIF(a.total_tips_sol_90d + a.total_fees_sol_90d, 0)) * 100 as roi_percentage,
    
    -- Profitability per sandwich
    a.total_gross_sol_90d / NULLIF(a.total_sandwiches, 0) as sol_per_sandwich,
    
    -- Monthly run rate (based on latest 30d)
    a.latest_sol_30d as monthly_run_rate,
    
    -- Annual run rate projection
    a.latest_sol_30d * 12 as annual_run_rate_projection,
    
    -- Classification
    CASE 
        WHEN m.market_rank <= 3 THEN 'TOP_3'
        WHEN m.market_rank <= 10 THEN 'TOP_10'
        WHEN m.market_share_pct >= 1 THEN 'MAJOR'
        WHEN m.market_share_pct >= 0.1 THEN 'SIGNIFICANT'
        ELSE 'MINOR'
    END as market_position,
    
    -- Performance tier
    CASE
        WHEN a.latest_sol_30d > 5000 THEN 'WHALE'
        WHEN a.latest_sol_30d > 1000 THEN 'SHARK'
        WHEN a.latest_sol_30d > 100 THEN 'DOLPHIN'
        WHEN a.latest_sol_30d > 10 THEN 'FISH'
        ELSE 'MINNOW'
    END as performance_tier
    
FROM entity_aggregates a
JOIN market_share m ON a.entity_id = m.entity_id
ORDER BY a.total_gross_sol_90d DESC;