-- MEV Sandwich Detector - Tip Escalation with CAS
-- Atomic tip escalation logic for competitive bundles

local bundle_key = KEYS[1]
local competitor_tips_key = KEYS[2]
local network_load_key = KEYS[3]

local expected_profit = tonumber(ARGV[1])
local current_tip = tonumber(ARGV[2])
local confidence = tonumber(ARGV[3])
local max_tip_ratio = tonumber(ARGV[4]) or 0.5  -- Max 50% of profit

-- Get current bundle state
local bundle_state = redis.call('HGETALL', bundle_key)
local bundle = {}
for i = 1, #bundle_state, 2 do
    bundle[bundle_state[i]] = bundle_state[i + 1]
end

-- Check if bundle already landed or expired
if bundle.status == 'landed' or bundle.status == 'expired' then
    return {0, current_tip, 'bundle_inactive'}
end

-- Get competitor activity
local competitor_tips = redis.call('ZRANGE', competitor_tips_key, -10, -1, 'WITHSCORES')
local avg_competitor_tip = 0
local max_competitor_tip = 0

if #competitor_tips > 0 then
    local sum = 0
    for i = 2, #competitor_tips, 2 do
        local tip = tonumber(competitor_tips[i])
        sum = sum + tip
        if tip > max_competitor_tip then
            max_competitor_tip = tip
        end
    end
    avg_competitor_tip = sum / (#competitor_tips / 2)
end

-- Get network load factor
local network_load = tonumber(redis.call('GET', network_load_key) or '0.5')

-- Calculate escalation factors
local competition_factor = 1.0
if max_competitor_tip > current_tip then
    -- Need to outbid
    competition_factor = 1.2 + (max_competitor_tip - current_tip) / current_tip * 0.5
end

local network_factor = 1.0 + network_load * 0.5  -- Up to 1.5x at full load
local confidence_factor = 0.7 + confidence * 0.3  -- 0.7x to 1.0x based on confidence

-- Calculate new tip with escalation
local base_escalation = current_tip * 1.1  -- 10% base escalation
local competitive_tip = math.max(max_competitor_tip * 1.05, base_escalation)
local new_tip = competitive_tip * network_factor * confidence_factor

-- Apply bounds
local min_tip = expected_profit * 0.01  -- Min 1%
local max_tip = expected_profit * max_tip_ratio
new_tip = math.max(min_tip, math.min(max_tip, new_tip))

-- CAS update - only if our tip is higher
local stored_tip = tonumber(bundle.tip or '0')
if new_tip <= stored_tip then
    return {0, stored_tip, 'tip_not_increased'}
end

-- Atomic update
redis.call('HSET', bundle_key, 
    'tip', new_tip,
    'escalation_count', (tonumber(bundle.escalation_count or '0') + 1),
    'last_escalation', redis.call('TIME')[1],
    'competition_factor', competition_factor,
    'network_factor', network_factor
)

-- Record escalation event
redis.call('ZADD', 'tip_escalations', redis.call('TIME')[1], 
    string.format("%s:%f:%f", bundle_key, current_tip, new_tip))

-- Update competitor tracking
redis.call('ZADD', competitor_tips_key, new_tip, bundle_key)
redis.call('EXPIRE', competitor_tips_key, 60)  -- 60 second window

-- Calculate escalation percentage
local escalation_pct = (new_tip - current_tip) / current_tip * 100

return {1, new_tip, string.format("escalated_%.1f_pct", escalation_pct)}