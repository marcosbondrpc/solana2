-- MEV Sandwich Detector - Bundle Tracking and State Management
-- Atomic bundle state transitions with optimistic locking

local bundle_id = KEYS[1]
local bundles_set = KEYS[2]
local metrics_key = KEYS[3]

local action = ARGV[1]  -- 'create', 'submit', 'land', 'expire'
local timestamp = tonumber(ARGV[2])
local metadata = ARGV[3]  -- JSON metadata

-- Helper function to parse JSON (simple implementation)
local function parse_json_simple(str)
    local result = {}
    for k, v in string.gmatch(str, '"([^"]+)"%s*:%s*([^,}]+)') do
        -- Remove quotes if string value
        v = string.gsub(v, '^"', '')
        v = string.gsub(v, '"$', '')
        -- Try to convert to number
        local num = tonumber(v)
        result[k] = num or v
    end
    return result
end

-- State machine for bundle lifecycle
local valid_transitions = {
    ['create'] = {from = nil, to = 'pending'},
    ['submit'] = {from = 'pending', to = 'submitted'},
    ['land'] = {from = 'submitted', to = 'landed'},
    ['expire'] = {from = {'pending', 'submitted'}, to = 'expired'}
}

-- Get current bundle state
local bundle_key = 'bundle:' .. bundle_id
local current_state = redis.call('HGET', bundle_key, 'status')

-- Validate transition
local transition = valid_transitions[action]
if not transition then
    return {0, 'invalid_action', action}
end

-- Check state transition validity
if action ~= 'create' then
    local valid_from = false
    if type(transition.from) == 'table' then
        for _, state in ipairs(transition.from) do
            if current_state == state then
                valid_from = true
                break
            end
        end
    else
        valid_from = (current_state == transition.from)
    end
    
    if not valid_from then
        return {0, 'invalid_transition', current_state or 'nil'}
    end
end

-- Parse metadata
local meta = parse_json_simple(metadata or '{}')

-- Execute action
if action == 'create' then
    -- Create new bundle
    redis.call('HSET', bundle_key,
        'id', bundle_id,
        'status', 'pending',
        'created_at', timestamp,
        'expected_profit', meta.expected_profit or 0,
        'gas_cost', meta.gas_cost or 0,
        'confidence', meta.confidence or 0,
        'tip', meta.tip or 0,
        'priority', meta.priority or 0
    )
    
    -- Add to active bundles set
    redis.call('ZADD', bundles_set, timestamp, bundle_id)
    
    -- Increment metrics
    redis.call('HINCRBY', metrics_key, 'bundles_created', 1)
    
elseif action == 'submit' then
    -- Mark bundle as submitted
    redis.call('HMSET', bundle_key,
        'status', 'submitted',
        'submitted_at', timestamp,
        'submission_path', meta.path or 'both',
        'final_tip', meta.tip or redis.call('HGET', bundle_key, 'tip')
    )
    
    -- Track submission paths
    local path = meta.path or 'both'
    redis.call('HINCRBY', metrics_key, 'submissions_' .. path, 1)
    
elseif action == 'land' then
    -- Bundle landed on-chain
    local expected_profit = tonumber(redis.call('HGET', bundle_key, 'expected_profit') or '0')
    local actual_profit = tonumber(meta.actual_profit or '0')
    local profit_variance = actual_profit - expected_profit
    
    redis.call('HMSET', bundle_key,
        'status', 'landed',
        'landed_at', timestamp,
        'actual_profit', actual_profit,
        'profit_variance', profit_variance,
        'landing_slot', meta.slot or 0,
        'block_hash', meta.block_hash or ''
    )
    
    -- Update metrics
    redis.call('HINCRBY', metrics_key, 'bundles_landed', 1)
    redis.call('HINCRBYFLOAT', metrics_key, 'total_profit', actual_profit)
    
    -- Calculate and update success rate
    local submitted = tonumber(redis.call('HGET', metrics_key, 'submissions_both') or '0')
    local landed = tonumber(redis.call('HGET', metrics_key, 'bundles_landed') or '0')
    if submitted > 0 then
        local success_rate = landed / submitted
        redis.call('HSET', metrics_key, 'success_rate', success_rate)
    end
    
    -- Remove from active bundles
    redis.call('ZREM', bundles_set, bundle_id)
    
elseif action == 'expire' then
    -- Bundle expired without landing
    redis.call('HMSET', bundle_key,
        'status', 'expired',
        'expired_at', timestamp,
        'reason', meta.reason or 'timeout'
    )
    
    -- Update metrics
    redis.call('HINCRBY', metrics_key, 'bundles_expired', 1)
    
    -- Remove from active bundles
    redis.call('ZREM', bundles_set, bundle_id)
end

-- Set TTL on bundle key (7 days)
redis.call('EXPIRE', bundle_key, 604800)

-- Clean old bundles from active set (older than 5 minutes)
local cutoff = timestamp - 300
redis.call('ZREMRANGEBYSCORE', bundles_set, 0, cutoff)

-- Return success with new state
return {1, transition.to, bundle_id}