/**
 * Ultra-High-Performance Virtual Scroller Component
 * Handles 100,000+ items with 60 FPS scrolling performance
 * Optimized for dynamic row heights and real-time updates
 */

import React, { useRef, useState, useEffect, useCallback, useMemo, memo } from 'react';
import { VariableSizeList as List, ListChildComponentProps } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { useLayoutEffect } from 'react';

interface VirtualScrollerProps<T> {
  items: T[];
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  estimatedItemHeight?: number;
  overscan?: number;
  onScroll?: (scrollTop: number, scrollHeight: number) => void;
  className?: string;
  maintainScrollPosition?: boolean;
  dynamicHeights?: boolean;
  getItemKey?: (item: T, index: number) => string;
}

// Row height cache for dynamic heights
class HeightCache {
  private cache: Map<number, number> = new Map();
  private defaultHeight: number;
  private version = 0;

  constructor(defaultHeight: number) {
    this.defaultHeight = defaultHeight;
  }

  get(index: number): number {
    return this.cache.get(index) || this.defaultHeight;
  }

  set(index: number, height: number): void {
    if (this.cache.get(index) !== height) {
      this.cache.set(index, height);
      this.version++;
    }
  }

  clear(): void {
    this.cache.clear();
    this.version++;
  }

  getVersion(): number {
    return this.version;
  }

  has(index: number): boolean {
    return this.cache.has(index);
  }
}

// Memoized row component
const Row = memo(<T,>({
  data,
  index,
  style
}: ListChildComponentProps<any>) => {
  const { items, renderItem, measureHeight } = data;
  const rowRef = useRef<HTMLDivElement>(null);
  
  useLayoutEffect(() => {
    if (measureHeight && rowRef.current) {
      const height = rowRef.current.getBoundingClientRect().height;
      measureHeight(index, rowRef.current);
    }
  }, [index, measureHeight]);

  const item = items[index];
  if (!item) return null;

  return (
    <div ref={rowRef} style={style}>
      {renderItem(item, index, style)}
    </div>
  );
});

Row.displayName = 'VirtualScrollerRow';

export function VirtualScroller<T>({
  items,
  renderItem,
  estimatedItemHeight = 50,
  overscan = 5,
  onScroll,
  className = '',
  maintainScrollPosition = false,
  dynamicHeights = false,
  getItemKey
}: VirtualScrollerProps<T>) {
  const listRef = useRef<List>(null);
  const outerRef = useRef<HTMLDivElement>(null);
  const heightCacheRef = useRef(new HeightCache(estimatedItemHeight));
  const [cacheVersion, setCacheVersion] = useState(0);
  const scrollPositionRef = useRef({ index: 0, offset: 0 });
  const isScrollingRef = useRef(false);
  const scrollTimeoutRef = useRef<number>();

  // Measure row height if dynamic heights enabled
  const measureHeight = useCallback((index: number, element: HTMLElement) => {
    if (!dynamicHeights) return;
    
    const height = element.getBoundingClientRect().height;
    const cache = heightCacheRef.current;
    
    if (cache.get(index) !== height) {
      cache.set(index, height);
      setCacheVersion(cache.getVersion());
      
      // Reset row heights in list
      if (listRef.current) {
        listRef.current.resetAfterIndex(index);
      }
    }
  }, [dynamicHeights]);

  // Get item size
  const getItemSize = useCallback((index: number) => {
    return heightCacheRef.current.get(index);
  }, [cacheVersion]); // Include cacheVersion to trigger re-render

  // Handle scroll events
  const handleScroll = useCallback(() => {
    const target = outerRef.current;
    if (!target) return;
    const scrollTop = target.scrollTop;
    const scrollHeight = target.scrollHeight;
    
    // Update scroll state
    isScrollingRef.current = true;
    clearTimeout(scrollTimeoutRef.current);
    scrollTimeoutRef.current = window.setTimeout(() => {
      isScrollingRef.current = false;
    }, 150);
    
    // Save scroll position
    if (listRef.current && maintainScrollPosition) {
      const visibleRange = (listRef.current as any)._getItemRangeToRender();
      if (visibleRange) {
        scrollPositionRef.current = {
          index: visibleRange[0],
          offset: scrollTop - (listRef.current as any)._getItemOffset(visibleRange[0])
        };
      }
    }
    
    if (onScroll) {
      onScroll(scrollTop, scrollHeight);
    }
  }, [maintainScrollPosition, onScroll]);

  // Restore scroll position when items change
  useEffect(() => {
    if (maintainScrollPosition && listRef.current && scrollPositionRef.current.index > 0) {
      listRef.current.scrollToItem(scrollPositionRef.current.index, 'start');
      
      // Fine-tune the scroll position
      requestAnimationFrame(() => {
        if (outerRef.current) {
          const itemOffset = (listRef.current as any)?._getItemOffset(scrollPositionRef.current.index) || 0;
          outerRef.current.scrollTop = itemOffset + scrollPositionRef.current.offset;
        }
      });
    }
  }, [items.length, maintainScrollPosition]);

  // Reset cache when items change significantly
  useEffect(() => {
    const cache = heightCacheRef.current;
    
    // Clear cache if item count changes dramatically
    if (Math.abs(items.length - cache['cache'].size) > 100) {
      cache.clear();
      setCacheVersion(cache.getVersion());
      
      if (listRef.current) {
        listRef.current.resetAfterIndex(0);
      }
    }
  }, [items.length]);

  // Memoize item data to prevent unnecessary re-renders
  const itemData = useMemo(() => ({
    items,
    renderItem,
    measureHeight: dynamicHeights ? measureHeight : undefined
  }), [items, renderItem, dynamicHeights, measureHeight]);

  // Generate item key
  const itemKey = useCallback((index: number, data: typeof itemData) => {
    if (getItemKey) {
      return getItemKey(data.items[index]!, index);
    }
    return `item-${index}`;
  }, [getItemKey]);

  // Optimize scrollbar rendering (CPU-only optimizations)
  useEffect(() => {
    if (outerRef.current) {
      // Use CSS containment for better performance
      outerRef.current.style.contain = 'layout style paint';
      
      // GPU acceleration disabled for CPU-only server
      // No transform or will-change properties
    }
  }, []);

  return (
    <div className={`virtual-scroller ${className}`} style={{ height: '100%', width: '100%' }}>
      <AutoSizer>
        {({ height, width }) => (
          <List<any>
            ref={listRef}
            outerRef={outerRef}
            height={height}
            width={width}
            itemCount={items.length}
            itemSize={getItemSize}
            itemData={itemData}
            itemKey={itemKey}
            overscanCount={overscan}
            onScroll={handleScroll as any}
            estimatedItemSize={estimatedItemHeight}
            style={{
              overflowX: 'hidden',
              overflowY: 'auto'
            }}
          >
            {Row}
          </List>
        )}
      </AutoSizer>
    </div>
  );
}

// Export optimized version for MEV opportunities list
export const MEVOpportunitiesVirtualScroller = memo(VirtualScroller) as typeof VirtualScroller;