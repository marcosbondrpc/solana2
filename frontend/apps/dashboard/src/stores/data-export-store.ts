import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ExportConfig, ExportProgress } from '../components/node/DatasetExporter';

interface ExportTemplate {
  id: string;
  name: string;
  description: string;
  config: ExportConfig;
  createdAt: number;
}

interface ExportHistoryItem {
  id: string;
  timestamp: number;
  status: 'complete' | 'error' | 'cancelled';
  config: ExportConfig;
  fileSize: number;
  rowCount: number;
  downloadUrl?: string;
  error?: string;
}

interface DataExportStore {
  // Current export state
  currentExport: ExportProgress | null;
  exportConfig: ExportConfig | null;
  
  // Templates
  templates: ExportTemplate[];
  
  // History
  history: ExportHistoryItem[];
  
  // UI state
  isExportModalOpen: boolean;
  activeTab: string;
  
  // Feature analysis cache
  featureAnalysis: {
    importance: Array<{
      feature: string;
      importance: number;
      category: string;
    }> | null;
    correlation: {
      features: string[];
      matrix: number[][];
    } | null;
    lastUpdated: number;
  };
  
  // Actions
  setCurrentExport: (progress: ExportProgress | null) => void;
  setExportConfig: (config: ExportConfig) => void;
  
  // Template actions
  addTemplate: (template: Omit<ExportTemplate, 'id' | 'createdAt'>) => void;
  removeTemplate: (id: string) => void;
  loadTemplate: (id: string) => ExportConfig | null;
  
  // History actions
  addToHistory: (item: Omit<ExportHistoryItem, 'id' | 'timestamp'>) => void;
  clearHistory: () => void;
  
  // UI actions
  openExportModal: () => void;
  closeExportModal: () => void;
  setActiveTab: (tab: string) => void;
  
  // Feature analysis actions
  setFeatureAnalysis: (analysis: {
    importance: Array<{
      feature: string;
      importance: number;
      category: string;
    }>;
    correlation: {
      features: string[];
      matrix: number[][];
    };
  }) => void;
  clearFeatureAnalysis: () => void;
  
  // Utility actions
  reset: () => void;
}

const initialState = {
  currentExport: null,
  exportConfig: null,
  templates: [],
  history: [],
  isExportModalOpen: false,
  activeTab: 'metrics',
  featureAnalysis: {
    importance: null,
    correlation: null,
    lastUpdated: 0,
  },
};

export const useDataExportStore = create<DataExportStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,
        
        // Current export actions
        setCurrentExport: (progress) => set({ currentExport: progress }),
        setExportConfig: (config) => set({ exportConfig: config }),
        
        // Template actions
        addTemplate: (template) => {
          const newTemplate: ExportTemplate = {
            ...template,
            id: `template-${Date.now()}`,
            createdAt: Date.now(),
          };
          set((state) => ({
            templates: [...state.templates, newTemplate],
          }));
        },
        
        removeTemplate: (id) => {
          set((state) => ({
            templates: state.templates.filter((t) => t.id !== id),
          }));
        },
        
        loadTemplate: (id) => {
          const template = get().templates.find((t) => t.id === id);
          if (template) {
            set({ exportConfig: template.config });
            return template.config;
          }
          return null;
        },
        
        // History actions
        addToHistory: (item) => {
          const historyItem: ExportHistoryItem = {
            ...item,
            id: `export-${Date.now()}`,
            timestamp: Date.now(),
          };
          set((state) => ({
            history: [historyItem, ...state.history].slice(0, 50), // Keep last 50 items
          }));
        },
        
        clearHistory: () => set({ history: [] }),
        
        // UI actions
        openExportModal: () => set({ isExportModalOpen: true }),
        closeExportModal: () => set({ isExportModalOpen: false }),
        setActiveTab: (tab) => set({ activeTab: tab }),
        
        // Feature analysis actions
        setFeatureAnalysis: (analysis) => {
          set({
            featureAnalysis: {
              importance: analysis.importance,
              correlation: analysis.correlation,
              lastUpdated: Date.now(),
            },
          });
        },
        
        clearFeatureAnalysis: () => {
          set({
            featureAnalysis: {
              importance: null,
              correlation: null,
              lastUpdated: 0,
            },
          });
        },
        
        // Utility actions
        reset: () => set(initialState),
      }),
      {
        name: 'data-export-store',
        partialize: (state) => ({
          templates: state.templates,
          history: state.history.slice(0, 10), // Only persist last 10 history items
        }),
      }
    ),
    {
      name: 'DataExportStore',
    }
  )
);

// Selectors
export const selectCurrentExport = (state: DataExportStore) => state.currentExport;
export const selectExportConfig = (state: DataExportStore) => state.exportConfig;
export const selectTemplates = (state: DataExportStore) => state.templates;
export const selectHistory = (state: DataExportStore) => state.history;
export const selectIsExportModalOpen = (state: DataExportStore) => state.isExportModalOpen;
export const selectFeatureAnalysis = (state: DataExportStore) => state.featureAnalysis;