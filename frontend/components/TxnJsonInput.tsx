"use client";
import React, { useState, useRef, useEffect, useContext, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Filter, Check, AlertCircle, ArrowRight, Send, Sparkles, Bot, 
  ChevronDown, ChevronUp, Eye, X, RefreshCw, Sliders, AlertTriangle,
  Users, TrendingUp, Clock, MapPin, DollarSign, Zap, Plus, Star, Trash2, Save
} from 'lucide-react';
import { getCsvData, searchCsvData } from '../lib/api';
import { AppContext } from '@/app/page';
import { 
  SavedSmartFilter, 
  getGlobalSmartFilters, 
  saveGlobalSmartFilter, 
  updateSmartFilterUsage,
  deleteGlobalSmartFilter 
} from '@/lib/storage';

interface CaseFilter {
  id: string;
  column: string;
  operator: 'equals' | 'not_equals' | 'greater' | 'less' | 'contains';
  value: string;
  label: string;
}

interface SelectedCase {
  index: number;
  data: Record<string, any>;
}

// Icon mapping for saved filters
const ICON_MAP: Record<string, any> = {
  AlertTriangle,
  Users,
  X,
  AlertCircle,
  DollarSign,
  Clock,
  MapPin,
  Zap,
  TrendingUp,
  Star,
  Filter,
  Check,
};

// Smart filter template definitions - these will be matched against actual columns
interface SmartFilterTemplate {
  id: string;
  label: string;
  description: string;
  icon: string;
  color: string;
  // Column patterns to match (will check if any of these exist)
  columnPatterns: string[];
  // How to build the filter once a column is matched
  buildFilter: (matchedColumn: string, sampleValues: string[]) => { column: string; operator: string; value: string; label: string }[] | null;
}

const SMART_FILTER_TEMPLATES: SmartFilterTemplate[] = [
  {
    id: 'override_cases',
    label: 'Override Cases',
    description: 'Cases where analyst overrode the system',
    icon: 'AlertTriangle',
    color: 'text-amber-400',
    columnPatterns: ['override', 'overrode', 'override_flag'],
    buildFilter: (col, values) => {
      // Check if column has 1/0, true/false, yes/no values
      const hasTrue = values.some(v => ['1', 'true', 'yes', 'True', 'Yes', 'TRUE', 'YES'].includes(String(v)));
      if (hasTrue) {
        const trueVal = values.find(v => ['1', 'true', 'yes', 'True', 'Yes', 'TRUE', 'YES'].includes(String(v))) || '1';
        return [{ column: col, operator: 'equals', value: String(trueVal), label: `${col} = ${trueVal}` }];
      }
      return null;
    }
  },
  {
    id: 'fraud_flagged',
    label: 'Flagged as Fraud',
    description: 'Cases marked as fraud by analysts',
    icon: 'AlertCircle',
    color: 'text-red-400',
    columnPatterns: ['decision', 'fraud', 'flag', 'result', 'outcome'],
    buildFilter: (col, values) => {
      const fraudVal = values.find(v => 
        ['fraud', 'Fraud', 'FRAUD', '1', 'true', 'yes', 'positive', 'confirmed'].includes(String(v).toLowerCase())
      );
      if (fraudVal) {
        return [{ column: col, operator: 'equals', value: String(fraudVal), label: `${col} = ${fraudVal}` }];
      }
      return null;
    }
  },
  {
    id: 'legit_cases',
    label: 'Cleared/Legitimate',
    description: 'Cases cleared as legitimate',
    icon: 'Check',
    color: 'text-green-400',
    columnPatterns: ['decision', 'fraud', 'flag', 'result', 'outcome'],
    buildFilter: (col, values) => {
      const legitVal = values.find(v => 
        ['legit', 'Legit', 'legitimate', 'clear', 'cleared', '0', 'false', 'no', 'negative'].includes(String(v).toLowerCase())
      );
      if (legitVal) {
        return [{ column: col, operator: 'equals', value: String(legitVal), label: `${col} = ${legitVal}` }];
      }
      return null;
    }
  },
  {
    id: 'high_value',
    label: 'High Value',
    description: 'High value transactions',
    icon: 'DollarSign',
    color: 'text-emerald-400',
    columnPatterns: ['amount', 'value', 'total', 'sum', 'price'],
    buildFilter: (col, values) => {
      // Calculate a reasonable threshold (e.g., 75th percentile)
      const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const p75 = numericValues[Math.floor(numericValues.length * 0.75)];
        const threshold = Math.round(p75);
        return [{ column: col, operator: 'greater', value: String(threshold), label: `${col} > ${threshold}` }];
      }
      return null;
    }
  },
  {
    id: 'low_value',
    label: 'Low Value',
    description: 'Low value transactions',
    icon: 'DollarSign',
    color: 'text-slate-400',
    columnPatterns: ['amount', 'value', 'total', 'sum', 'price'],
    buildFilter: (col, values) => {
      const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const p25 = numericValues[Math.floor(numericValues.length * 0.25)];
        const threshold = Math.round(p25);
        return [{ column: col, operator: 'less', value: String(threshold), label: `${col} < ${threshold}` }];
      }
      return null;
    }
  },
  {
    id: 'quick_handling',
    label: 'Quick Handling',
    description: 'Cases handled quickly',
    icon: 'Clock',
    color: 'text-blue-400',
    columnPatterns: ['time', 'duration', 'seconds', 'minutes', 'handling'],
    buildFilter: (col, values) => {
      const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const p25 = numericValues[Math.floor(numericValues.length * 0.25)];
        const threshold = Math.round(p25);
        return [{ column: col, operator: 'less', value: String(threshold), label: `${col} < ${threshold}` }];
      }
      return null;
    }
  },
  {
    id: 'slow_handling',
    label: 'Slow Handling',
    description: 'Cases that took long to process',
    icon: 'Clock',
    color: 'text-orange-400',
    columnPatterns: ['time', 'duration', 'seconds', 'minutes', 'handling'],
    buildFilter: (col, values) => {
      const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const p75 = numericValues[Math.floor(numericValues.length * 0.75)];
        const threshold = Math.round(p75);
        return [{ column: col, operator: 'greater', value: String(threshold), label: `${col} > ${threshold}` }];
      }
      return null;
    }
  },
  {
    id: 'high_risk',
    label: 'High Risk Score',
    description: 'Cases with high risk/score',
    icon: 'TrendingUp',
    color: 'text-rose-400',
    columnPatterns: ['risk', 'score', 'probability', 'confidence', 'rating'],
    buildFilter: (col, values) => {
      const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const max = numericValues[numericValues.length - 1];
        // Use 70% of max as threshold for "high"
        const threshold = max <= 1 ? 0.7 : Math.round(max * 0.7);
        return [{ column: col, operator: 'greater', value: String(threshold), label: `${col} > ${threshold}` }];
      }
      return null;
    }
  },
  {
    id: 'mismatch',
    label: 'Mismatch Cases',
    description: 'Cases with mismatched values',
    icon: 'MapPin',
    color: 'text-violet-400',
    columnPatterns: ['mismatch', 'different', 'discrepancy'],
    buildFilter: (col, values) => {
      const hasTrue = values.some(v => ['1', 'true', 'yes', 'True', 'Yes'].includes(String(v)));
      if (hasTrue) {
        const trueVal = values.find(v => ['1', 'true', 'yes', 'True', 'Yes'].includes(String(v))) || '1';
        return [{ column: col, operator: 'equals', value: String(trueVal), label: `${col} = ${trueVal}` }];
      }
      return null;
    }
  }
];

// Function to generate dynamic smart filters based on actual data columns
function generateDynamicSmartFilters(
  columns: string[], 
  sampleData: Record<string, any>[]
): SavedSmartFilter[] {
  const dynamicFilters: SavedSmartFilter[] = [];
  
  // Get sample values for each column
  const columnValues: Record<string, string[]> = {};
  columns.forEach(col => {
    const values = sampleData
      .slice(0, 100)  // Sample first 100 rows
      .map(row => row[col])
      .filter(v => v !== null && v !== undefined && v !== '');
    columnValues[col] = [...new Set(values.map(v => String(v)))];  // Unique values
  });
  
  // Try to match each template
  SMART_FILTER_TEMPLATES.forEach(template => {
    // Find columns that match the pattern
    const matchingColumns = columns.filter(col => 
      template.columnPatterns.some(pattern => 
        col.toLowerCase().includes(pattern.toLowerCase())
      )
    );
    
    // Try to build a filter for each matching column
    for (const col of matchingColumns) {
      const filterDef = template.buildFilter(col, columnValues[col] || []);
      if (filterDef) {
        dynamicFilters.push({
          id: `dynamic_${template.id}_${col}`,
          label: `${template.label} (${col})`,
          description: template.description,
          icon: template.icon,
          color: template.color,
          filters: filterDef.map((f, i) => ({ ...f, id: `${template.id}_${col}_${i}` })),
          isBuiltIn: true,
          createdAt: '',
          usageCount: 0,
        });
        break;  // Only use first matching column for each template
      }
    }
  });
  
  return dynamicFilters;
}

export default function TxnJsonInput({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<Record<string, any>[]>([]);
  const [filteredData, setFilteredData] = useState<Record<string, any>[]>([]);
  const [activeFilters, setActiveFilters] = useState<CaseFilter[]>([]);
  const [selectedCases, setSelectedCases] = useState<SelectedCase[]>([]);
  const [expandedCase, setExpandedCase] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [customFilterColumn, setCustomFilterColumn] = useState('');
  const [customFilterValue, setCustomFilterValue] = useState('');
  const [customFilterOperator, setCustomFilterOperator] = useState<'equals' | 'not_equals' | 'greater' | 'less' | 'contains'>('equals');
  const [columns, setColumns] = useState<string[]>([]);
  
  // Smart filters state
  const [savedFilters, setSavedFilters] = useState<SavedSmartFilter[]>([]);
  const [dynamicFilters, setDynamicFilters] = useState<SavedSmartFilter[]>([]);
  const [showSaveFilterModal, setShowSaveFilterModal] = useState(false);
  const [newFilterName, setNewFilterName] = useState('');
  const [newFilterDescription, setNewFilterDescription] = useState('');

  // Load saved smart filters from localStorage
  useEffect(() => {
    const userFilters = getGlobalSmartFilters();
    setSavedFilters(userFilters);
  }, []);

  // Generate dynamic filters when data is loaded
  useEffect(() => {
    if (csvData.length > 0 && columns.length > 0) {
      const generated = generateDynamicSmartFilters(columns, csvData);
      setDynamicFilters(generated);
    }
  }, [csvData, columns]);

  // Combine dynamic and saved filters, sorted by usage
  const allSmartFilters = useCallback(() => {
    const combined = [...dynamicFilters, ...savedFilters];
    // Sort: dynamic first, then saved by usage count
    return combined.sort((a, b) => {
      if (a.isBuiltIn && !b.isBuiltIn) return -1;
      if (!a.isBuiltIn && b.isBuiltIn) return 1;
      return b.usageCount - a.usageCount;
    });
  }, [dynamicFilters, savedFilters]);

  // Load CSV data from backend
  useEffect(() => {
    const loadData = async () => {
      if (!context?.session.hasCsvData) return;
      
      setLoading(true);
      try {
        const result = await getCsvData(context.session.id);
        if (result.data && result.data.length > 0) {
          setCsvData(result.data);
          setFilteredData(result.data);
          setColumns(Object.keys(result.data[0]));
        }
      } catch (e) {
        setError('Failed to load case data');
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [context?.session.id, context?.session.hasCsvData]);

  // Apply filters
  const applyFilters = useCallback(() => {
    if (activeFilters.length === 0) {
      setFilteredData(csvData);
      return;
    }

    const filtered = csvData.filter(row => {
      return activeFilters.every(filter => {
        const value = String(row[filter.column] ?? '').toLowerCase();
        const filterValue = filter.value.toLowerCase();
        
        switch (filter.operator) {
          case 'equals':
            return value === filterValue;
          case 'not_equals':
            return value !== filterValue;
          case 'greater':
            return parseFloat(value) > parseFloat(filterValue);
          case 'less':
            return parseFloat(value) < parseFloat(filterValue);
          case 'contains':
            return value.includes(filterValue);
          default:
            return true;
        }
      });
    });

    setFilteredData(filtered);
  }, [csvData, activeFilters]);

  useEffect(() => {
    applyFilters();
  }, [applyFilters]);

  // Apply smart filter
  const applySmartFilter = (smartFilter: SavedSmartFilter) => {
    const newFilters: CaseFilter[] = smartFilter.filters.map((f, i) => ({
      id: `${smartFilter.id}_${i}`,
      column: f.column,
      operator: f.operator as CaseFilter['operator'],
      value: f.value,
      label: f.label || `${f.column} ${f.operator} ${f.value}`
    }));
    setActiveFilters(newFilters);
    
    // Update usage count for non-builtin filters
    if (!smartFilter.isBuiltIn) {
      updateSmartFilterUsage(smartFilter.id);
      // Refresh saved filters to update UI
      setSavedFilters(getGlobalSmartFilters());
    }
  };

  // Save current filters as a new smart filter
  const saveCurrentAsSmartFilter = () => {
    if (!newFilterName.trim() || activeFilters.length === 0) return;
    
    const newFilter = saveGlobalSmartFilter({
      label: newFilterName.trim(),
      description: newFilterDescription.trim() || `Custom filter with ${activeFilters.length} condition(s)`,
      icon: 'Star',
      color: 'text-cyan-400',
      filters: activeFilters.map(f => ({
        id: f.id,
        column: f.column,
        operator: f.operator,
        value: f.value,
        label: f.label,
      })),
      isBuiltIn: false,
    });
    
    setSavedFilters([...savedFilters, newFilter]);
    setShowSaveFilterModal(false);
    setNewFilterName('');
    setNewFilterDescription('');
  };

  // Delete a saved smart filter
  const handleDeleteSmartFilter = (filterId: string) => {
    deleteGlobalSmartFilter(filterId);
    setSavedFilters(savedFilters.filter(f => f.id !== filterId));
  };

  // Get operator symbol for display
  const getOperatorSymbol = (op: string) => {
    switch (op) {
      case 'equals': return '=';
      case 'not_equals': return '≠';
      case 'greater': return '>';
      case 'less': return '<';
      case 'contains': return '∋';
      default: return op;
    }
  };

  // Add custom filter
  const addCustomFilter = () => {
    if (!customFilterColumn || !customFilterValue) return;
    
    const operatorSymbol = getOperatorSymbol(customFilterOperator);
    const newFilter: CaseFilter = {
      id: `custom_${Date.now()}`,
      column: customFilterColumn,
      operator: customFilterOperator,
      value: customFilterValue,
      label: `${customFilterColumn} ${operatorSymbol} ${customFilterValue}`
    };
    setActiveFilters([...activeFilters, newFilter]);
    setCustomFilterColumn('');
    setCustomFilterValue('');
  };

  // Remove filter
  const removeFilter = (filterId: string) => {
    setActiveFilters(activeFilters.filter(f => f.id !== filterId));
  };

  // Clear all filters
  const clearFilters = () => {
    setActiveFilters([]);
    setFilteredData(csvData);
  };

  // Toggle case selection
  const toggleCaseSelection = (index: number, data: Record<string, any>) => {
    const existing = selectedCases.find(c => c.index === index);
    if (existing) {
      setSelectedCases(selectedCases.filter(c => c.index !== index));
    } else {
      setSelectedCases([...selectedCases, { index, data }]);
    }
  };

  // Select all visible cases
  const selectAllVisible = () => {
    const newSelections = filteredData.slice(0, 20).map((data, idx) => ({ index: idx, data }));
    setSelectedCases(newSelections);
  };

  // Proceed with selected cases
  const handleProceed = () => {
    if (selectedCases.length === 0) {
      setError('Please select at least one case to analyze');
      return;
    }
    
    // Store selected cases as txnJson
    context?.setTxnJson({
      selectedCases: selectedCases.map(c => c.data),
      filters: activeFilters,
      totalCases: selectedCases.length,
      selectedAt: new Date().toISOString()
    });
    
    onComplete();
  };

  // Get decision badge color
  const getDecisionColor = (decision: string) => {
    const d = String(decision).toLowerCase();
    if (d === 'fraud' || d === '1' || d === 'true') return 'bg-red-500/20 text-red-300 border-red-500/30';
    if (d === 'legit' || d === '0' || d === 'false') return 'bg-green-500/20 text-green-300 border-green-500/30';
    return 'bg-slate-500/20 text-slate-300 border-slate-500/30';
  };

  // Check if data is available
  const hasData = context?.session.hasCsvData && csvData.length > 0;

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl mb-4 shadow-lg shadow-amber-500/20">
          <Search className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Select Cases to Analyze</h2>
        <p className="text-slate-400 max-w-lg mx-auto">
          Filter and select specific fraud alert cases to analyze analyst decisions. Use smart filters or create custom ones.
        </p>
      </div>

      {/* No data warning */}
      {!hasData && !loading && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 text-center mb-6">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-amber-300 mb-2">No Data Available</h3>
          <p className="text-amber-200/80 text-sm">
            Please upload a CSV file in the Data step first to browse and select cases.
          </p>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 text-amber-400 animate-spin" />
          <span className="ml-3 text-slate-400">Loading case data...</span>
        </div>
      )}

      {hasData && (
        <>
          {/* Smart Filters */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                Smart Filters
                {savedFilters.length > 0 && (
                  <span className="text-xs bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded">
                    +{savedFilters.length} saved
                  </span>
                )}
              </h3>
            </div>
            {allSmartFilters().length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {allSmartFilters().map((filter) => {
                  const Icon = ICON_MAP[filter.icon || 'Filter'] || Filter;
                  return (
                    <div
                      key={filter.id}
                      className="relative group"
                    >
                      <button
                        onClick={() => applySmartFilter(filter)}
                        className="w-full flex items-start gap-3 p-3 bg-slate-900/50 border border-slate-800 rounded-xl hover:border-slate-700 transition-all text-left"
                      >
                        <div className={`p-1.5 rounded-lg bg-slate-800 group-hover:bg-slate-700 ${filter.color || 'text-slate-400'}`}>
                          <Icon className="w-4 h-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="text-sm font-medium text-white truncate">{filter.label}</span>
                            {!filter.isBuiltIn && (
                              <Star className="w-3 h-3 text-cyan-400 flex-shrink-0" />
                            )}
                          </div>
                          <div className="text-xs text-slate-500 truncate">{filter.description}</div>
                          {filter.usageCount > 0 && (
                            <div className="text-xs text-slate-600 mt-0.5">Used {filter.usageCount}x</div>
                          )}
                        </div>
                      </button>
                      {/* Delete button for user-created filters */}
                      {!filter.isBuiltIn && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteSmartFilter(filter.id);
                          }}
                          className="absolute top-1 right-1 p-1 bg-red-500/20 hover:bg-red-500/40 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                          title="Delete filter"
                        >
                          <Trash2 className="w-3 h-3 text-red-400" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-6 text-slate-500">
                <Filter className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No smart filters detected for your data.</p>
                <p className="text-xs mt-1">Create custom filters below and save them for quick access.</p>
              </div>
            )}
          </div>

          {/* Active Filters & Custom Filter */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-medium text-white">Active Filters</span>
                {activeFilters.length > 0 && (
                  <span className="text-xs bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded-full">
                    {activeFilters.length}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {activeFilters.length > 0 && (
                  <button
                    onClick={() => setShowSaveFilterModal(true)}
                    className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                  >
                    <Save className="w-3 h-3" />
                    Save as Smart Filter
                  </button>
                )}
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1"
                >
                  <Plus className="w-3 h-3" />
                  Add Filter
                </button>
                {activeFilters.length > 0 && (
                  <button
                    onClick={clearFilters}
                    className="text-xs text-red-400 hover:text-red-300"
                  >
                    Clear All
                  </button>
                )}
              </div>
            </div>

            {/* Active filter chips */}
            {activeFilters.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {activeFilters.map((filter) => (
                  <div
                    key={filter.id}
                    className="flex items-center gap-1.5 px-2 py-1 bg-amber-500/20 border border-amber-500/30 rounded-lg text-xs text-amber-300"
                  >
                    <span className="font-mono">{filter.label}</span>
                    <button
                      onClick={() => removeFilter(filter.id)}
                      className="hover:text-amber-100"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Custom filter form */}
            <AnimatePresence>
              {showFilters && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex flex-wrap gap-2 pt-3 border-t border-slate-800"
                >
                  <select
                    value={customFilterColumn}
                    onChange={(e) => setCustomFilterColumn(e.target.value)}
                    className="flex-1 min-w-[150px] px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white"
                  >
                    <option value="">Select column...</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <select
                    value={customFilterOperator}
                    onChange={(e) => setCustomFilterOperator(e.target.value as any)}
                    className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white"
                  >
                    <option value="equals">=</option>
                    <option value="not_equals">≠</option>
                    <option value="greater">&gt;</option>
                    <option value="less">&lt;</option>
                    <option value="contains">contains</option>
                  </select>
                  <input
                    type="text"
                    value={customFilterValue}
                    onChange={(e) => setCustomFilterValue(e.target.value)}
                    placeholder="Value..."
                    className="flex-1 min-w-[120px] px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500"
                  />
                  <button
                    onClick={addCustomFilter}
                    disabled={!customFilterColumn || !customFilterValue}
                    className="px-4 py-2 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 rounded-lg text-white text-sm font-medium"
                  >
                    Add
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results count */}
            <div className="text-xs text-slate-500 mt-2">
              Showing {filteredData.length.toLocaleString()} of {csvData.length.toLocaleString()} cases
            </div>
          </div>

          {/* Cases Table */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden mb-6">
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-white">Cases</h3>
                {selectedCases.length > 0 && (
                  <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full">
                    {selectedCases.length} selected
                  </span>
                )}
              </div>
              <button
                onClick={selectAllVisible}
                className="text-xs text-slate-400 hover:text-white"
              >
                Select first 20
              </button>
            </div>

            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full">
                <thead className="bg-slate-950/50 sticky top-0">
                  <tr className="text-xs text-slate-400">
                    <th className="w-10 px-4 py-2"></th>
                    <th className="text-left px-4 py-2 font-medium">Alert ID</th>
                    <th className="text-left px-4 py-2 font-medium">Amount</th>
                    <th className="text-left px-4 py-2 font-medium">L1 Decision</th>
                    <th className="text-left px-4 py-2 font-medium">L2 Decision</th>
                    <th className="text-left px-4 py-2 font-medium">True Fraud</th>
                    <th className="text-left px-4 py-2 font-medium">Override</th>
                    <th className="w-10 px-4 py-2"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {filteredData.slice(0, 100).map((row, idx) => {
                    const isSelected = selectedCases.some(c => c.index === idx);
                    const isExpanded = expandedCase === idx;
                    
                    return (
                      <React.Fragment key={idx}>
                        <tr 
                          className={`hover:bg-slate-800/30 cursor-pointer transition-colors ${isSelected ? 'bg-amber-500/10' : ''}`}
                          onClick={() => toggleCaseSelection(idx, row)}
                        >
                          <td className="px-4 py-3">
                            <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                              isSelected ? 'bg-amber-500 border-amber-500' : 'border-slate-600'
                            }`}>
                              {isSelected && <Check className="w-3 h-3 text-white" />}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-sm text-white font-mono">
                              {row.alert_id || row.txn_id || row.id || `#${idx + 1}`}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-sm text-slate-300">
                              ${parseFloat(row.transaction_amount || 0).toLocaleString()}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`text-xs px-2 py-0.5 rounded border ${getDecisionColor(row.l1_decision)}`}>
                              {row.l1_decision || 'N/A'}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`text-xs px-2 py-0.5 rounded border ${getDecisionColor(row.l2_decision)}`}>
                              {row.l2_decision || 'N/A'}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`text-xs px-2 py-0.5 rounded border ${getDecisionColor(row.true_fraud_flag)}`}>
                              {row.true_fraud_flag === 1 || row.true_fraud_flag === '1' ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {(row.l1_override_flag === 1 || row.l1_override_flag === '1' || 
                              row.l2_override_flag === 1 || row.l2_override_flag === '1') && (
                              <span className="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                                Override
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setExpandedCase(isExpanded ? null : idx);
                              }}
                              className="text-slate-400 hover:text-white"
                            >
                              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            </button>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={8} className="bg-slate-950/30 px-4 py-4">
                              <div className="grid grid-cols-4 gap-4 text-sm">
                                {Object.entries(row).slice(0, 16).map(([key, value]) => (
                                  <div key={key}>
                                    <div className="text-slate-500 text-xs mb-0.5">{key}</div>
                                    <div className="text-white font-mono text-xs truncate">
                                      {String(value ?? 'N/A')}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>

              {filteredData.length > 100 && (
                <div className="text-center py-4 text-sm text-slate-500">
                  Showing first 100 of {filteredData.length.toLocaleString()} results
                </div>
              )}

              {filteredData.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                  <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No cases match the current filters</p>
                </div>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-300 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-between items-center">
            <div className="text-sm text-slate-400">
              {selectedCases.length > 0 ? (
                <span className="text-emerald-400">
                  {selectedCases.length} case{selectedCases.length > 1 ? 's' : ''} selected for analysis
                </span>
              ) : (
                'Select cases to analyze analyst decisions'
              )}
            </div>
            <button
              onClick={handleProceed}
              disabled={selectedCases.length === 0}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-white font-medium transition-all shadow-lg shadow-amber-500/20"
            >
              Analyze Selected Cases
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </>
      )}

      {/* Save Smart Filter Modal */}
      <AnimatePresence>
        {showSaveFilterModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50"
            onClick={() => setShowSaveFilterModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-cyan-500/20 rounded-lg">
                  <Star className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Save Smart Filter</h3>
                  <p className="text-xs text-slate-400">Save current filters for quick reuse</p>
                </div>
              </div>

              {/* Current filters preview */}
              <div className="mb-4 p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500 mb-2">Filters to save:</div>
                <div className="flex flex-wrap gap-1">
                  {activeFilters.map((f) => (
                    <span key={f.id} className="text-xs bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded">
                      {f.label}
                    </span>
                  ))}
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Filter Name *</label>
                  <input
                    type="text"
                    value={newFilterName}
                    onChange={(e) => setNewFilterName(e.target.value)}
                    placeholder="e.g., High Risk Night Transactions"
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:border-cyan-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Description (optional)</label>
                  <input
                    type="text"
                    value={newFilterDescription}
                    onChange={(e) => setNewFilterDescription(e.target.value)}
                    placeholder="Brief description of what this filter finds"
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:border-cyan-500 outline-none"
                  />
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowSaveFilterModal(false)}
                  className="flex-1 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={saveCurrentAsSmartFilter}
                  disabled={!newFilterName.trim()}
                  className="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded-lg text-sm text-white font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  Save Filter
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
