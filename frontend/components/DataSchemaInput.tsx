"use client";
import { useState, useRef, useContext, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileSpreadsheet, 
  Check, 
  AlertCircle, 
  ArrowRight, 
  Trash2,
  BarChart3,
  Hash,
  Type,
  Calendar,
  HelpCircle,
  TrendingUp,
  AlertTriangle,
  Target,
  Layers,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import Papa from 'papaparse';
import { AppContext } from '@/app/page';

interface ColumnStats {
  name: string;
  dtype: 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'unknown';
  totalCount: number;
  missingCount: number;
  missingPercent: number;
  uniqueCount: number;
  // Numeric stats
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  median?: number;
  // Categorical stats
  topValues?: { value: string; count: number }[];
  // Inferred info
  isPotentialId?: boolean;
  isOneHotEncoded?: boolean;
  isDecisionColumn?: boolean;
  isAnalystColumn?: boolean;
  // User-provided description
  description?: string;
}

// Known column descriptions for fraud alert data
const KNOWN_COLUMN_DESCRIPTIONS: Record<string, string> = {
  'customer_id': 'Unique identifier for the customer',
  'tenure_months': 'Number of months the customer has been with the bank',
  'customer_segment': 'Customer tier such as mass, affluent, or premium',
  'age_band': 'Customer age range',
  'geography': "Customer's home country or region",
  'avg_txn_amount': "Customer's historical average transaction amount",
  'txn_frequency': 'Average number of transactions per month',
  'prior_fraud_count': 'Number of past confirmed fraud cases for the customer',
  'kyc_risk_rating': 'Risk level assigned during KYC',
  'digital_savviness': "Customer's comfort level with digital channels",
  'alert_id': 'Unique identifier for the fraud alert',
  'transaction_amount': 'Amount of the flagged transaction',
  'channel': 'Channel used for the transaction (card, ACH, wire, P2P)',
  'merchant_category': 'Merchant category or MCC',
  'merchant_country': 'Country where the merchant is located',
  'customer_country': "Customer's registered country",
  'geo_mismatch': 'Whether merchant and customer countries differ',
  'is_night_txn': 'Whether the transaction occurred at night',
  'is_new_merchant': 'Whether this is the first transaction with the merchant',
  'velocity_score': 'Score indicating rapid or unusual transaction activity',
  'device_risk_score': "Risk score associated with the customer's device",
  'model_risk_score': 'Fraud probability score produced by the ML model',
  'expected_action': 'Action expected by the system (block, allow, review)',
  'l1_analyst_id': 'Identifier of the Level 1 analyst',
  'l1_tenure_months': 'Experience of the L1 analyst in months',
  'l1_region': 'Region where the L1 analyst operates',
  'l1_avg_cases_per_day': 'Average number of cases handled daily by L1 analyst',
  'l1_risk_appetite': 'Risk tolerance level of the L1 analyst',
  'l1_decision': 'Decision taken by the L1 analyst',
  'l1_override_flag': 'Whether the L1 analyst overrode the system decision',
  'l1_handling_time_sec': 'Time taken by L1 analyst to handle the case',
  'l2_decision': 'Final decision taken at Level 2',
  'customer_response': "Customer's response during verification",
  'verification_outcome': 'Result of customer verification',
  'true_fraud_flag': 'Final confirmed fraud outcome',
  'l2_analyst_id': 'Identifier of the Level 2 analyst',
  'l2_analyst_level': 'Seniority level of the L2 analyst',
  'l2_tenure_months': 'Experience of the L2 analyst in months',
  'l2_region': 'Region where the L2 analyst operates',
  'l2_avg_cases_per_day': 'Average number of cases handled daily by L2 analyst',
  'l2_risk_appetite': 'Risk tolerance level of the L2 analyst',
  'l2_performance_rating': 'Performance rating of the L2 analyst',
  'l2_handling_time_sec': 'Time taken by L2 analyst to handle the case',
  'l2_sla_breach': 'Whether SLA was breached at Level 2',
  'l2_override_flag': 'Whether the L2 analyst overrode a prior decision',
  'l2_override_reason': 'Reason provided for the L2 override',
};

interface DataSchema {
  fileName: string;
  totalRows: number;
  totalColumns: number;
  columns: ColumnStats[];
  memoryEstimate: string;
  dataQuality: {
    totalMissing: number;
    missingPercent: number;
    duplicateRows: number;
    potentialIssues: string[];
  };
  decisionColumns: string[];
  analystColumns: string[];
  potentialIdColumns: string[];
  oneHotGroups: string[][];
}

export default function DataSchemaInput({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schema, setSchema] = useState<DataSchema | null>(null);
  const [expandedColumn, setExpandedColumn] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);

  // Helper to build schema from saved data
  const buildSchemaFromSaved = (savedSchema: any, rowCount: number, fileName: string): DataSchema => ({
    fileName: savedSchema.fileName || fileName || 'restored.csv',
    totalRows: savedSchema.dataset_info?.total_samples || rowCount,
    totalColumns: savedSchema.dataset_info?.total_features || savedSchema.features?.length || 0,
    columns: (savedSchema.features || []).map((f: any) => ({
      name: f.name,
      dtype: f.dtype,
      totalCount: savedSchema.dataset_info?.total_samples || rowCount,
      missingCount: f.missing_count || 0,
      missingPercent: f.missing_percentage || 0,
      uniqueCount: f.unique_values || 0,
      min: f.stats?.min,
      max: f.stats?.max,
      mean: f.stats?.mean,
      std: f.stats?.std,
      topValues: f.top_values ? Object.entries(f.top_values).map(([value, count]) => ({ value, count: count as number })) : undefined,
      isDecisionColumn: f.is_decision_column,
      isAnalystColumn: f.is_analyst_column,
      isPotentialId: savedSchema.feature_engineering?.potential_id_columns?.includes(f.name),
      isOneHotEncoded: savedSchema.feature_engineering?.one_hot_encoded?.includes(f.name),
      description: f.description || '',
    })),
    memoryEstimate: savedSchema.dataset_info?.memory_usage || 'Unknown',
    dataQuality: savedSchema.data_quality || {
      totalMissing: 0,
      missingPercent: 0,
      duplicateRows: 0,
      potentialIssues: [],
    },
    decisionColumns: savedSchema.decision_columns || [],
    analystColumns: savedSchema.analyst_columns || [],
    potentialIdColumns: savedSchema.feature_engineering?.potential_id_columns || [],
    oneHotGroups: [],
  });

  // Restore schema from session on mount
  useEffect(() => {
    if (!initialized && context?.dataSchema && context.session.hasCsvData) {
      // Rebuild the local schema from saved context (CSV data is on backend)
      const savedSchema = context.dataSchema;
      const restoredSchema = buildSchemaFromSaved(
        savedSchema, 
        context.session.csvRowCount, 
        context.session.csvFileName
      );
      setSchema(restoredSchema);
      setInitialized(true);
    } else if (!initialized) {
      setInitialized(true);
    }
  }, [initialized, context?.dataSchema, context?.session.hasCsvData, context?.session.csvRowCount, context?.session.csvFileName]);

  // Re-sync when session changes
  useEffect(() => {
    if (context?.session.id) {
      if (context.dataSchema && context.session.hasCsvData) {
        // Session has data, restore it
        const savedSchema = context.dataSchema;
        const restoredSchema = buildSchemaFromSaved(
          savedSchema,
          context.session.csvRowCount,
          context.session.csvFileName
        );
        setSchema(restoredSchema);
      } else {
        // New session or session without data
        setSchema(null);
      }
    }
  }, [context?.session.id]);

  const inferDataType = (values: any[]): 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'unknown' => {
    const nonNull = values.filter(v => v !== null && v !== undefined && v !== '');
    if (nonNull.length === 0) return 'unknown';

    // Check boolean
    const boolValues = new Set(nonNull.map(v => String(v).toLowerCase()));
    if (boolValues.size <= 2 && 
        [...boolValues].every(v => ['true', 'false', '0', '1', 'yes', 'no'].includes(v))) {
      return 'boolean';
    }

    // Check numeric
    const numericCount = nonNull.filter(v => !isNaN(parseFloat(v)) && isFinite(Number(v))).length;
    if (numericCount / nonNull.length > 0.9) return 'numeric';

    // Check datetime
    const datePatterns = [
      /^\d{4}-\d{2}-\d{2}/, // YYYY-MM-DD
      /^\d{2}\/\d{2}\/\d{4}/, // MM/DD/YYYY
      /^\d{2}-\d{2}-\d{4}/, // DD-MM-YYYY
    ];
    const dateCount = nonNull.filter(v => datePatterns.some(p => p.test(String(v)))).length;
    if (dateCount / nonNull.length > 0.8) return 'datetime';

    return 'categorical';
  };

  const calculateStats = (values: any[], dtype: string): Partial<ColumnStats> => {
    const nonNull = values.filter(v => v !== null && v !== undefined && v !== '');
    
    if (dtype === 'numeric') {
      const nums = nonNull.map(v => parseFloat(v)).filter(n => !isNaN(n));
      if (nums.length === 0) return {};
      
      const sorted = [...nums].sort((a, b) => a - b);
      const sum = nums.reduce((a, b) => a + b, 0);
      const mean = sum / nums.length;
      const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length;
      
      return {
        min: Math.min(...nums),
        max: Math.max(...nums),
        mean: Math.round(mean * 100) / 100,
        std: Math.round(Math.sqrt(variance) * 100) / 100,
        median: sorted[Math.floor(sorted.length / 2)],
      };
    }

    if (dtype === 'categorical' || dtype === 'boolean') {
      const counts: Record<string, number> = {};
      nonNull.forEach(v => {
        const key = String(v);
        counts[key] = (counts[key] || 0) + 1;
      });
      
      const topValues = Object.entries(counts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([value, count]) => ({ value, count }));

      return { topValues };
    }

    return {};
  };

  const detectOneHotGroups = (columnNames: string[]): string[][] => {
    const groups: Map<string, string[]> = new Map();
    
    columnNames.forEach(col => {
      // Check for common one-hot patterns: prefix_value, prefix.value
      const patterns = [
        /^(.+)_([^_]+)$/,
        /^(.+)\.([^.]+)$/,
      ];
      
      for (const pattern of patterns) {
        const match = col.match(pattern);
        if (match) {
          const prefix = match[1];
          if (!groups.has(prefix)) groups.set(prefix, []);
          groups.get(prefix)!.push(col);
          break;
        }
      }
    });

    // Only return groups with 2+ columns that look like one-hot
    return Array.from(groups.entries())
      .filter(([, cols]) => cols.length >= 2 && cols.length <= 20)
      .map(([, cols]) => cols);
  };

  const processCSV = useCallback((file: File) => {
    setIsProcessing(true);
    setError(null);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const data = results.data as Record<string, any>[];
          if (data.length === 0) {
            setError('CSV file is empty or has no valid data rows');
            setIsProcessing(false);
            return;
          }

          const columnNames = Object.keys(data[0]);
          const columns: ColumnStats[] = [];
          let totalMissing = 0;

          columnNames.forEach(colName => {
            const values = data.map(row => row[colName]);
            const missingCount = values.filter(v => v === null || v === undefined || v === '').length;
            totalMissing += missingCount;

            const dtype = inferDataType(values);
            const stats = calculateStats(values, dtype);
            const uniqueCount = new Set(values.filter(v => v !== null && v !== undefined && v !== '')).size;

            // Infer column purpose for fraud analyst decision data
            const lowerName = colName.toLowerCase();
            const isDecisionColumn = ['l1_decision', 'l2_decision', 'true_fraud_flag', 'expected_action', 'verification_outcome'].some(t => lowerName === t);
            const isAnalystColumn = lowerName.startsWith('l1_') || lowerName.startsWith('l2_');
            const isPotentialId = ['id', '_id', 'uuid', 'index', 'key'].some(t => lowerName.includes(t)) || 
                                   (dtype === 'numeric' && uniqueCount === data.length);
            const isOneHotEncoded = dtype === 'boolean' && (lowerName.includes('_') || lowerName.includes('.'));

            // Get known description or empty
            const description = KNOWN_COLUMN_DESCRIPTIONS[colName] || KNOWN_COLUMN_DESCRIPTIONS[lowerName] || '';

            columns.push({
              name: colName,
              dtype,
              totalCount: values.length,
              missingCount,
              missingPercent: Math.round((missingCount / values.length) * 1000) / 10,
              uniqueCount,
              isDecisionColumn,
              isAnalystColumn,
              isPotentialId,
              isOneHotEncoded,
              description,
              ...stats,
            });
          });

          // Detect one-hot groups
          const oneHotGroups = detectOneHotGroups(columnNames);
          
          // Find potential issues
          const potentialIssues: string[] = [];
          const highMissingCols = columns.filter(c => c.missingPercent > 20);
          if (highMissingCols.length > 0) {
            potentialIssues.push(`${highMissingCols.length} column(s) have >20% missing values`);
          }
          
          const lowVarianceCols = columns.filter(c => c.dtype === 'numeric' && c.std !== undefined && c.std === 0);
          if (lowVarianceCols.length > 0) {
            potentialIssues.push(`${lowVarianceCols.length} numeric column(s) have zero variance`);
          }

          const singleValueCols = columns.filter(c => c.uniqueCount === 1);
          if (singleValueCols.length > 0) {
            potentialIssues.push(`${singleValueCols.length} column(s) have only one unique value`);
          }

          // Estimate memory
          const bytesPerRow = columnNames.length * 8; // rough estimate
          const totalBytes = bytesPerRow * data.length;
          const memoryEstimate = totalBytes < 1024 * 1024 
            ? `${Math.round(totalBytes / 1024)} KB`
            : `${Math.round(totalBytes / (1024 * 1024) * 10) / 10} MB`;

          // Detect duplicates (simple check on first 1000 rows)
          const sampleForDupes = data.slice(0, 1000);
          const rowStrings = new Set(sampleForDupes.map(r => JSON.stringify(r)));
          const duplicateRows = sampleForDupes.length - rowStrings.size;

          const schemaResult: DataSchema = {
            fileName: file.name,
            totalRows: data.length,
            totalColumns: columnNames.length,
            columns,
            memoryEstimate,
            dataQuality: {
              totalMissing,
              missingPercent: Math.round((totalMissing / (data.length * columnNames.length)) * 1000) / 10,
              duplicateRows,
              potentialIssues,
            },
            decisionColumns: columns.filter(c => c.isDecisionColumn).map(c => c.name),
            analystColumns: columns.filter(c => c.isAnalystColumn).map(c => c.name),
            potentialIdColumns: columns.filter(c => c.isPotentialId).map(c => c.name),
            oneHotGroups,
          };

          setSchema(schemaResult);
          
          // Auto-save schema to context
          context?.setDataSchema({
            fileName: schemaResult.fileName,
            dataset_info: {
              total_samples: schemaResult.totalRows,
              total_features: schemaResult.totalColumns,
              memory_usage: schemaResult.memoryEstimate,
            },
            features: schemaResult.columns.map(c => ({
              name: c.name,
              dtype: c.dtype,
              description: c.description || '',
              missing_count: c.missingCount,
              missing_percentage: c.missingPercent,
              unique_values: c.uniqueCount,
              stats: c.dtype === 'numeric' ? { min: c.min, max: c.max, mean: c.mean, std: c.std } : null,
              top_values: c.topValues?.reduce((acc, tv) => ({ ...acc, [tv.value]: tv.count }), {}),
              is_decision_column: c.isDecisionColumn,
              is_analyst_column: c.isAnalystColumn,
            })),
            decision_columns: schemaResult.decisionColumns,
            analyst_columns: schemaResult.analystColumns,
            feature_engineering: {
              one_hot_encoded: schemaResult.oneHotGroups.flat(),
              potential_id_columns: schemaResult.potentialIdColumns,
            },
            data_quality: schemaResult.dataQuality,
          });

          // Save raw CSV data to BACKEND (avoids localStorage quota limits)
          context?.uploadCsvToBackend(data, file.name);

        } catch (e: any) {
          setError(`Failed to process CSV: ${e.message}`);
        } finally {
          setIsProcessing(false);
        }
      },
      error: (err) => {
        setError(`Failed to parse CSV: ${err.message}`);
        setIsProcessing(false);
      }
    });
  }, [context]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    const csvFile = files.find(f => f.name.endsWith('.csv'));
    
    if (csvFile) {
      processCSV(csvFile);
    } else {
      setError('Please upload a CSV file');
    }
  }, [processCSV]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processCSV(file);
    }
  };

  const getTypeIcon = (dtype: string) => {
    switch (dtype) {
      case 'numeric': return <Hash className="w-4 h-4 text-blue-400" />;
      case 'categorical': return <Type className="w-4 h-4 text-purple-400" />;
      case 'datetime': return <Calendar className="w-4 h-4 text-amber-400" />;
      case 'boolean': return <Check className="w-4 h-4 text-emerald-400" />;
      default: return <HelpCircle className="w-4 h-4 text-slate-400" />;
    }
  };

  const getTypeColor = (dtype: string) => {
    switch (dtype) {
      case 'numeric': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'categorical': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      case 'datetime': return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'boolean': return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
      default: return 'bg-slate-500/20 text-slate-300 border-slate-500/30';
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl mb-4 shadow-lg shadow-emerald-500/20">
          <FileSpreadsheet className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Upload Fraud Alert Data</h2>
        <p className="text-slate-400 max-w-lg mx-auto">
          Upload your fraud alert dataset with analyst decisions. We'll analyze how L1 and L2 analysts reach their conclusions.
        </p>
      </div>

      <AnimatePresence mode="wait">
        {!schema ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {/* Upload Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
                isDragging
                  ? 'border-emerald-500 bg-emerald-500/10'
                  : 'border-slate-700 bg-slate-900/30 hover:border-slate-600 hover:bg-slate-900/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              {isProcessing ? (
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-4" />
                  <p className="text-slate-300">Analyzing your data...</p>
                </div>
              ) : (
                <>
                  <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragging ? 'text-emerald-400' : 'text-slate-500'}`} />
                  <p className="text-lg font-medium text-white mb-2">
                    {isDragging ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
                  </p>
                  <p className="text-sm text-slate-500">or click to browse</p>
                  <p className="text-xs text-slate-600 mt-4">Supports CSV files up to 50MB</p>
                </>
              )}
            </div>

            {error && (
              <div className="mt-4 flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-300 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Data upload is mandatory - no skip option */}
            <div className="text-center mt-6">
              <p className="text-xs text-slate-600">
                Upload your fraud alert dataset to begin analyst decision analysis
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Overview Cards */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                  <Layers className="w-4 h-4" />
                  Rows
                </div>
                <div className="text-2xl font-bold text-white">{schema.totalRows.toLocaleString()}</div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                  <BarChart3 className="w-4 h-4" />
                  Columns
                </div>
                <div className="text-2xl font-bold text-white">{schema.totalColumns}</div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                  <Target className="w-4 h-4" />
                  Decision Cols
                </div>
                <div className="text-lg font-bold text-emerald-400 truncate">
                  {schema.decisionColumns.length > 0 ? schema.decisionColumns.length : 'None found'}
                </div>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                  <TrendingUp className="w-4 h-4" />
                  Memory
                </div>
                <div className="text-2xl font-bold text-white">{schema.memoryEstimate}</div>
              </div>
            </div>

            {/* Data Quality Alert */}
            {schema.dataQuality.potentialIssues.length > 0 && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
                <div className="flex items-center gap-2 text-amber-300 font-medium mb-2">
                  <AlertTriangle className="w-4 h-4" />
                  Data Quality Insights
                </div>
                <ul className="space-y-1">
                  {schema.dataQuality.potentialIssues.map((issue, i) => (
                    <li key={i} className="text-sm text-amber-200/80 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                      {issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Key Columns for Analyst Decision Analysis */}
            {(schema.decisionColumns.length > 0 || schema.analystColumns.length > 0) && (
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                <div className="flex items-center gap-2 text-emerald-300 font-medium mb-3">
                  <Target className="w-4 h-4" />
                  Key Columns for Analyst Decision Analysis
                </div>
                <div className="space-y-3">
                  {schema.decisionColumns.length > 0 && (
                    <div>
                      <div className="text-xs text-emerald-400 mb-1.5">Decision Columns (L1/L2 outcomes)</div>
                      <div className="flex flex-wrap gap-2">
                        {schema.decisionColumns.map(col => (
                          <span key={col} className="px-2 py-1 bg-emerald-500/20 text-emerald-300 rounded text-xs font-mono">
                            {col}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {schema.analystColumns.length > 0 && (
                    <div>
                      <div className="text-xs text-violet-400 mb-1.5">Analyst Columns (L1/L2 attributes)</div>
                      <div className="flex flex-wrap gap-2">
                        {schema.analystColumns.filter(c => !schema.decisionColumns.includes(c)).slice(0, 10).map(col => (
                          <span key={col} className="px-2 py-1 bg-violet-500/20 text-violet-300 rounded text-xs font-mono">
                            {col}
                          </span>
                        ))}
                        {schema.analystColumns.filter(c => !schema.decisionColumns.includes(c)).length > 10 && (
                          <span className="px-2 py-1 text-violet-400 text-xs">
                            +{schema.analystColumns.filter(c => !schema.decisionColumns.includes(c)).length - 10} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Column Type Distribution */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
              <h3 className="text-sm font-medium text-white mb-3">Column Types</h3>
              <div className="flex gap-3 flex-wrap">
                {['numeric', 'categorical', 'boolean', 'datetime'].map(dtype => {
                  const count = schema.columns.filter(c => c.dtype === dtype).length;
                  if (count === 0) return null;
                  return (
                    <div key={dtype} className={`px-3 py-2 rounded-lg border ${getTypeColor(dtype)}`}>
                      <div className="flex items-center gap-2">
                        {getTypeIcon(dtype)}
                        <span className="text-sm font-medium capitalize">{dtype}</span>
                        <span className="text-xs opacity-70">({count})</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Columns Table */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-medium text-white">Column Details</h3>
                <span className="text-xs text-slate-500">{schema.columns.length} columns</span>
              </div>
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-slate-950/50 sticky top-0">
                    <tr className="text-xs text-slate-400">
                      <th className="text-left px-4 py-2 font-medium">Column</th>
                      <th className="text-left px-4 py-2 font-medium">Type</th>
                      <th className="text-right px-4 py-2 font-medium">Unique</th>
                      <th className="text-right px-4 py-2 font-medium">Missing</th>
                      <th className="text-left px-4 py-2 font-medium">Stats</th>
                      <th className="px-4 py-2 font-medium"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {schema.columns.map((col) => (
                      <>
                        <tr 
                          key={col.name}
                          className="hover:bg-slate-800/30 cursor-pointer transition-colors"
                          onClick={() => setExpandedColumn(expandedColumn === col.name ? null : col.name)}
                        >
                          <td className="px-4 py-3">
                            <div className="flex flex-col gap-1">
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-white font-medium">{col.name}</span>
                                {col.isDecisionColumn && (
                                  <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/20 text-emerald-300 rounded">DECISION</span>
                                )}
                                {col.isAnalystColumn && !col.isDecisionColumn && (
                                  <span className="text-[10px] px-1.5 py-0.5 bg-violet-500/20 text-violet-300 rounded">ANALYST</span>
                                )}
                                {col.isPotentialId && (
                                  <span className="text-[10px] px-1.5 py-0.5 bg-slate-500/20 text-slate-400 rounded">ID</span>
                                )}
                              </div>
                              {col.description && (
                                <span className="text-xs text-slate-500 truncate max-w-[300px]">{col.description}</span>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-1.5">
                              {getTypeIcon(col.dtype)}
                              <span className="text-xs text-slate-300 capitalize">{col.dtype}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="text-sm text-slate-300">{col.uniqueCount.toLocaleString()}</span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className={`text-sm ${col.missingPercent > 10 ? 'text-amber-400' : 'text-slate-300'}`}>
                              {col.missingPercent}%
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {col.dtype === 'numeric' && col.mean !== undefined && (
                              <span className="text-xs text-slate-400">
                                μ={col.mean}, σ={col.std}
                              </span>
                            )}
                            {col.dtype === 'categorical' && col.topValues && (
                              <span className="text-xs text-slate-400">
                                Top: {col.topValues[0]?.value}
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-center">
                            {expandedColumn === col.name ? (
                              <ChevronUp className="w-4 h-4 text-slate-500" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-slate-500" />
                            )}
                          </td>
                        </tr>
                        {expandedColumn === col.name && (
                          <tr key={`${col.name}-expanded`}>
                            <td colSpan={6} className="bg-slate-950/30 px-4 py-4">
                              {/* Description Field */}
                              <div className="mb-4">
                                <label className="text-slate-500 text-xs mb-1 block">Description</label>
                                <input
                                  type="text"
                                  value={col.description || ''}
                                  onChange={(e) => {
                                    const newColumns = schema.columns.map(c => 
                                      c.name === col.name ? { ...c, description: e.target.value } : c
                                    );
                                    const newSchema = { ...schema, columns: newColumns };
                                    setSchema(newSchema);
                                    // Update context with new description
                                    if (context?.dataSchema) {
                                      const updatedFeatures = context.dataSchema.features.map((f: any) =>
                                        f.name === col.name ? { ...f, description: e.target.value } : f
                                      );
                                      context.setDataSchema({ ...context.dataSchema, features: updatedFeatures });
                                    }
                                  }}
                                  placeholder="Add a description for this column..."
                                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
                                />
                              </div>
                              <div className="grid grid-cols-3 gap-4 text-sm">
                                {col.dtype === 'numeric' && (
                                  <>
                                    <div>
                                      <div className="text-slate-500 text-xs mb-1">Range</div>
                                      <div className="text-white">{col.min} → {col.max}</div>
                                    </div>
                                    <div>
                                      <div className="text-slate-500 text-xs mb-1">Mean ± Std</div>
                                      <div className="text-white">{col.mean} ± {col.std}</div>
                                    </div>
                                    <div>
                                      <div className="text-slate-500 text-xs mb-1">Median</div>
                                      <div className="text-white">{col.median}</div>
                                    </div>
                                  </>
                                )}
                                {(col.dtype === 'categorical' || col.dtype === 'boolean') && col.topValues && (
                                  <div className="col-span-3">
                                    <div className="text-slate-500 text-xs mb-2">Top Values</div>
                                    <div className="flex flex-wrap gap-2">
                                      {col.topValues.map((tv, i) => (
                                        <div key={i} className="px-2 py-1 bg-slate-800 rounded text-xs">
                                          <span className="text-white">{tv.value}</span>
                                          <span className="text-slate-500 ml-1">({tv.count})</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* One-Hot Groups */}
            {schema.oneHotGroups.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-white font-medium mb-3">
                  <Info className="w-4 h-4 text-blue-400" />
                  Detected One-Hot Encoded Groups
                </div>
                <div className="space-y-2">
                  {schema.oneHotGroups.map((group, i) => (
                    <div key={i} className="text-sm">
                      <span className="text-slate-400">Group {i + 1}:</span>
                      <span className="text-blue-300 ml-2">{group.join(', ')}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-between pt-4">
              <button
                onClick={() => {
                  setSchema(null);
                  context?.setDataSchema(null);
                }}
                className="flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Upload Different File
              </button>
              <button
                onClick={onComplete}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-xl text-white font-medium transition-all shadow-lg shadow-emerald-500/20"
              >
                Continue to Code
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
