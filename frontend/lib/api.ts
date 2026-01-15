const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function explainGlobal(request: any) {
  const res = await fetch(`${API_URL}/explain/global`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to fetch global explanation');
  return res.json();
}

export async function explainTransaction(request: any) {
  const res = await fetch(`${API_URL}/explain/transaction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to fetch transaction explanation');
  return res.json();
}

export interface ChatApiResponse {
  response: string;
  global_json_suggestion: string | null;
}

export interface TxnChatApiResponse {
  response: string;
  txn_json_suggestion: string | null;
  what_if_insight: string | null;
  risk_flag: string | null;
  shadow_rule_detected: string | null;
  guideline_reference: string | null;
  compliance_note: string | null;
}

// Shadow Rules Conversion
export interface HumanReadableShadowRule {
  simple_rule: string;
  original_rule: string;
  target_decision: string;
  predicted_outcome: string;
  key_factors: string[];
  confidence_level: string;
  samples_affected: number;
}

export interface ConvertShadowRulesResponse {
  success: boolean;
  rules: HumanReadableShadowRule[];
  count: number;
}

export async function convertShadowRules(
  l1Rules: string[],
  l2Rules: string[],
  dataSchema?: any,
  existingRules?: string[]
): Promise<ConvertShadowRulesResponse> {
  const res = await fetch(`${API_URL}/convert-shadow-rules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      l1_rules: l1Rules,
      l2_rules: l2Rules,
      data_schema: dataSchema,
      existing_rules: existingRules || []
    }),
  });
  if (!res.ok) throw new Error('Failed to convert shadow rules');
  return res.json();
}

// ============ Shadow Rules Semantic Search / Deduplication ============

export interface SimilarShadowRule {
  rule_id: string;
  rule_text: string;
  simple_rule: string;
  similarity_score: number;
  source_analysis: string;
  target_decision: string;
  predicted_outcome: string;
  is_duplicate: boolean;
}

export interface DeduplicationResult {
  is_duplicate: boolean;
  similar_rules: SimilarShadowRule[];
  suggested_action: 'save_new' | 'use_existing' | 'review';
}

export interface VectorStoreStats {
  session_id: string;
  total_rules: number;
  decision_tree_rules: number;
  chat_discovered_rules: number;
  manual_rules: number;
  index_size: number;
}

export interface AddRuleToIndexRequest {
  rule_id: string;
  rule_text: string;
  source_analysis: string;
  simple_rule?: string;
  target_decision?: string;
  predicted_outcome?: string;
  confidence_level?: string;
  samples_affected?: number;
}

export async function checkShadowRuleDuplicate(
  sessionId: string,
  ruleText: string,
  threshold: number = 0.95
): Promise<DeduplicationResult> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/check-duplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rule_text: ruleText, threshold }),
  });
  if (!res.ok) throw new Error('Failed to check duplicate');
  return res.json();
}

export async function addShadowRuleToIndex(
  sessionId: string,
  request: AddRuleToIndexRequest
): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/add-to-index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to add rule to index');
  return res.json();
}

export async function addShadowRulesBulk(
  sessionId: string,
  rules: AddRuleToIndexRequest[]
): Promise<{ success: boolean; added_count: number; message: string }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/add-bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rules }),
  });
  if (!res.ok) throw new Error('Failed to add rules in bulk');
  return res.json();
}

export async function removeShadowRuleFromIndex(
  sessionId: string,
  ruleId: string
): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/remove-from-index/${ruleId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to remove rule from index');
  return res.json();
}

export async function clearShadowRulesBySource(
  sessionId: string,
  source: 'decision-tree' | 'chat-discovered' | 'manual'
): Promise<{ success: boolean; deleted_count: number; message: string }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/clear-by-source/${source}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to clear rules by source');
  return res.json();
}

export async function searchSimilarShadowRules(
  sessionId: string,
  query: string,
  topK: number = 5,
  threshold: number = 0.0
): Promise<{ results: SimilarShadowRule[]; count: number }> {
  const params = new URLSearchParams({
    query,
    top_k: topK.toString(),
    threshold: threshold.toString(),
  });
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/search?${params}`);
  if (!res.ok) throw new Error('Failed to search similar rules');
  return res.json();
}

export async function getShadowRulesStats(
  sessionId: string
): Promise<VectorStoreStats> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/stats`);
  if (!res.ok) throw new Error('Failed to get stats');
  return res.json();
}

export async function getAllShadowRulesFromIndex(
  sessionId: string
): Promise<{ rules: any[]; count: number }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/all`);
  if (!res.ok) throw new Error('Failed to get all rules');
  return res.json();
}

export async function rebuildShadowRulesIndex(
  sessionId: string
): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/shadow-rules/rebuild-index`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to rebuild index');
  return res.json();
}


// --- Dedicated Shadow Rule Extraction ---

export interface ExtractedShadowRule {
  rule_description: string;
  target_decision: string;
  predicted_outcome: string;
  reasoning: string;
  confidence: string;
  affected_transactions: number;
  key_features: string[];
}

export interface ExtractShadowRulesResponse {
  success: boolean;
  shadow_rules: ExtractedShadowRule[];
  summary: string;
  transactions_analyzed: number;
  rules_count: number;
}

export interface ExtractShadowRulesRequest {
  session_id: string;
  selected_transactions: any[];
  data_schema?: any;
  decision_tree_rules?: {
    l1_decision_rules?: string[];
    l2_decision_rules?: string[];
    l1_accuracy?: number;
    l2_accuracy?: number;
    l1_top_features?: string[];
    l2_top_features?: string[];
  };
  chat_history?: { role: string; content: string }[];
}

export async function extractShadowRulesFromTransactions(
  request: ExtractShadowRulesRequest
): Promise<ExtractShadowRulesResponse> {
  const res = await fetch(`${API_URL}/extract-shadow-rules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to extract shadow rules');
  return res.json();
}


export async function chat(request: { session_id: string; message: string; context?: any }): Promise<ChatApiResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to send chat message');
  return res.json();
}

export async function chatTxn(request: { session_id: string; message: string; context?: any }): Promise<TxnChatApiResponse> {
  const res = await fetch(`${API_URL}/chat-txn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to send transaction chat message');
  return res.json();
}

export async function guideCode(session_id: string, code: string, data_schema?: string) {
  const res = await fetch(`${API_URL}/guide-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, code, data_schema }),
  });
  if (!res.ok) throw new Error('Failed to get code guidance');
  return res.json();
}

export async function guideTxnCode(session_id: string, code: string, global_json_context?: string) {
  const res = await fetch(`${API_URL}/guide-txn-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, code, global_json_context }),
  });
  if (!res.ok) throw new Error('Failed to get transaction code guidance');
  return res.json();
}

// ============ Session API ============

export interface SessionSummary {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  hasCode: boolean;
  codeLength: number;
  hasGlobalJson: boolean;
  hasTxnJson: boolean;
  messageCount: number;
  step: string;
}

export interface CodeSuggestionApi {
  id: string;
  type: string;
  title: string;
  description: string;
  code?: string;
  timestamp: string;
  fromStep: string;
  dismissed: boolean;
}

export interface ContextSnapshotApi {
  hasCode: boolean;
  codeLength?: number;
  globalJsonVersion?: string;
  globalJsonFeatureCount?: number;
  txnId?: string;
  timestamp: string;
}

export interface ChatMessageApi {
  type: 'user' | 'ai';
  content: string;
  context?: ContextSnapshotApi;
}

export interface FullSession {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  mlCode: string;
  dataSchema: any;
  csvFileName: string;
  hasCsvData: boolean;
  csvRowCount: number;
  globalJson: any;
  txnJson: any;
  chatHistory: {
    codeAnalyzer: ChatMessageApi[];
    globalChat: ChatMessageApi[];
    txnChat: ChatMessageApi[];
  };
  suggestions?: CodeSuggestionApi[];
  step?: string;
}

// ============ CSV Data API ============

export interface CsvDataResponse {
  fileName: string;
  rowCount: number;
  data: Record<string, any>[];
}

export async function uploadCsvData(sessionId: string, csvData: Record<string, any>[], fileName: string): Promise<{ success: boolean; rowCount: number }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/csv`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ csv_data: csvData, file_name: fileName }),
  });
  if (!res.ok) throw new Error('Failed to upload CSV data');
  return res.json();
}

export async function getCsvData(sessionId: string): Promise<CsvDataResponse> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/csv`);
  if (!res.ok) throw new Error('Failed to get CSV data');
  return res.json();
}

export async function getCsvRow(sessionId: string, rowIndex: number): Promise<{ row: Record<string, any>; index: number }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/csv/row/${rowIndex}`);
  if (!res.ok) throw new Error('Failed to get CSV row');
  return res.json();
}

export async function searchCsvData(sessionId: string, column: string, value: string): Promise<{ rows: Record<string, any>[]; count: number }> {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/csv/search?column=${encodeURIComponent(column)}&value=${encodeURIComponent(value)}`);
  if (!res.ok) throw new Error('Failed to search CSV data');
  return res.json();
}

export async function getAllSessionsFromBackend(): Promise<{ sessions: FullSession[]; count: number }> {
  try {
    const res = await fetch(`${API_URL}/sessions`);
    if (!res.ok) throw new Error('Failed to fetch sessions');
    return res.json();
  } catch (e) {
    console.warn('Failed to fetch sessions from backend:', e);
    return { sessions: [], count: 0 };
  }
}

export async function getSessionSummaries(): Promise<{ summaries: SessionSummary[]; count: number }> {
  try {
    const res = await fetch(`${API_URL}/sessions/summaries`);
    if (!res.ok) throw new Error('Failed to fetch session summaries');
    return res.json();
  } catch (e) {
    console.warn('Failed to fetch session summaries:', e);
    return { summaries: [], count: 0 };
  }
}

export async function getSessionFromBackend(sessionId: string): Promise<FullSession | null> {
  try {
    const res = await fetch(`${API_URL}/sessions/${sessionId}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.session;
  } catch (e) {
    console.warn('Failed to fetch session from backend:', e);
    return null;
  }
}

export async function saveSessionToBackend(session: FullSession): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(session),
    });
    if (!res.ok) throw new Error('Failed to save session');
    const data = await res.json();
    console.log('Session saved to backend:', data.message);
    return true;
  } catch (e) {
    console.warn('Failed to save session to backend:', e);
    return false;
  }
}

export async function deleteSessionFromBackend(sessionId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    return res.ok;
  } catch (e) {
    console.warn('Failed to delete session from backend:', e);
    return false;
  }
}

// ============ Report Generation API ============

export type ReportType = 'executive' | 'technical' | 'full_export' | 'shadow_report';
export type ReportFormat = 'markdown' | 'json';

export interface ReportRequest {
  session_id: string;
  report_type: ReportType;
  format: ReportFormat;
  include_code: boolean;
  include_schema: boolean;
  include_chat_history: boolean;
  include_json_data: boolean;
}

export interface GeneratedReport {
  format: string;
  content: string | object;
  filename: string;
  title: string;
}

// Record saved to session history
export interface ReportRecord {
  id: string;
  reportType: string;
  format: string;
  title: string;
  generatedAt: string;
  includeCode: boolean;
  includeSchema: boolean;
  includeChatHistory: boolean;
  includeJsonData: boolean;
  filename: string;
  summary?: string;
  content?: string;  // Full content for viewing later
}

export interface GenerateReportResponse {
  success: boolean;
  report: GeneratedReport;
  reportRecord?: ReportRecord;
}

export async function generateReport(request: ReportRequest): Promise<GenerateReportResponse> {
  const res = await fetch(`${API_URL}/reports/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to generate report');
  return res.json();
}

// ============ Kernel (Developer Mode) API ============

export interface KernelOutput {
  type: 'stream' | 'result' | 'error';
  data?: string;
  name?: string;
  text?: string;
  ename?: string;
  evalue?: string;
  traceback?: string;
}

export interface CodeExecutionResult {
  success: boolean;
  outputs: KernelOutput[];
  error: string | null;
  execution_time_ms: number;
  variables: string[];
}

export async function executeCode(
  sessionId: string,
  code: string,
  timeoutSeconds: number = 30
): Promise<CodeExecutionResult> {
  const res = await fetch(`${API_URL}/kernel/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      code,
      timeout_seconds: timeoutSeconds
    }),
  });
  if (!res.ok) throw new Error('Failed to execute code');
  return res.json();
}

export async function resetKernel(sessionId: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_URL}/kernel/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error('Failed to reset kernel');
  return res.json();
}

export async function getKernelInfo(sessionId: string): Promise<{ exists: boolean; info?: any }> {
  try {
    const res = await fetch(`${API_URL}/kernel/${sessionId}/info`);
    if (!res.ok) return { exists: false };
    return res.json();
  } catch {
    return { exists: false };
  }
}

export async function injectKernelContext(
  sessionId: string,
  mlCode?: string,
  globalJson?: any
): Promise<{ success: boolean }> {
  const res = await fetch(`${API_URL}/kernel/inject-context`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      ml_code: mlCode,
      global_json: globalJson
    }),
  });
  if (!res.ok) throw new Error('Failed to inject context');
  return res.json();
}

export interface FileUploadResult {
  success: boolean;
  message: string;
  preview: string;
  shape?: [number, number];
  columns?: string[];
}

export async function uploadFileToKernel(
  sessionId: string,
  file: File,
  variableName: string = 'uploaded_data'
): Promise<FileUploadResult> {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('file', file);
  formData.append('variable_name', variableName);

  const res = await fetch(`${API_URL}/kernel/upload-file`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to upload file');
  return res.json();
}
