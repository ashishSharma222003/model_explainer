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

export async function chat(request: { session_id: string; message: string; context?: any }) {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to send chat message');
  return res.json();
}

export async function guideCode(session_id: string, code: string) {
  const res = await fetch(`${API_URL}/guide-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, code }),
  });
  if (!res.ok) throw new Error('Failed to get code guidance');
  return res.json();
}
