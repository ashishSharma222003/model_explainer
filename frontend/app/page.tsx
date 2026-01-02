"use client";
import { useState } from 'react';
import SetupPanel from '@/components/SetupPanel';
import GlobalDashboard from '@/components/GlobalDashboard';
import TransactionExplainer from '@/components/TransactionExplainer';
import { Bot, Layers } from 'lucide-react';

export default function Home() {
  const [step, setStep] = useState<'setup' | 'dashboard'>('setup');
  const [specs, setSpecs] = useState<any>(null);
  const [globalData, setGlobalData] = useState<any>(null);
  const [complianceMode, setComplianceMode] = useState(false);

  const handleSetupComplete = (submitted: any) => {
    // Directly use the JSON provided by user
    setSpecs({ model_version: submitted.model_version });
    setGlobalData(submitted.global_json);
    setStep('dashboard');
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Background Ambience */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black -z-10" />
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 -z-10 mix-blend-overlay"></div>

      {/* Header */}
      <header className="border-b border-slate-800/50 bg-slate-950/50 backdrop-blur top-0 z-50 sticky">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-blue-500 to-cyan-400 p-2 rounded-lg">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              Antigravity <span className="font-light text-slate-500">Model Explainer</span>
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-xs font-mono text-slate-500">POC v0.1</div>
            <button
              onClick={() => setComplianceMode(!complianceMode)}
              className={`px-3 py-1 rounded-full border text-xs font-medium flex items-center gap-2 transition-all ${complianceMode
                  ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                }`}
            >
              <Bot className="w-3 h-3" /> Compliance Mode: {complianceMode ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {step === 'setup' && (
          <div className="max-w-3xl mx-auto mt-12">
            <div className="text-center mb-10 space-y-2">
              <h2 className="text-4xl font-bold text-white">Initialize Explanation Engine</h2>
              <p className="text-slate-400">Paste your generated Global JSON to visualize the model.</p>
            </div>
            <SetupPanel onComplete={handleSetupComplete} />
          </div>
        )}

        {step === 'dashboard' && globalData && (
          <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-700">
            {/* Global View */}
            <section>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-white">Global Behavior</h2>
                <span className="text-sm text-slate-400">Model: {globalData.model_version}</span>
              </div>
              <GlobalDashboard globalData={globalData} />
            </section>

            {/* Transaction View */}
            <section className="pt-8 border-t border-slate-800/50">
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-white">Transaction Drilldown</h2>
                <p className="text-slate-400 text-sm">Analyze individual decisions. Paste Transaction JSON to drill down.</p>
              </div>
              <TransactionExplainer
                modelSpecs={specs}
                globalContext={globalData}
                complianceMode={complianceMode}
              />
            </section>
          </div>
        )}
      </div>
    </main>
  );
}
