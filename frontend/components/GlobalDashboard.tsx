"use client";
import { motion } from 'framer-motion';
import { BarChart, Activity, AlertTriangle } from 'lucide-react';

export default function GlobalDashboard({ globalData }: { globalData: any }) {
    if (!globalData) return null;

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in duration-500">
            {/* Feature Importance Card */}
            <div className="bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl">
                <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
                    <BarChart className="w-5 h-5 text-emerald-400" /> Global Feature Importance
                </h3>
                <div className="space-y-3">
                    {globalData.global_importance.map((item: any, idx: number) => (
                        <div key={idx} className="relative">
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-300 font-medium">{item.feature_or_group}</span>
                                <span className="text-slate-400">{(item.importance * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${item.importance * 100}%` }}
                                    transition={{ delay: idx * 0.1, duration: 0.8 }}
                                    className={`h-full rounded-full ${item.direction === 'positive' ? 'bg-emerald-500' : 'bg-rose-500'
                                        }`}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Reliability & Limits Card */}
            <div className="bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl">
                <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-amber-400" /> Model Reliability
                </h3>

                <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="bg-slate-800/50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-slate-100">{globalData.reliability.sample_size}</div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Samples Modeled</div>
                    </div>
                    <div className="bg-slate-800/50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-slate-100 flex justify-center items-center gap-1">
                            {globalData.reliability.stability_score}
                        </div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Stability Score</div>
                    </div>
                </div>

                {globalData.limits.length > 0 && (
                    <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-4">
                        <h4 className="text-amber-400 text-sm font-semibold mb-2 flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" /> Known Limitations
                        </h4>
                        <ul className="text-sm text-amber-200/80 space-y-1 list-disc pl-4">
                            {globalData.limits.map((limit: string, i: number) => (
                                <li key={i}>{limit}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>

            {/* Trends (Placeholder for Chart) */}
            <div className="col-span-1 lg:col-span-2 bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl">
                <h3 className="text-lg font-semibold text-slate-100 mb-4">Top Feature Trends</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {globalData.global_trends.map((trend: any, idx: number) => (
                        <div key={idx} className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">
                            <div className="text-sm text-slate-400 mb-2">{trend.feature}</div>
                            <div className="h-24 flex items-end gap-1">
                                {/* Mock bars */}
                                {trend.numeric_trends && trend.numeric_trends.map((bin: any, bIdx: number) => (
                                    <div key={bIdx} className="flex-1 bg-blue-500/40 hover:bg-blue-400/60 transition-colors rounded-t-sm relative group" style={{ height: `${bin.avg_score * 100}%` }}>
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-10">
                                            Score: {bin.avg_score} <br /> Range: {bin.bin_start.toFixed(0)}-{bin.bin_end.toFixed(0)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
