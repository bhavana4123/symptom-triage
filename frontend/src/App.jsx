import { useState } from "react"
import SymptomForm from "./components/SymptomForm"
import TriageResult from "./components/TriageResult"

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col items-center px-4 py-10">

      {/* header */}
      <header className="text-center mb-6">
        <h1 className="text-3xl font-bold text-slate-800 tracking-tight">
          Symptom Triage Assistant
        </h1>
        <p className="text-slate-500 mt-1 text-sm">
          AI-powered preliminary symptom assessment
        </p>
      </header>

      {/* disclaimer banner */}
      <div className="w-full max-w-lg bg-amber-50 border border-amber-300 rounded-xl px-4 py-2.5 mb-6 flex items-center gap-2">
        <span className="text-amber-500 text-base leading-none">⚠</span>
        <p className="text-xs text-amber-800 font-medium">
          This tool is not a substitute for professional medical advice
        </p>
      </div>

      {/* main content: form or result */}
      {result === null ? (
        <SymptomForm onResult={setResult} />
      ) : (
        <TriageResult result={result} onReset={() => setResult(null)} />
      )}

    </div>
  )
}
