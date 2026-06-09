const URGENCY_CONFIG = {
  emergency: {
    badge: "bg-red-600 text-white",
    card: "border-red-200 bg-red-50",
    bar: "bg-red-500",
    pill: "bg-red-100 text-red-700 border-red-200",
    label: "Emergency",
  },
  urgent: {
    badge: "bg-orange-500 text-white",
    card: "border-orange-200 bg-orange-50",
    bar: "bg-orange-500",
    pill: "bg-orange-100 text-orange-700 border-orange-200",
    label: "Urgent",
  },
  routine: {
    badge: "bg-green-600 text-white",
    card: "border-green-200 bg-green-50",
    bar: "bg-green-500",
    pill: "bg-green-100 text-green-700 border-green-200",
    label: "Routine",
  },
}

const FALLBACK_CONFIG = {
  badge: "bg-slate-500 text-white",
  card: "border-slate-200 bg-slate-50",
  bar: "bg-slate-400",
  pill: "bg-slate-100 text-slate-700 border-slate-200",
  label: "Unknown",
}

export default function TriageResult({ result, onReset }) {
  const cfg = URGENCY_CONFIG[result.urgency_label] ?? FALLBACK_CONFIG
  const confidencePct = Math.round(result.confidence * 100)

  return (
    <div
      className={`w-full max-w-lg rounded-2xl shadow-sm border-2 p-6 space-y-5 ${cfg.card}`}
    >
      {/* header row: title + urgency badge */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-800">Triage Result</h2>
        <span
          className={`text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide ${cfg.badge}`}
        >
          {cfg.label}
        </span>
      </div>

      {/* confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1.5">
          <span className="font-medium">Model Confidence</span>
          <span className="font-semibold text-slate-700">{confidencePct}%</span>
        </div>
        <div className="w-full bg-white/60 rounded-full h-2.5 border border-white/80 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ${cfg.bar}`}
            style={{ width: `${confidencePct}%` }}
            role="progressbar"
            aria-valuenow={confidencePct}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      </div>

      {/* explanation */}
      <p className="text-sm text-slate-700 leading-relaxed">{result.explanation}</p>

      {/* possible conditions */}
      {result.possible_conditions?.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-2">
            Possible Conditions
          </h3>
          <div className="flex flex-wrap gap-2">
            {result.possible_conditions.map((condition) => (
              <span
                key={condition}
                className={`text-xs font-medium px-3 py-1 rounded-full border ${cfg.pill}`}
              >
                {condition}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* what to do next */}
      {result.next_steps?.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-2">
            What To Do Next
          </h3>
          <ol className="space-y-2">
            {result.next_steps.map((step, i) => (
              <li key={i} className="flex gap-3 text-sm text-slate-700">
                <span
                  className={`shrink-0 w-6 h-6 rounded-full flex items-center justify-center
                               text-xs font-bold ${cfg.badge}`}
                >
                  {i + 1}
                </span>
                <span className="pt-0.5">{step}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* disclaimer */}
      <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-200/70 pt-4">
        {result.disclaimer}
      </p>

      {/* check again */}
      <button
        onClick={onReset}
        className="w-full border-2 border-slate-300 hover:border-slate-400 hover:bg-white
                   text-slate-600 hover:text-slate-800 font-semibold py-2.5 rounded-xl
                   transition-colors text-sm"
      >
        Check Again
      </button>
    </div>
  )
}
