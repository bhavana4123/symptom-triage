import { useState } from "react"
import axios from "axios"

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12" cy="12" r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

export default function SymptomForm({ onResult }) {
  const [symptoms, setSymptoms] = useState("")
  const [durationDays, setDurationDays] = useState("")
  const [patientAge, setPatientAge] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const { data } = await axios.post("/api/triage", {
        symptoms,
        duration_days: parseInt(durationDays, 10),
        patient_age: parseInt(patientAge, 10),
      })
      onResult(data)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
          ? detail.map((d) => d.msg).join("; ")
          : "Request failed — is the API server running on port 8000?"
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-lg bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-5"
    >
      {/* symptoms textarea */}
      <div>
        <label
          htmlFor="symptoms"
          className="block text-sm font-medium text-slate-700 mb-1.5"
        >
          Symptoms <span className="text-red-500">*</span>
        </label>
        <textarea
          id="symptoms"
          className="w-full border border-slate-300 rounded-xl p-3 text-sm text-slate-800
                     placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500
                     focus:border-transparent resize-none transition"
          rows={4}
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder="Describe your symptoms e.g. fever, cough, fatigue"
          required
          minLength={3}
        />
      </div>

      {/* duration + age row */}
      <div className="flex gap-4">
        <div className="flex-1">
          <label
            htmlFor="duration"
            className="block text-sm font-medium text-slate-700 mb-1.5"
          >
            Duration (days) <span className="text-red-500">*</span>
          </label>
          <input
            id="duration"
            type="number"
            className="w-full border border-slate-300 rounded-xl p-3 text-sm text-slate-800
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       transition"
            value={durationDays}
            onChange={(e) => setDurationDays(e.target.value)}
            min={1}
            max={30}
            required
            placeholder="1 – 30"
          />
        </div>

        <div className="flex-1">
          <label
            htmlFor="age"
            className="block text-sm font-medium text-slate-700 mb-1.5"
          >
            Patient Age <span className="text-red-500">*</span>
          </label>
          <input
            id="age"
            type="number"
            className="w-full border border-slate-300 rounded-xl p-3 text-sm text-slate-800
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       transition"
            value={patientAge}
            onChange={(e) => setPatientAge(e.target.value)}
            min={1}
            max={100}
            required
            placeholder="1 – 100"
          />
        </div>
      </div>

      {/* error banner */}
      {error && (
        <div
          role="alert"
          className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700
                     rounded-xl px-4 py-3 text-sm"
        >
          <span className="mt-0.5 shrink-0">✕</span>
          <span>{error}</span>
        </div>
      )}

      {/* submit */}
      <button
        type="submit"
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700
                   active:bg-blue-800 disabled:opacity-60 text-white font-semibold py-3 rounded-xl
                   transition-colors"
      >
        {loading ? (
          <>
            <Spinner />
            <span>Analyzing…</span>
          </>
        ) : (
          "Analyze Symptoms"
        )}
      </button>
    </form>
  )
}
