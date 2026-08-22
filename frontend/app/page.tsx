"use client";

import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const REQUIRED_COUNT = 19;
const FIELD_ORDER = [
  "gender", "Senior_Citizen", "Is_Married", "Dependents", "tenure",
  "Phone_Service", "Dual", "Internet_Service", "Online_Security", "Online_Backup",
  "Device_Protection", "Tech_Support", "Streaming_TV", "Streaming_Movies", "Contract",
  "Paperless_Billing", "Payment_Method", "Monthly_Charges", "Total_Charges",
];

type Message = { id: string; role: "assistant" | "user"; text: string; tone?: "error" | "success" };
type Prediction = {
  prediction: number;
  prediction_label: string;
  churn_probability: number;
  threshold: number;
};
type Explanation = {
  risk_level: "Low" | "Medium" | "High";
  summary: string;
  profile_signals: string[];
  recommended_action: string;
  note: string;
};
type ChatResponse = {
  status: "collecting" | "complete" | "clarification";
  message: string;
  collected_fields: Record<string, string | number>;
  missing_fields: string[];
  completed_count: number;
  required_count: number;
  prediction?: Prediction;
  explanation?: Explanation;
};

const example = "She is not a senior, unmarried, has no dependents, and joined 3 months ago. She has phone service with one line, fiber internet, no online security, no backup, no device protection, no tech support, but uses streaming TV and movies. Her contract is month-to-month, billing is paperless, she pays by electronic check, monthly charges are 95 and total charges are 285.";

const prettyName = (name: string) => name
  .replaceAll("_", " ")
  .replace(/\b\w/g, (letter) => letter.toUpperCase());

const makeSessionId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
};

const initialMessages = (): Message[] => [{
  id: "welcome",
  role: "assistant",
  text: "Tell me about the customer in your own words. I’ll ask for anything the model still needs.",
}];

export default function Home() {
  const [sessionId, setSessionId] = useState(makeSessionId);
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState<"checking" | "ready" | "unavailable">("checking");
  const [collected, setCollected] = useState<Record<string, string | number>>({});
  const [missing, setMissing] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [bulkLoading, setBulkLoading] = useState(false);
  const [fileNotice, setFileNotice] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => setModelStatus(data.ollama?.status === "ok" ? "ready" : "unavailable"))
      .catch(() => setModelStatus("unavailable"));
  }, []);

  useEffect(() => {
    conversationRef.current?.scrollTo({ top: conversationRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const submitMessage = async (event?: FormEvent) => {
    event?.preventDefault();
    const message = input.trim();
    if (!message || loading || !sessionId) return;

    setMessages((items) => [...items, { id: crypto.randomUUID(), role: "user", text: message }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "The assistant could not process that message.");

      const result = data as ChatResponse;
      setCollected(result.collected_fields);
      setMissing(result.missing_fields);
      setPrediction(result.prediction ?? null);
      setExplanation(result.explanation ?? null);
      setMessages((items) => [...items, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: result.message,
        tone: result.status === "complete" ? "success" : undefined,
      }]);
    } catch (error) {
      const errorMessage = error instanceof TypeError && error.message === "Failed to fetch"
        ? "I can’t reach the prediction API. Start the FastAPI server on port 8000, then try again."
        : error instanceof Error
          ? error.message
          : "The local assistant is unavailable.";
      setMessages((items) => [...items, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: errorMessage,
        tone: "error",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    if (sessionId) fetch(`${API_URL}/chat/${sessionId}`, { method: "DELETE" }).catch(() => undefined);
    setSessionId(makeSessionId());
    setMessages(initialMessages());
    setCollected({});
    setMissing([]);
    setPrediction(null);
    setExplanation(null);
    setInput("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  };

  const completedCount = Object.keys(collected).length;
  const progress = Math.round((completedCount / REQUIRED_COUNT) * 100);
  const modelCopy = modelStatus === "ready" ? "Local AI · Ready" : modelStatus === "checking" ? "Checking local AI" : "Local AI unavailable";

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const generateReport = async () => {
    if (!prediction || reportLoading) return;
    setReportLoading(true);
    setFileNotice("Preparing customer assessment report...");
    try {
      const response = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collected),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail ?? "The report could not be generated.");
      }
      downloadBlob(await response.blob(), "churnsignal-customer-assessment.pdf");
      setFileNotice("Professional assessment report downloaded.");
    } catch (error) {
      setFileNotice(error instanceof Error ? error.message : "The report could not be generated.");
    } finally {
      setReportLoading(false);
    }
  };

  const scoreFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || bulkLoading) return;
    setBulkLoading(true);
    setFileNotice(`Scoring ${file.name} locally...`);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_URL}/bulk-predict`, { method: "POST", body: form });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail ?? "The customer file could not be scored.");
      }
      const total = response.headers.get("X-Total-Rows") ?? "0";
      const scored = response.headers.get("X-Scored-Rows") ?? "0";
      const invalid = response.headers.get("X-Invalid-Rows") ?? "0";
      const churnRate = Number(response.headers.get("X-Churn-Rate") ?? 0);
      downloadBlob(await response.blob(), "churnsignal-assessment-package.zip");
      setFileNotice(
        `${scored} of ${total} customers scored. Predicted churn rate: ${(churnRate * 100).toFixed(1)}%. Downloaded Excel copy + PDF findings. ${invalid} row${invalid === "1" ? "" : "s"} need correction.`,
      );
    } catch (error) {
      setFileNotice(error instanceof Error ? error.message : "The customer file could not be scored.");
    } finally {
      setBulkLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">C</span><span>ChurnSignal</span></div>
        <div className="top-actions">
          <input ref={fileInputRef} className="sr-only" type="file" accept=".csv,.xlsx" onChange={scoreFile} />
          <button className="bulk-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={bulkLoading}>
            {bulkLoading ? "Scoring file..." : "Score CSV / Excel"}
          </button>
          <div className={`runtime-pill ${modelStatus}`}><span className="status-dot" /> {modelCopy}</div>
        </div>
      </header>
      {fileNotice && <div className="file-notice" role="status"><span>{fileNotice}</span><button type="button" onClick={() => setFileNotice("")} aria-label="Dismiss notification">Close</button></div>}

      <section className="workspace">
        <aside className="profile-panel">
          <div>
            <p className="eyebrow">Customer profile</p>
            <h1>Build a churn picture through conversation.</h1>
            <p className="muted">Share what you know. The assistant collects the remaining details before scoring.</p>
          </div>
          <div className="progress-card">
            <div className="progress-copy"><span>Profile readiness</span><strong>{completedCount} / {REQUIRED_COUNT}</strong></div>
            <div className="progress-track" aria-label={`${progress}% profile complete`}><span style={{ width: `${Math.max(progress, 2)}%` }} /></div>
          </div>
          <div className="field-preview">
            {FIELD_ORDER.map((name) => (
              <div className={`field-row ${name in collected ? "complete" : "pending"}`} key={name}>
                <span>{prettyName(name)}</span><strong>{name in collected ? String(collected[name]) : "Pending"}</strong>
              </div>
            ))}
          </div>
          <div className="privacy-note"><span className="lock-icon">●</span><div><strong>Stays on this device</strong><p>Conversation and inference run locally.</p></div></div>
        </aside>

        <section className="chat-panel">
          <div className="chat-heading">
            <div><p className="eyebrow">Retention copilot</p><h2>{prediction ? "Assessment complete" : "New assessment"}</h2></div>
            <button className="secondary-button" type="button" onClick={reset}>Reset</button>
          </div>
          <div className="conversation" ref={conversationRef} aria-live="polite">
            {messages.map((message) => (
              <article className={`message ${message.role}-message ${message.tone ?? ""}`} key={message.id}>
                {message.role === "assistant" && <div className="avatar">AI</div>}
                <div className="message-bubble">
                  <p className="message-label">{message.role === "assistant" ? "ChurnSignal" : "You"}</p>
                  <p>{message.text}</p>
                </div>
              </article>
            ))}
            {messages.length === 1 && (
              <button className="prompt-card" type="button" onClick={() => setInput(example)}>
                <span>Use a complete example</span>
                Fill the composer with a high-risk customer profile
              </button>
            )}
            {loading && <article className="message assistant-message"><div className="avatar">AI</div><div className="typing" aria-label="Assistant is thinking"><span /><span /><span /></div></article>}
          </div>
          <form className="composer" onSubmit={submitMessage}>
            <label htmlFor="message" className="sr-only">Customer information</label>
            <textarea id="message" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleKeyDown} placeholder="Describe the customer…" rows={2} disabled={loading} />
            <button type="submit" aria-label="Send message" disabled={loading || !input.trim()}>{loading ? "Working" : "Send"} <span>↗</span></button>
          </form>
          {!!missing.length && <p className="missing-caption">Still needed: {missing.length} field{missing.length === 1 ? "" : "s"}</p>}
        </section>

        <aside className={`result-panel ${prediction ? "has-result" : ""}`}>
          <p className="eyebrow">Live assessment</p>
          {!prediction || !explanation ? (
            <div className="empty-result"><div className="pulse-rings"><span /></div><h3>Awaiting profile</h3><p>The churn score and recommended action will appear here when the profile is complete.</p></div>
          ) : (
            <div className="result-content">
              <div className={`risk-tag risk-${explanation.risk_level.toLowerCase()}`}>{explanation.risk_level} risk</div>
              <div className="score-ring" style={{ "--score": `${prediction.churn_probability * 360}deg` } as React.CSSProperties}>
                <div><strong>{Math.round(prediction.churn_probability * 100)}%</strong><span>churn probability</span></div>
              </div>
              <div className="threshold-line"><span>Intervention threshold</span><strong>{Math.round(prediction.threshold * 100)}%</strong></div>
              <div className="signals"><p className="result-label">Profile signals</p>{explanation.profile_signals.map((signal) => <div className="signal" key={signal}><span>↗</span>{signal}</div>)}</div>
              <div className="action-card"><p className="result-label">Recommended action</p><strong>{explanation.recommended_action}</strong></div>
              <button className="report-button" type="button" onClick={generateReport} disabled={reportLoading}>
                {reportLoading ? "Generating report..." : "Download professional report"}
              </button>
              <p className="result-note">{explanation.note}</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
