import React, { useState, useEffect } from "react";
import { predictImage } from "../api/client";
import UploadCard from "../components/UploadCard";
import ResultPanel from "../components/ResultPanel";

export default function Home() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const onPick = (e) => {
    const f = e.target.files?.[0];
    // revoke previous preview URL
    if (!f && previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (f && previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(f || null);
    setResult(null);
    setErr(null);
    if (f) setPreviewUrl(URL.createObjectURL(f));
    else setPreviewUrl(null);
  };

  const handleRemove = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErr(null);
  };

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setErr(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: 16 }}>
      <UploadCard 
        file={file}
        previewUrl={previewUrl}
        loading={loading}
        onPick={onPick}
        onSubmit={onSubmit}
        onRemove={handleRemove}
      />

      {err && <p style={{ color: "crimson", marginTop: 16 }}><strong>Error:</strong> {err}</p>}

      {result && <ResultPanel result={result} previewUrl={previewUrl} />}
    </div>
  );
}
