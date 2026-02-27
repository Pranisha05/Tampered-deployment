import React, { useCallback } from "react";

export default function UploadCard({ file, previewUrl, loading, onPick, onSubmit, onRemove }) {
  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f) onPick({ target: { files: [f] } });
    },
    [onPick]
  );

  const handleDragOver = useCallback((e) => e.preventDefault(), []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!loading && file) onSubmit();
  };

  return (
    <section className="card upload-card" style={{ marginTop: 16 }}>
      <form
        className="upload-form"
        action="/predict"
        method="post"
        encType="multipart/form-data"
        onSubmit={handleSubmit}
      >
        <div
          className="dropzone"
          id="dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          style={{ border: "1px dashed #bbb", borderRadius: 8, textAlign: "center" }}
        >
          <input
            type="file"
            name="image"
            id="image"
            accept="image/*"
            required
            onChange={onPick}
            disabled={loading}
            style={{ display: "block", margin: "0 auto" }}
          />
          <div className="dropzone__content">
            {previewUrl ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <img
                  src={previewUrl}
                  alt="preview"
                  style={{ maxWidth: "100%", borderRadius: 8, marginTop: 8 }}
                />
                <div className="againButton" >
                  <button
                    type="button"
                    onClick={onRemove}
                  >
                    Try another image
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="dropzone__icon" style={{ fontSize: 28, lineHeight: 1 }}>
                  +
                </div>
                <div className="dropzone__text">Drag & drop or click to upload</div>
                <div className="dropzone__hint">JPG, PNG, BMP</div>
              </>
            )}
          </div>
        </div>

        <button className="btn primary" type="submit" disabled={!file || loading} style={{ marginTop: 12 }}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      </form>
    </section>
  );
}
