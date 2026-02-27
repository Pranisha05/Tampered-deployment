import React from "react";
import Home from "./pages/Home";
import "../styles.css";

export default function App() {
  return (
    <div className="page">
      <header className="hero">
        <div className="hero__badge">Minor Project Deployment</div>
        <h1 className="hero__title">TamperSight</h1>
        <p className="hero__subtitle">
          Upload a raw image. We generate ELA and noise map, then classify authenticity and highlight suspected tamper regions with Grad-CAM.
        </p>
      </header>

      <Home />
    </div>
  );
}
