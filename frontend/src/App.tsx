/**
 * Main application component with React Router configuration.
 */

import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import CandidateExam from "./pages/CandidateExam";
import ProctorDashboard from "./pages/ProctorDashboard";
import ReportView from "./pages/ReportView";
import AdminPanel from "./pages/AdminPanel";

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/exam" element={<CandidateExam />} />
            <Route path="/dashboard" element={<ProctorDashboard />} />
            <Route path="/reports" element={<ReportView />} />
            <Route path="/admin" element={<AdminPanel />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <footer className="footer">
          <p>
            BE Final-Year Project: Multimodal Cheating Detection System &bull; Designed as decision-support for human proctors.
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;

