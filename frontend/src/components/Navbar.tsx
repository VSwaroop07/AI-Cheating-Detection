/**
 * Top navigation bar component linking all routes.
 */

import React from "react";
import { NavLink } from "react-router-dom";
import { HealthBadge } from "./HealthBadge";

export const Navbar: React.FC = () => {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <NavLink to="/" className="brand-link">
          <span className="brand-icon">🛡️</span>
          <span className="brand-text">Cheating Detection Proctor</span>
        </NavLink>
      </div>

      <nav className="navbar-links">
        <NavLink to="/login" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Login
        </NavLink>
        <NavLink to="/register" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Register
        </NavLink>
        <NavLink to="/exam" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Candidate Exam
        </NavLink>
        <NavLink to="/dashboard" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Proctor Dashboard
        </NavLink>
        <NavLink to="/reports" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Report View
        </NavLink>
        <NavLink to="/admin" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
          Admin Panel
        </NavLink>
      </nav>

      <div className="navbar-status">
        <HealthBadge />
      </div>
    </header>
  );
};

