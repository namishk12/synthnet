"use client";

import {
  Archive,
  Binary,
  Boxes,
  BrainCircuit,
  Database,
  GitBranch,
  Home,
  Map,
  Network,
  RadioTower,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

const navigation = [
  { href: "/", label: "Home", icon: Home },
  { href: "/pipeline", label: "Full pipeline", icon: GitBranch },
  { href: "/generator", label: "CGAN generator", icon: Sparkles },
  { href: "/correlator", label: "IPDR correlator", icon: RadioTower },
  { href: "/community", label: "Community intel", icon: Network },
  { href: "/models", label: "Neural network", icon: BrainCircuit },
  { href: "/explorer", label: "India explorer", icon: Map },
  { href: "/datasets", label: "Datasets & runs", icon: Database },
  { href: "/reports", label: "Reports", icon: Archive },
];

export function NexusShell({
  children,
  title,
  eyebrow,
  actions,
}: {
  children: ReactNode;
  title: string;
  eyebrow: string;
  actions?: ReactNode;
}) {
  const pathname = usePathname();
  const [role, setRole] = useState("analyst");
  useEffect(() => {
    const storedRole = window.localStorage.getItem("nexus-role") ?? "analyst";
    queueMicrotask(() => setRole(storedRole));
  }, []);
  function selectRole(value: string) {
    setRole(value);
    window.localStorage.setItem("nexus-role", value);
  }
  return (
    <div className="nexus-app">
      <aside className="nexus-sidebar">
        <Link href="/" className="nexus-logo" aria-label="Nexus India home">
          <span>NI</span>
          <div>
            <strong>NEXUS INDIA</strong>
            <small>Telecom intelligence</small>
          </div>
        </Link>
        <nav aria-label="Primary navigation">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                href={item.href}
                className={active ? "active" : ""}
                key={item.href}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="nexus-sidebar-note">
          <ShieldCheck size={17} />
          <div>
            <strong>Local-first</strong>
            <span>Raw telecom records stay on this machine.</span>
          </div>
        </div>
        <label className="nexus-role-select">
          <span>LOCAL WORKSPACE ROLE</span>
          <select value={role} onChange={(event) => selectRole(event.target.value)}>
            <option value="administrator">Administrator</option>
            <option value="analyst">Analyst</option>
            <option value="reviewer">Reviewer</option>
          </select>
          <small>Role guard for this trusted local session.</small>
        </label>
      </aside>
      <div className="nexus-main">
        <header className="nexus-page-header">
          <div>
            <span className="nexus-eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
          </div>
          <div className="nexus-header-actions">
            {actions}
            <span className="engine-pill">
              <i />
              Local engine
            </span>
          </div>
        </header>
        <main className="nexus-content">{children}</main>
        <footer className="nexus-footer">
          <span><Boxes size={14} /> Synthetic, observed, and inferred records remain explicitly labelled.</span>
          <span><Binary size={14} /> Scores are evidence, not proof of friendship or intent.</span>
        </footer>
      </div>
    </div>
  );
}
