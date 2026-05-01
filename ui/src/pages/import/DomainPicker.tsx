/**
 * DomainPicker — operator-decided domain control for import flows.
 *
 * Modes:
 *   1. Checkbox "Import into a new domain" (default checked) + text input for slug
 *   2. Unchecked → dropdown of existing domains in the namespace
 *   3. If unchecked AND no existing domains: flat-import sub-checkbox
 */
import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api/client';

interface DomainInfo {
  name: string;
  description: string;
  intent_count: number;
}

interface DomainPickerProps {
  /** Initial slug suggestion derived from the import source (e.g. server name, API title). */
  suggestedSlug: string;
  namespaceId: string;
  /**
   * Called whenever the resolved domain changes.
   * - string  → use this domain (empty string = flat/no domain)
   * - null    → Apply should be disabled (no valid selection)
   */
  onChange: (domain: string | null) => void;
}

// ---------------------------------------------------------------------------
// Slug helpers
// ---------------------------------------------------------------------------

function toSlug(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 31);
}

function isValidSlug(s: string): boolean {
  return /^[a-z0-9][a-z0-9-]*$/.test(s);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DomainPicker({ suggestedSlug, namespaceId, onChange }: DomainPickerProps) {
  const [newDomain, setNewDomain] = useState(true); // checkbox state
  const [slug, setSlug] = useState(() => toSlug(suggestedSlug || 'tools'));
  const [existingDomains, setExistingDomains] = useState<DomainInfo[]>([]);
  const [selectedExisting, setSelectedExisting] = useState('');
  const [flat, setFlat] = useState(false); // sub-checkbox for flat import
  const [loadingDomains, setLoadingDomains] = useState(false);

  // Re-derive slug when suggestion changes (e.g. server fetched after mount)
  useEffect(() => {
    const derived = toSlug(suggestedSlug || 'tools');
    setSlug(derived);
  }, [suggestedSlug]);

  // Load existing domains when switching to "existing" mode
  useEffect(() => {
    if (newDomain) return;
    setLoadingDomains(true);
    api.listDomainsFor(namespaceId)
      .then((d: DomainInfo[]) => {
        setExistingDomains(d);
        if (d.length > 0 && !selectedExisting) {
          setSelectedExisting(d[0].name);
        }
      })
      .catch(() => setExistingDomains([]))
      .finally(() => setLoadingDomains(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newDomain, namespaceId]);

  // Notify parent of resolved domain
  const notify = useCallback((
    isNew: boolean,
    currentSlug: string,
    currentExisting: string,
    isFlat: boolean,
    domains: DomainInfo[],
  ) => {
    if (isNew) {
      // null when slug is empty or invalid — disables Apply
      onChange(isValidSlug(currentSlug) ? currentSlug : null);
    } else if (domains.length > 0) {
      onChange(currentExisting || null);
    } else {
      // No existing domains: only allow Apply if flat is explicitly chosen
      onChange(isFlat ? '' : null);
    }
  }, [onChange]);

  useEffect(() => {
    notify(newDomain, slug, selectedExisting, flat, existingDomains);
  }, [newDomain, slug, selectedExisting, flat, existingDomains, notify]);

  const slugError = newDomain && slug.length > 0 && !isValidSlug(slug)
    ? 'Must start with a letter or digit and contain only a–z, 0–9, or hyphens.'
    : null;

  return (
    <div className="border border-zinc-800 rounded-lg px-4 py-3 space-y-3 bg-zinc-900/40">
      {/* Primary checkbox */}
      <label className="flex items-center gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={newDomain}
          onChange={e => {
            setNewDomain(e.target.checked);
            setFlat(false);
          }}
          className="accent-emerald-500"
        />
        <span className="text-xs text-zinc-200 font-medium">Import into a new domain</span>
      </label>

      {/* New-domain slug input */}
      {newDomain && (
        <div className="pl-5 space-y-1">
          <label className="text-[10px] text-zinc-500 uppercase tracking-wide">Domain name</label>
          <input
            value={slug}
            onChange={e => setSlug(toSlug(e.target.value))}
            placeholder="e.g. stripe-payments"
            maxLength={31}
            className={`w-full bg-zinc-800 border rounded px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none ${
              slugError ? 'border-red-700 focus:border-red-500' : 'border-zinc-700 focus:border-emerald-500'
            }`}
          />
          {slugError && <p className="text-[10px] text-red-400">{slugError}</p>}
          {!slugError && slug && (
            <p className="text-[10px] text-zinc-600">
              Intents will be stored as <span className="font-mono text-zinc-400">{slug}:&lt;intent&gt;</span>
            </p>
          )}
        </div>
      )}

      {/* Existing-domain dropdown */}
      {!newDomain && (
        <div className="pl-5 space-y-1">
          {loadingDomains ? (
            <p className="text-[10px] text-zinc-500">Loading domains…</p>
          ) : existingDomains.length > 0 ? (
            <>
              <label className="text-[10px] text-zinc-500 uppercase tracking-wide">Add to existing domain</label>
              <select
                value={selectedExisting}
                onChange={e => setSelectedExisting(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:border-emerald-500"
              >
                {existingDomains.map(d => (
                  <option key={d.name} value={d.name}>{d.name} ({d.intent_count} intents)</option>
                ))}
              </select>
            </>
          ) : (
            <div className="space-y-2">
              <p className="text-[10px] text-zinc-500 italic">
                No existing domains — check the box above to create one.
              </p>
              {/* Flat import sub-checkbox — only when unchecked + no existing domains */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={flat}
                  onChange={e => setFlat(e.target.checked)}
                  className="accent-zinc-400"
                />
                <span className="text-[10px] text-zinc-400">Or import as flat (no domain)</span>
              </label>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
