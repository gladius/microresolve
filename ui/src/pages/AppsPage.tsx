import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, setApiAppId } from '@/api/client';

interface AppInfo {
  id: string;
  intents: number;
}

export default function AppsPage() {
  const navigate = useNavigate();
  const [apps, setApps] = useState<AppInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const appIds = await api.listApps();
      const infos: AppInfo[] = [];
      for (const id of appIds) {
        // Temporarily switch to each app to get intent count
        setApiAppId(id);
        try {
          const intents = await api.listIntents();
          infos.push({ id, intents: intents.length });
        } catch {
          infos.push({ id, intents: 0 });
        }
      }
      // Restore current app
      const current = localStorage.getItem('asv_app_id') || 'default';
      setApiAppId(current);
      setApps(infos);
    } catch { /* */ }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const switchToApp = (appId: string) => {
    setApiAppId(appId);
    localStorage.setItem('asv_app_id', appId);
    window.location.href = '/intents';
  };

  const deleteApp = async (appId: string) => {
    if (appId === 'default') return;
    if (!confirm(`Delete app "${appId}" and all its intents?`)) return;
    try {
      await api.deleteApp(appId);
      refresh();
    } catch (e) {
      alert('Delete failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
  };

  const currentApp = localStorage.getItem('asv_app_id') || 'default';

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white mb-1">Apps</h2>
          <p className="text-xs text-zinc-500">Each app is an isolated routing workspace with its own intents and seeds.</p>
        </div>
        <button
          onClick={() => navigate('/import')}
          className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500"
        >
          Import API Spec
        </button>
      </div>

      {loading ? (
        <div className="text-xs text-zinc-500 text-center py-8">Loading apps...</div>
      ) : (
        <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50">
          {apps.map(app => (
            <div key={app.id} className={`flex items-center gap-4 px-4 py-3 ${app.id === currentApp ? 'bg-violet-500/5' : 'hover:bg-zinc-800/40'}`}>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-white font-mono">{app.id}</span>
                  {app.id === currentApp && (
                    <span className="text-[9px] text-violet-400 bg-violet-500/20 px-1.5 py-0.5 rounded">active</span>
                  )}
                </div>
                <div className="text-xs text-zinc-500 mt-0.5">
                  {app.intents} {app.intents === 1 ? 'intent' : 'intents'}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => switchToApp(app.id)}
                  className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-white hover:border-violet-500"
                >
                  {app.id === currentApp ? 'View Intents' : 'Switch'}
                </button>
                {app.id !== 'default' && (
                  <button
                    onClick={() => deleteApp(app.id)}
                    className="text-xs px-2 py-1.5 text-zinc-600 hover:text-red-400"
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
