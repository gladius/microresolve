import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';

interface Props {
  title: string;
}

export default function ImportBreadcrumb({ title }: Props) {
  const navigate = useNavigate();
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const domain = settings.selectedDomain;

  return (
    <div className="flex items-center gap-3">
      <button onClick={() => navigate('/import')} className="text-xs text-zinc-500 hover:text-white transition-colors">← Back</button>
      <div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="text-xs text-zinc-500">
          Into:{' '}
          <span className="text-violet-400 font-mono">{ns}</span>
          {domain && (
            <>
              <span className="text-zinc-600 mx-1">/</span>
              <span className="text-violet-400 font-mono">{domain}</span>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
