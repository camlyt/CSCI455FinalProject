import { ShieldCheck, Send, Settings, BarChart3, BookOpen, Info } from 'lucide-react';
import { Page } from '../types';

interface NavigationProps {
  activePage: Page;
  setActivePage: (p: Page) => void;
}

export const Navigation = ({ activePage, setActivePage }: NavigationProps) => (
  <nav className="dashboard-header border-slate-800 bg-slate-950/80">
    <div className="flex items-center gap-2">
      <div className="w-10 h-10 bg-sky-500 rounded-xl flex items-center justify-center text-slate-950">
        <ShieldCheck className="w-6 h-6" />
      </div>
      <div>
        <h1 className="text-xl font-display font-black tracking-tighter text-white">VERIFIED<span className="text-sky-500">.AI</span></h1>
        <span className="text-[10px] font-mono uppercase tracking-widest text-slate-500 block -mt-1 leading-none">Research Demo v2</span>
      </div>
    </div>

    <div className="hidden md:flex items-center bg-slate-900 rounded-full p-1 border border-slate-800">
      {[
        { id: 'verify' as Page, label: 'Verify', icon: Send },
        { id: 'settings' as Page, label: 'Settings', icon: Settings },
        { id: 'analytics' as Page, label: 'Analytics', icon: BarChart3 },
        { id: 'tutorial' as Page, label: 'Tutorial', icon: BookOpen },
      ].map((item) => (
        <button
          key={item.id}
          onClick={() => setActivePage(item.id)}
          className={`flex items-center gap-2 px-6 py-2 rounded-full text-sm font-medium transition-all ${
            activePage === item.id 
              ? 'bg-sky-500 text-slate-950 shadow-lg shadow-sky-500/20' 
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <item.icon className="w-4 h-4" />
          {item.label}
        </button>
      ))}
    </div>

    <div className="flex items-center gap-4">
      <div className="hidden lg:block text-right">
        <span className="text-[10px] font-mono text-slate-500 uppercase block">System Status</span>
        <span className="text-xs font-bold text-emerald-500 flex items-center gap-1 justify-end">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> Optimal
        </span>
      </div>
      <button className="w-10 h-10 rounded-full border flex items-center justify-center border-slate-800 hover:bg-slate-900 transition-colors">
        <Info className="w-5 h-5 text-slate-400" />
      </button>
    </div>
  </nav>
);
