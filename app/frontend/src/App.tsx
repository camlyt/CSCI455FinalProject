import { useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Navigation } from './components/Navigation';
import { VerifyPage } from './components/VerifyPage';
import { SettingsPage } from './components/SettingsPage';
import { AnalyticsPage } from './components/AnalyticsPage';
import { TutorialPage } from './components/TutorialPage';
import { Page, AppSettings } from './types';
import { verifyClaim } from "./api";

export default function App() {
  const [activePage, setActivePage] = useState<Page>('verify');
  const [isVerifying, setIsVerifying] = useState(false);
  const [showResult, setShowResult] = useState(false);

  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Interactive Settings State
  const [settings, setSettings] = useState<AppSettings>({
    topK: 5,
    threshold: 0.8,
    retriever: 'minilm'
  });

  const updateSettings = (newSettings: Partial<AppSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  const handleVerify = async (claim: string) => {
    setIsVerifying(true);
    setShowResult(false);
    setError(null);

    try {
        const result = await verifyClaim(claim, settings);
        setVerificationResult(result);
        setShowResult(true);
    } catch (err) {
        setError("Could not verify claim. Make sure the backend is running.");
    } finally {
        setIsVerifying(false);
    }
  };

  return (
    <div className="app-shell bg-slate-950 text-slate-50 min-h-screen font-sans">
      <Navigation activePage={activePage} setActivePage={setActivePage} />
      
      <main className="pt-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activePage}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.02 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            {activePage === 'verify' && (
              <VerifyPage 
                isVerifying={isVerifying} 
                handleVerify={handleVerify} 
                showResult={showResult}
                settings={settings}
                result={verificationResult}
                error={error}
              />
            )}
            {activePage === 'settings' && (
              <SettingsPage 
                settings={settings} 
                updateSettings={updateSettings} 
              />
            )}
            {activePage === 'analytics' && (
              <AnalyticsPage />
            )}
            {activePage === 'tutorial' && (
              <TutorialPage />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="fixed bottom-0 left-0 right-0 py-4 px-8 bg-slate-950/80 backdrop-blur-md border-t border-slate-800 flex justify-between items-center text-[10px] font-mono text-slate-500 uppercase tracking-widest pointer-events-none z-50">
        <span>© 2026 FACT LAB</span>
        <div className="flex gap-8">
           <span className="flex items-center gap-1">
            <span className="text-sky-500">K:</span> {settings.topK}
          </span>
          <span className="flex items-center gap-1">
            <span className="text-sky-500">T:</span> {settings.threshold}
          </span>
          <span>Model: DeBERTa-v3-NLI</span>
          <span>Latency: 24ms</span>
        </div>
      </footer>
    </div>
  );
}
