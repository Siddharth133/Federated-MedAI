import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';
import TrainingDashboard from './components/TrainingDashboard';

function App() {
  const [view, setView] = useState('upload');

  return (
    <div>
      <h1 style={{ textAlign: 'center' }}>Federated MedAI Client</h1>

      <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
        <button onClick={() => setView('upload')} style={{ marginRight: '1rem' }}>
          Upload
        </button>
        <button onClick={() => setView('dashboard')} style={{ marginRight: '1rem' }}>
          Dashboard
        </button>
        <button onClick={() => setView('training')}>
          Training Dashboard
        </button>
      </div>

      {view === 'upload' && <FileUpload />}
      {view === 'dashboard' && <Dashboard />}
      {view === 'training' && <TrainingDashboard />}
    </div>
  );
}

export default App;
