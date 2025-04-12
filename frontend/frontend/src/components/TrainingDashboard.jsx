import { useState } from 'react';
import axios from 'axios';

export default function TrainingDashboard() {
  const [trainingResult, setTrainingResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    const res = await axios.post('http://localhost:8000/api/train/');
    console.log(res.data);
    setTrainingResult(res.data);
    setLoading(false);
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h2>🧠 Client Training Dashboard</h2>
      <button onClick={handleTrain} disabled={loading}>
        {loading ? 'Training...' : 'Start Training'}
      </button>

      {trainingResult && (
        <div style={{ marginTop: '1rem' }}>
          <p>✅ Accuracy: {trainingResult.accuracy}</p>
          <p>📉 Loss: {trainingResult.loss}</p>
        </div>
      )}
    </div>
  );
}
