import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './TrainingDashboard.css';

const TrainingProgress = ({ username }) => {
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!username) return;

    const fetchProgress = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/get_training_progress/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ username }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch progress');
        }

        const data = await response.json();
        if (data.status === 'success') {
          setProgress(data.progress);
          setError(null);
        } else {
          setError(data.error || 'Failed to fetch progress');
        }
      } catch (err) {
        setError(err.message);
      }
    };

    // Fetch progress every second
    const interval = setInterval(fetchProgress, 1000);
    return () => clearInterval(interval);
  }, [username]);

  if (!progress) return null;

  const formatTime = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Calculate percentage complete
  const percentComplete = progress.total_epochs > 0 
    ? Math.round((progress.current_epoch / progress.total_epochs) * 100) 
    : 0;

  return (
    <div className="training-progress">
      <div className="progress-header">
        <div className="progress-header-icon">📊</div>
        <h3 className="progress-title">Training Progress</h3>
      </div>
      
      {error ? (
        <div className="error-container">
          <span className="error-icon">⚠️</span>
          <span className="error-text">{error}</span>
        </div>
      ) : (
        <div className="progress-stats">
          <div className="progress-stat">
            <div className="progress-stat-label">Status</div>
            <div className="progress-stat-value capitalize">{progress.status}</div>
          </div>
          <div className="progress-stat">
            <div className="progress-stat-label">Epoch</div>
            <div className="progress-stat-value">{progress.current_epoch} / {progress.total_epochs}</div>
          </div>
          <div className="progress-stat">
            <div className="progress-stat-label">Loss</div>
            <div className="progress-stat-value">{progress.current_loss.toFixed(4)}</div>
          </div>
          <div className="progress-stat">
            <div className="progress-stat-label">Time Elapsed</div>
            <div className="progress-stat-value">{formatTime(progress.time_elapsed)}</div>
          </div>
        </div>
      )}

      <div className="time-remaining-box">
        <div className="time-remaining-label">Estimated Time Remaining</div>
        <div className="time-remaining-value">{formatTime(progress.estimated_time_remaining)}</div>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar-label">
          <span className="progress-bar-label-left">Progress: {percentComplete}%</span>
          <span className="progress-bar-label-right">{progress.current_epoch}/{progress.total_epochs} epochs</span>
        </div>
        <div className="progress-bar">
          <div
            className="progress-bar-fill"
            style={{ width: `${percentComplete}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

const TrainingDashboard = () => {
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, completed, error
  const [trainingStats, setTrainingStats] = useState(null);
  const [error, setError] = useState(null);
  const [username, setUsername] = useState('hospital_alpha');
  const [trainingConfig, setTrainingConfig] = useState({
    dataPath: '/s:/SiDhU/Codes/Major_Project/Federated-MedAI/Dataset/test',
    epochs: 10,
    batchSize: 1,
    learningRate: 0.0002,
    beta1: 0.5
  });

  const getDataPathHelperText = () => {
    return `The data path should contain 'ct' and 'mri' subdirectories with matching image pairs.
           Example: ${trainingConfig.dataPath}`;
  };

  const handleConfigChange = (e) => {
    const { name, value } = e.target;
    setTrainingConfig(prev => ({
      ...prev,
      [name]: name === 'epochs' || name === 'batchSize' ? parseInt(value) : 
              name === 'learningRate' || name === 'beta1' ? parseFloat(value) : value
    }));
  };

  const handleTrain = async () => {
    try {
      setTrainingStatus('training');
      setError(null);
      setTrainingStats(null);
      
      // Create training configuration
      const formData = new FormData();
      formData.append('username', username);
      formData.append('data_path', trainingConfig.dataPath);
      formData.append('epochs', trainingConfig.epochs);
      formData.append('batch_size', trainingConfig.batchSize);
      formData.append('learning_rate', trainingConfig.learningRate);
      formData.append('beta1', trainingConfig.beta1);
      
      // Start training
      const response = await axios.post('http://localhost:8000/api/train/', formData);
      
      if (response.data && response.data.status === 'success') {
        setTrainingStats({
          loss: response.data.loss || 0,
          time_taken: response.data.time_taken || 0
        });
        setTrainingStatus('completed');
      } else {
        throw new Error(response.data?.error || response.data?.message || 'Training failed');
      }
    } catch (err) {
      console.error('Training error:', err);
      // Extract error message from various possible locations in the error object
      const errorMessage = err.response?.data?.error || // Our custom error messages
                          err.response?.data?.message || // Alternative error message field
                          err.message || // Error object's message
                          'Training failed'; // Default message
      setError(errorMessage);
      setTrainingStatus('error');
    }
  };

  return (
    <div className="training-dashboard">
      <div className="dashboard-header">
        <h1 className="dashboard-title">Training Dashboard</h1>
        <p className="dashboard-subtitle">Configure and monitor your model training process</p>
      </div>
      
      <div className="training-form">
        <div className="form-header">
          <div className="form-header-icon">🧠</div>
          <h2 className="form-title">Training Configuration</h2>
        </div>
        
        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Hospital Name</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="form-control"
              placeholder="Enter hospital name"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Data Path</label>
            <input
              type="text"
              name="dataPath"
              value={trainingConfig.dataPath}
              onChange={handleConfigChange}
              className="form-control"
              placeholder="Enter absolute path to data directory"
            />
            <p className="form-helper">{getDataPathHelperText()}</p>
          </div>

          <div className="form-group">
            <label className="form-label">Number of Epochs</label>
            <input
              type="number"
              name="epochs"
              value={trainingConfig.epochs}
              onChange={handleConfigChange}
              min="1"
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Batch Size</label>
            <input
              type="number"
              name="batchSize"
              value={trainingConfig.batchSize}
              onChange={handleConfigChange}
              min="1"
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Learning Rate</label>
            <input
              type="number"
              name="learningRate"
              value={trainingConfig.learningRate}
              onChange={handleConfigChange}
              step="0.0001"
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              Beta 1 (Optimizer Parameter)
              <span className="form-helper">(Adam's β₁ parameter)</span>
            </label>
            <input
              type="number"
              name="beta1"
              value={trainingConfig.beta1}
              onChange={handleConfigChange}
              step="0.1"
              min="0"
              max="1"
              className="form-control"
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            onClick={handleTrain}
            disabled={trainingStatus === 'training'}
            className={`button ${trainingStatus === 'training' ? 'button-outline' : 'button-primary'}`}
          >
            {trainingStatus === 'training' ? 'Training in Progress...' : 'Start Training'}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-container">
          <span className="error-icon">⚠️</span>
          <span className="error-text">{error}</span>
        </div>
      )}

      {trainingStats && trainingStatus === 'completed' && (
        <div className="model-stats">
          <div className="stats-header">
            <div className="stats-header-icon">📈</div>
            <h3 className="stats-title">Training Results</h3>
          </div>
          
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-card-label">Loss</div>
              <div className="stat-card-value">
                {typeof trainingStats.loss === 'number' ? trainingStats.loss.toFixed(4) : 'N/A'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">Time Taken</div>
              <div className="stat-card-value">
                {typeof trainingStats.time_taken === 'number' ? `${trainingStats.time_taken.toFixed(2)}s` : 'N/A'}
              </div>
            </div>
          </div>
        </div>
      )}

      {trainingStatus === 'training' && <TrainingProgress username={username} />}
    </div>
  );
};

export default TrainingDashboard;
