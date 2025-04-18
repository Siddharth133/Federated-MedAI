import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Stats.css';

const Stats = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/get_metrics/');
                setStats(response.data);
            } catch (err) {
                setError('Failed to load statistics');
                console.error('Error fetching stats:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchStats();
        // Refresh stats every 30 seconds
        const interval = setInterval(fetchStats, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="stats-container">
                <div className="stats-header">
                    <div className="stats-header-icon">📊</div>
                    <h2 className="stats-title">Model Statistics</h2>
                </div>
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <div className="loading-text">Loading statistics...</div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="stats-container">
                <div className="stats-header">
                    <div className="stats-header-icon">📊</div>
                    <h2 className="stats-title">Model Statistics</h2>
                </div>
                <div className="error-container">
                    <span className="error-icon">⚠️</span>
                    <span className="error-text">{error}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="stats-container">
            <div className="stats-header">
                <div className="stats-header-icon">📊</div>
                <h2 className="stats-title">Model Statistics</h2>
            </div>
            
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-card-label">Total Updates</div>
                    <div className="stat-card-value">{stats?.total_updates || 0}</div>
                </div>
                
                <div className="stat-card">
                    <div className="stat-card-label">Inference Count</div>
                    <div className="stat-card-value">{stats?.inference_count || 0}</div>
                </div>
                
                <div className="stat-card">
                    <div className="stat-card-label">Current Version</div>
                    <div className="stat-card-value">{stats?.current_version || 1}</div>
                </div>
            </div>

            <div className="client-stats">
                <div className="stats-header">
                    <div className="stats-header-icon">🏥</div>
                    <h3 className="stats-title">Client Statistics</h3>
                </div>
                
                {stats?.clients && stats.clients.length > 0 ? (
                    <div className="stats-table-container">
                        <table className="stats-table">
                            <thead>
                                <tr>
                                    <th>Hospital</th>
                                    <th>Contributions</th>
                                    <th>Avg. Loss</th>
                                    <th>Avg. Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {stats.clients.map((client, index) => (
                                    <tr key={index}>
                                        <td>{client.username}</td>
                                        <td>{client.contributions}</td>
                                        <td>{client.avg_loss?.toFixed(4) || 'N/A'}</td>
                                        <td>{client.avg_time?.toFixed(2) || 'N/A'}s</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="empty-state">
                        No client statistics available
                    </div>
                )}
            </div>
        </div>
    );
};

export default Stats;

