import React, { useState } from 'react';
import CTtoMRI from './components/CTtoMRI';
import Stats from './components/Stats';
import TrainingDashboard from './components/TrainingDashboard';
import UploadWeights from './components/UploadWeights';
import './App.css';

const App = () => {
    const [activeTab, setActiveTab] = useState('train');

    return (
        <div className="app-container">
            {/* Enhanced Header with Improved Navigation */}
            <header className="header">
                <div className="header-content">
                    <a href="/" className="logo">
                        <svg className="logo-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="currentColor"/>
                            <path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z" fill="currentColor"/>
                        </svg>
                        <span className="logo-text">Federated Learning System</span>
                    </a>
                    
                    <nav className="nav-links">
                        <button
                            onClick={() => setActiveTab('train')}
                            className={`nav-link ${activeTab === 'train' ? 'active' : ''}`}
                        >
                            <svg viewBox="0 0 24 24" fill="none">
                                <path d="M20 14V18M4 14V18M16 8V6C16 4.89543 15.1046 4 14 4H10C8.89543 4 8 4.89543 8 6V8M5 8H19C20.1046 8 21 8.89543 21 10V20C21 21.1046 20.1046 22 19 22H5C3.89543 22 3 21.1046 3 20V10C3 8.89543 3.89543 8 5 8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                            Training
                        </button>
                        <button
                            onClick={() => setActiveTab('convert')}
                            className={`nav-link ${activeTab === 'convert' ? 'active' : ''}`}
                        >
                            <svg viewBox="0 0 24 24" fill="none">
                                <path d="M8 16V8M16 8V16M3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                            CT to MRI
                        </button>
                        <button
                            onClick={() => setActiveTab('upload')}
                            className={`nav-link ${activeTab === 'upload' ? 'active' : ''}`}
                        >
                            <svg viewBox="0 0 24 24" fill="none">
                                <path d="M7 16C4.79086 16 3 14.2091 3 12C3 9.79086 4.79086 8 7 8M7 16C9.20914 16 11 14.2091 11 12C11 9.79086 9.20914 8 7 8M7 16L17 16M17 16C19.2091 16 21 14.2091 21 12C21 9.79086 19.2091 8 17 8M17 16C14.7909 16 13 14.2091 13 12C13 9.79086 14.7909 8 17 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                            Upload Weights
                        </button>
                    </nav>
                </div>
            </header>

            <main className="main-content">
                {/* Tab Content with Smooth Transitions */}
                <div className="tab-content">
                    {activeTab === 'train' && (
                        <div className="fade-in">
                            <TrainingDashboard />
                        </div>
                    )}
                    {activeTab === 'convert' && (
                        <div className="fade-in">
                            <CTtoMRI />
                        </div>
                    )}
                    {activeTab === 'upload' && (
                        <div className="fade-in">
                            <UploadWeights />
                        </div>
                    )}
                </div>

                {/* Stats Section with Improved Layout */}
                <div className="stats-section">
                    <h2 className="section-title">
        
                    </h2>
                    <Stats />
                </div>
            </main>

            {/* Enhanced Footer with Better Structure */}
            <footer className="footer">
                <div className="footer-content">
                    <p className="footer-text">
                        © {new Date().getFullYear()} Federated Learning System. All rights reserved.
                    </p>
                    <div className="footer-links">
                        <a href="/privacy" className="footer-link">Privacy Policy</a>
                        <a href="/terms" className="footer-link">Terms of Service</a>
                        <a href="/contact" className="footer-link">Contact Us</a>
                        <a href="/docs" className="footer-link">Documentation</a>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default App;