import React, { useState } from 'react';
import axios from 'axios';
import './CTtoMRI.css';

const CTtoMRI = () => {
    const [ctImage, setCtImage] = useState(null);
    const [mriImage, setMriImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [username, setUsername] = useState('hospital_alpha');

    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setCtImage(e.target.files[0]);
            setError(null);
            setMriImage(null);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!ctImage) {
            setError('Please select a CT image first');
            return;
        }
        
        setLoading(true);
        setError(null);
        
        const formData = new FormData();
        formData.append('username', username);
        formData.append('image', ctImage);

        try {
            const response = await axios.post('http://localhost:8000/api/convert_ct_to_mri/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });

            if (response.data.mri_image) {
                setMriImage(`data:image/png;base64,${response.data.mri_image}`);
            } else {
                setError('No image data received from server');
            }
        } catch (err) {
            console.error('Error converting CT to MRI:', err);
            setError(err.response?.data?.error || 'Failed to convert CT to MRI. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="ct-to-mri">
            <div className="ct-to-mri-header">
                <h1 className="ct-to-mri-title">CT to MRI Conversion</h1>
                <p className="ct-to-mri-subtitle">
                    Transform your CT scans into high-quality MRI-like images using our advanced deep learning model
                </p>
            </div>
            
            <form onSubmit={handleSubmit} className="upload-form">
                <div className="form-header">
                    <div className="form-header-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 12a9 9 0 0 1-9 9 9 9 0 0 1-9-9 9 9 0 0 1 9-9 9 9 0 0 1 9 9z"></path>
                            <path d="M9 10a3 3 0 0 1 3-3 3 3 0 0 1 3 3 3 3 0 0 1-3 3 3 3 0 0 1-3-3z"></path>
                            <path d="M8 21v-1a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v1"></path>
                        </svg>
                    </div>
                    <h2 className="form-title">Upload CT Scan</h2>
                </div>
                
                <div className="form-grid">
                    <div className="form-group">
                        <label className="form-label">Hospital Name</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="form-control"
                            placeholder="Enter your hospital name"
                            required
                        />
                        <p className="form-helper">This helps us track your conversions</p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">CT Scan Image</label>
                        <div className="file-upload-area" onClick={() => document.getElementById('ct-upload').click()}>
                            {ctImage ? (
                                <>
                                    <div className="file-upload-icon">📄</div>
                                    <div className="file-upload-text">{ctImage.name}</div>
                                    <div className="file-upload-hint">Click to change file</div>
                                </>
                            ) : (
                                <>
                                    <div className="file-upload-icon">📁</div>
                                    <div className="file-upload-text">Drag & drop or click to browse</div>
                                    <div className="file-upload-hint">Supports PNG, JPG, DICOM formats (max 10MB)</div>
                                </>
                            )}
                            <input
                                id="ct-upload"
                                type="file"
                                onChange={handleImageChange}
                                className="file-input"
                                accept="image/*"
                                required
                            />
                        </div>
                    </div>

                    <div className="form-actions">
                        <button
                            type="submit"
                            disabled={loading || !ctImage}
                            className={`button ${loading ? 'button-outline' : 'button-primary'}`}
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Processing...
                                </>
                            ) : (
                                'Convert to MRI'
                            )}
                        </button>
                    </div>
                </div>
            </form>

            {error && (
                <div className="error-container">
                    <span className="error-icon">⚠️</span>
                    <span className="error-text">{error}</span>
                </div>
            )}

            {loading && (
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <div className="loading-text">Generating MRI from your CT scan...</div>
                    <p className="text-gray-500 mt-2">This may take a few moments</p>
                </div>
            )}

            {mriImage && (
                <div className="results-section">
                    <div className="results-header">
                        <div className="results-header-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                            </svg>
                        </div>
                        <h3 className="results-title">Conversion Results</h3>
                    </div>
                    
                    <div className="results-grid">
                        <div className="result-card">
                            <div className="result-card-title">Original CT Scan</div>
                            <div className="result-image-container">
                                <img
                                    src={URL.createObjectURL(ctImage)}
                                    alt="Original CT Scan"
                                    className="result-image"
                                />
                            </div>
                        </div>
                        <div className="result-card">
                            <div className="result-card-title">Generated MRI</div>
                            <div className="result-image-container">
                                <img
                                    src={mriImage}
                                    alt="Converted MRI"
                                    className="result-image"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CTtoMRI;