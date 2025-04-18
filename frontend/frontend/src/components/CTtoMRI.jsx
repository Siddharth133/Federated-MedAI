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
        setCtImage(e.target.files[0]);
        setError(null);
        setMriImage(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
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
            setError(err.response?.data?.error || 'Failed to convert CT to MRI');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="ct-to-mri">
            <div className="ct-to-mri-header">
                <h1 className="ct-to-mri-title">Convert CT to MRI</h1>
                <p className="ct-to-mri-subtitle">Transform CT scans into MRI-like images using our AI model</p>
            </div>
            
            <div className="upload-form">
                <div className="form-header">
                    <div className="form-header-icon">🔄</div>
                    <h2 className="form-title">Image Conversion</h2>
                </div>
                
                <form onSubmit={handleSubmit} className="form-grid">
                    <div className="form-group">
                        <label className="form-label">Hospital Name</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="form-control"
                            placeholder="Enter hospital name"
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">CT Image</label>
                        <div className="file-upload-area" onClick={() => document.querySelector('input[type="file"]').click()}>
                            <div className="file-upload-icon">📁</div>
                            <div className="file-upload-text">
                                {ctImage ? ctImage.name : 'Click to select a CT image'}
                            </div>
                            <div className="file-upload-hint">
                                Supports PNG, JPG, JPEG formats
                            </div>
                            <input
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
                            disabled={loading}
                            className={`button ${loading ? 'button-outline' : 'button-primary'}`}
                        >
                            {loading ? 'Converting...' : 'Convert to MRI'}
                        </button>
                    </div>
                </form>
            </div>

            {error && (
                <div className="error-container">
                    <span className="error-icon">⚠️</span>
                    <span className="error-text">{error}</span>
                </div>
            )}

            {mriImage && (
                <div className="results-section">
                    <div className="results-header">
                        <div className="results-header-icon">✨</div>
                        <h3 className="results-title">Converted MRI Image</h3>
                    </div>
                    
                    <div className="results-grid">
                        <div className="result-card">
                            <div className="result-card-title">Original CT</div>
                            <div className="result-image-container">
                                <img
                                    src={URL.createObjectURL(ctImage)}
                                    alt="Original CT"
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

            {loading && (
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <div className="loading-text">Converting your image...</div>
                </div>
            )}
        </div>
    );
};

export default CTtoMRI;
