import React, { useState } from 'react';
import axios from 'axios';
import './UploadWeights.css';

const UploadWeights = () => {
  const [hospitalName, setHospitalName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setMessage({ type: '', text: '' });
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!hospitalName || !selectedFile) {
      setMessage({
        type: 'error',
        text: 'Please provide both hospital name and weights file'
      });
      return;
    }

    const formData = new FormData();
    formData.append('username', hospitalName);
    formData.append('weights_file', selectedFile);

    setUploading(true);
    setMessage({ type: '', text: '' });

    try {
      const response = await axios.post('http://localhost:8000/api/upload_weights/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.status === 'success') {
        setMessage({
          type: 'success',
          text: 'Weights uploaded successfully!'
        });
        // Clear form
        setHospitalName('');
        setSelectedFile(null);
      } else {
        throw new Error(response.data.message || 'Upload failed');
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.message || error.message || 'Failed to upload weights'
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-weights">
      <div className="upload-header">
        <h1 className="upload-title">Upload Model Weights</h1>
      </div>

      <form className="upload-form" onSubmit={handleUpload}>
        <div className="form-group">
          <label className="form-label">Hospital Name</label>
          <input
            type="text"
            className="form-control"
            value={hospitalName}
            onChange={(e) => setHospitalName(e.target.value)}
            placeholder="Enter your hospital name"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Model Weights</label>
          <div className="file-upload">
            <input
              type="file"
              className="file-upload-input"
              onChange={handleFileChange}
              accept=".h5,.pth,.weights"
            />
            <div className="file-upload-icon">📤</div>
            <div className="file-upload-text">
              {selectedFile ? (
                <div className="selected-file">
                  <span className="selected-file-icon">📄</span>
                  {selectedFile.name}
                </div>
              ) : (
                <>
                  <p>Drag and drop your weights file here</p>
                  <p>or click to select a file</p>
                  <p className="text-sm">(Supported formats: .h5, .pth, .weights)</p>
                </>
              )}
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="upload-button"
          disabled={uploading}
        >
          {uploading ? (
            <>
              <span className="upload-button-icon">⏳</span>
              Uploading...
            </>
          ) : (
            <>
              <span className="upload-button-icon">⬆️</span>
              Upload Weights
            </>
          )}
        </button>

        {message.text && (
          <div className={message.type === 'success' ? 'success-message' : 'error-message'}>
            <span>{message.type === 'success' ? '✅' : '⚠️'}</span>
            {message.text}
          </div>
        )}
      </form>
    </div>
  );
};

export default UploadWeights; 