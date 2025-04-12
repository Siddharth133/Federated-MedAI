import React, { useState } from 'react';
import './FileUpload.css';

const FileUpload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!file) return alert("Please select a file");

    const formData = new FormData();
    formData.append('file', file);

    // Later: replace with your backend upload endpoint
    fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        alert("File uploaded successfully!");
        console.log(data);
        // After a successful upload
        const uploaded = JSON.parse(localStorage.getItem('uploadedFiles')) || [];
        uploaded.push({ name: file.name, status: 'Uploaded' });
        localStorage.setItem('uploadedFiles', JSON.stringify(uploaded));
      })
      .catch((err) => {
        console.error(err);
        alert("Upload failed");
      });
  };

  return (
    <div className="upload-container">
      <h2>Upload File</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Upload</button>
      </form>
    </div>
  );
};

export default FileUpload;
