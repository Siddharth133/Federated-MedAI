import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    // Placeholder: this will be fetched from backend in future
    const uploaded = JSON.parse(localStorage.getItem('uploadedFiles')) || [];
    setFiles(uploaded);
  }, []);

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Dashboard</h2>
      {files.length === 0 ? (
        <p>No files uploaded yet.</p>
      ) : (
        <ul>
          {files.map((file, idx) => (
            <li key={idx}>
              <strong>{file.name}</strong> — {file.status}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dashboard;
