# Federated-MedAI

A federated learning system for medical image processing, specifically focused on CT to MRI conversion using distributed machine learning.

## Features

- **Federated Training**: Train models across multiple hospitals while keeping data private
- **CT to MRI Conversion**: Convert CT scans to MRI-like images using trained models
- **Model Weight Management**: Upload and manage model weights from different hospitals
- **Training Statistics**: Monitor training progress and model performance
- **Dark Mode UI**: Modern, accessible interface with dark mode support

## Tech Stack

### Frontend
- React.js
- Tailwind CSS
- Axios for API calls

### Backend
- Django
- SQLite Database
- TensorFlow/Keras for ML operations

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Federated-MedAI.git
cd Federated-MedAI
```

2. Set up the backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

3. Set up the frontend
```bash
cd frontend
npm install
npm run dev
```

The application should now be running at:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Usage

1. **Training a Model**
   - Navigate to "Train Model"
   - Enter your hospital name and configure training parameters
   - Select your dataset path
   - Start training

2. **Converting CT to MRI**
   - Go to "Convert CT to MRI"
   - Upload a CT scan
   - View the converted MRI result

3. **Uploading Weights**
   - Select "Upload Weights"
   - Enter your hospital name
   - Upload your model weights file
   - Monitor the upload status

## Project Structure

```
Federated-MedAI/
├── backend/
│   ├── core/           # Main Django app
│   ├── client_weights/ # Stored model weights
│   └── manage.py
└── frontend/
    └── frontend/
        ├── src/
        │   ├── components/
        │   └── App.jsx
        └── index.html
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all hospitals participating in the federated learning network
- Special thanks to contributors and maintainers
