import React, { useState } from 'react';
import CTtoMRI from './components/CTtoMRI';
import Stats from './components/Stats';
import TrainingDashboard from './components/TrainingDashboard';
import UploadWeights from './components/UploadWeights';

const App = () => {
    const [activeTab, setActiveTab] = useState('train'); // 'train', 'convert', 'upload'

    return (
        <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
            <nav className="bg-white shadow-lg dark:bg-gray-800">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between h-16">
                        <div className="flex">
                            <div className="flex-shrink-0 flex items-center">
                                <h1 className="text-2xl font-bold text-gray-800 dark:text-white">Federated Learning System</h1>
                            </div>
                        </div>
                    </div>
                </div>
            </nav>

            <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                {/* Tab Navigation */}
                <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
                    <nav className="-mb-px flex space-x-8">
                        <button
                            onClick={() => setActiveTab('train')}
                            className={`${
                                activeTab === 'train'
                                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                            } whitespace-nowrap py-4 px-1 border-b-2 font-medium`}
                        >
                            Train Model
                        </button>
                        <button
                            onClick={() => setActiveTab('convert')}
                            className={`${
                                activeTab === 'convert'
                                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                            } whitespace-nowrap py-4 px-1 border-b-2 font-medium`}
                        >
                            Convert CT to MRI
                        </button>
                        <button
                            onClick={() => setActiveTab('upload')}
                            className={`${
                                activeTab === 'upload'
                                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                            } whitespace-nowrap py-4 px-1 border-b-2 font-medium`}
                        >
                            Upload Weights
                        </button>
                    </nav>
                </div>

                {/* Tab Content */}
                <div className="mt-4">
                    {activeTab === 'train' && <TrainingDashboard />}
                    {activeTab === 'convert' && <CTtoMRI />}
                    {activeTab === 'upload' && <UploadWeights />}
                </div>

                {/* Stats Section */}
                <div className="mt-8">
                    <Stats />
                </div>
            </div>
        </div>
    );
};

export default App;
