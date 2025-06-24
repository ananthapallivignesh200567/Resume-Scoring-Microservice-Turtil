import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';

// Components
import Dashboard from './components/Dashboard';
import ResumeAnalysis from './components/ResumeAnalysis';
import CareerCoach from './components/CareerCoach';
import LearningPaths from './components/LearningPaths';
import JobMarket from './components/JobMarket';
import Profile from './components/Profile';
import Navigation from './components/Navigation';
import LoadingSpinner from './components/LoadingSpinner';

// Styles
import './App.css';

// Configure Amplify
const amplifyConfig = {
  Auth: {
    region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
    userPoolId: process.env.REACT_APP_USER_POOL_ID,
    userPoolWebClientId: process.env.REACT_APP_USER_POOL_CLIENT_ID,
  },
  API: {
    endpoints: [
      {
        name: 'CareerPlatformAPI',
        endpoint: process.env.REACT_APP_API_ENDPOINT,
        region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
      }
    ]
  }
};

Amplify.configure(amplifyConfig);

function App() {
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    // Initialize app
    const initializeApp = async () => {
      try {
        // Any initialization logic here
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate loading
        setLoading(false);
      } catch (error) {
        console.error('Error initializing app:', error);
        setLoading(false);
      }
    };

    initializeApp();
  }, []);

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="App">
      <Authenticator
        hideSignUp={false}
        components={{
          Header() {
            return (
              <div className="auth-header">
                <h1>🚀 AI Career Platform</h1>
                <p>Your intelligent career development companion</p>
              </div>
            );
          }
        }}
      >
        {({ signOut, user }) => (
          <Router>
            <div className="app-container">
              <Navigation user={user} signOut={signOut} />
              
              <main className="main-content">
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard user={user} />} />
                  <Route path="/resume" element={<ResumeAnalysis user={user} />} />
                  <Route path="/coach" element={<CareerCoach user={user} />} />
                  <Route path="/learning" element={<LearningPaths user={user} />} />
                  <Route path="/market" element={<JobMarket user={user} />} />
                  <Route path="/profile" element={<Profile user={user} />} />
                </Routes>
              </main>
            </div>
          </Router>
        )}
      </Authenticator>
    </div>
  );
}

export default App;