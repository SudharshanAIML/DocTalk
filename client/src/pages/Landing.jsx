import { Link } from 'react-router-dom';
import { FileText, MessageSquare, Upload, Shield, Zap, Users } from 'lucide-react';

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-primary-100">
      {/* Hero Section */}
      <header className="container mx-auto px-4 py-8">
        <nav className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 bg-primary-600 rounded-lg flex items-center justify-center">
              <FileText className="h-6 w-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-gray-900">DocTalk</span>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/login" className="btn-secondary">
              Sign In
            </Link>
            <Link to="/register" className="btn-primary">
              Get Started
            </Link>
          </div>
        </nav>
      </header>

      {/* Hero Content */}
      <main className="container mx-auto px-4 py-16 md:py-24">
        <div className="max-w-4xl mx-auto text-center animate-fade-in">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Chat with Your Documents
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-8">
            Upload your documents and get instant answers powered by AI
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <Link to="/register" className="btn-primary px-8 py-3 text-lg">
              Start Free Trial
            </Link>
            <Link to="/login" className="btn-secondary px-8 py-3 text-lg">
              Learn More
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="card text-center animate-slide-up">
            <div className="mx-auto h-12 w-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
              <Upload className="h-6 w-6 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Easy Upload</h3>
            <p className="text-gray-600">
              Upload PDFs, Word documents, and text files with a single click
            </p>
          </div>

          <div className="card text-center animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <div className="mx-auto h-12 w-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
              <MessageSquare className="h-6 w-6 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Smart Answers</h3>
            <p className="text-gray-600">
              Ask questions and get accurate answers from your documents instantly
            </p>
          </div>

          <div className="card text-center animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <div className="mx-auto h-12 w-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
              <Shield className="h-6 w-6 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Secure & Private</h3>
            <p className="text-gray-600">
              Your documents are encrypted and stored securely with enterprise-grade security
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 mt-24 border-t border-gray-200">
        <div className="text-center text-gray-600">
          <p>&copy; 2026 DocTalk. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
