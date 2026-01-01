import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { documentsAPI, queryAPI } from '../api';
import { 
  Upload, File, Trash2, LogOut, Send, FileText, 
  MessageSquare, Loader2, AlertCircle, CheckCircle,
  Bot, User, Sparkles, Clock, FileCheck, X
} from 'lucide-react';

export default function Dashboard() {
  const { logout } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchDocuments = async () => {
    try {
      const response = await documentsAPI.list();
      console.log('Documents response:', response.data);
      // Backend returns array directly or { documents: [] }
      const docs = Array.isArray(response.data) ? response.data : (response.data.documents || []);
      setDocuments(docs);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError('Failed to load documents');
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      await documentsAPI.upload(file, (progress) => {
        setUploadProgress(progress);
      });
      setSuccess('Document uploaded successfully!');
      fetchDocuments();
      setUploadProgress(0);
      e.target.value = '';
    } catch (err) {
      console.error('Upload error:', err);
      console.error('Error response:', err.response);
      const errorMessage = err.response?.data?.detail 
        || (Array.isArray(err.response?.data) ? err.response.data[0]?.msg : null)
        || err.message 
        || 'Failed to upload document';
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (fileId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;

    try {
      await documentsAPI.delete(fileId);
      setSuccess('Document deleted successfully!');
      fetchDocuments();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to delete document');
    }
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    const currentQuestion = question;
    setQuestion('');
    setLoading(true);
    setError('');

    try {
      // Use regular query API instead of streaming for reliability
      const response = await queryAPI.query(currentQuestion);
      const { answer, sources } = response.data;
      
      const assistantMessage = { 
        role: 'assistant', 
        content: answer || 'No response received.', 
        sources: sources || [] 
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Query error:', err);
      const errorMsg = err.response?.data?.detail || 'Failed to get response. Please try again.';
      setError(errorMsg);
      // Remove the loading message and show error
      setMessages((prev) => [...prev, { 
        role: 'assistant', 
        content: `Error: ${errorMsg}`, 
        sources: [] 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* Header with gradient */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-gradient-to-br from-primary-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <FileText className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-indigo-600 bg-clip-text text-transparent">
                  DocTalk
                </h1>
                <p className="text-xs text-gray-500">AI-Powered Document Assistant</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="btn-secondary flex items-center space-x-2 hover:bg-red-50 hover:text-red-600 transition-all duration-200"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Bar */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-primary-100 rounded-lg flex items-center justify-center">
                <FileCheck className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{documents.length}</p>
                <p className="text-xs text-gray-500">Documents Uploaded</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                <MessageSquare className="h-5 w-5 text-indigo-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{messages.length}</p>
                <p className="text-xs text-gray-500">Conversations</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-green-100 rounded-lg flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">AI-Powered</p>
                <p className="text-xs text-gray-500">Gemini 2.5 Flash</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Documents Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
              {/* Panel Header */}
              <div className="bg-gradient-to-r from-primary-600 to-indigo-600 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <File className="h-5 w-5 text-white" />
                    <h2 className="text-lg font-semibold text-white">My Documents</h2>
                  </div>
                  <div className="bg-white/20 backdrop-blur-sm px-2 py-1 rounded-full">
                    <span className="text-xs font-medium text-white">{documents.length}</span>
                  </div>
                </div>
              </div>

              <div className="p-6">
                {/* Upload Button */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className="w-full bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-700 hover:to-indigo-700 text-white font-medium py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Upload className="h-5 w-5" />
                  <span>{uploading ? 'Uploading...' : 'Upload Document'}</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileUpload}
                  className="hidden"
                  accept=".pdf,.doc,.docx,.txt"
                />
                <p className="text-xs text-gray-500 text-center mt-2">
                  Supports PDF, DOC, DOCX, TXT
                </p>

                {/* Upload Progress */}
                {uploading && (
                  <div className="mt-4 bg-blue-50 rounded-lg p-4 border border-blue-100">
                    <div className="flex items-center justify-between text-sm text-blue-900 mb-2 font-medium">
                      <span>Processing document...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-blue-200 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-primary-600 to-indigo-600 h-2.5 rounded-full transition-all duration-300 animate-pulse"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Alerts */}
                {error && (
                  <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-start space-x-2 animate-in fade-in slide-in-from-top-2 duration-300">
                    <AlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">{error}</div>
                    <button onClick={() => setError('')} className="text-red-500 hover:text-red-700">
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}

                {success && (
                  <div className="mt-4 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm flex items-start space-x-2 animate-in fade-in slide-in-from-top-2 duration-300">
                    <CheckCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">{success}</div>
                    <button onClick={() => setSuccess('')} className="text-green-500 hover:text-green-700">
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}

                {/* Documents List */}
                <div className="mt-6 space-y-2 max-h-[500px] overflow-y-auto custom-scrollbar">
                  {documents.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <div className="h-20 w-20 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                        <File className="h-10 w-10 text-gray-400" />
                      </div>
                      <p className="text-sm font-medium text-gray-700">No documents yet</p>
                      <p className="text-xs mt-2">Upload your first document to get started</p>
                    </div>
                  ) : (
                    documents.map((doc, index) => (
                      <div
                        key={doc.file_id}
                        className="group flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg hover:from-primary-50 hover:to-indigo-50 transition-all duration-200 border border-gray-100 hover:border-primary-200 hover:shadow-md animate-in fade-in slide-in-from-left duration-300"
                        style={{ animationDelay: `${index * 50}ms` }}
                      >
                        <div className="flex items-center space-x-3 flex-1 min-w-0">
                          <div className="h-10 w-10 bg-gradient-to-br from-primary-500 to-indigo-500 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm">
                            <File className="h-5 w-5 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate group-hover:text-primary-700 transition-colors">
                              {doc.filename}
                            </p>
                            <div className="flex items-center space-x-2 mt-1">
                              <Clock className="h-3 w-3 text-gray-400" />
                              <p className="text-xs text-gray-500">
                                {new Date(doc.uploaded_at).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteDocument(doc.file_id)}
                          className="ml-2 p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200 opacity-0 group-hover:opacity-100"
                          title="Delete document"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Chat Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg border border-gray-100 h-[700px] flex flex-col overflow-hidden">
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-primary-600 to-indigo-600 px-6 py-4">
                <div className="flex items-center space-x-3">
                  <div className="h-10 w-10 bg-white/20 backdrop-blur-sm rounded-lg flex items-center justify-center">
                    <Bot className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
                    <p className="text-xs text-white/80">Ask anything about your documents</p>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-gray-50 to-white custom-scrollbar">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center max-w-md">
                      <div className="h-24 w-24 mx-auto mb-6 bg-gradient-to-br from-primary-100 to-indigo-100 rounded-full flex items-center justify-center shadow-lg">
                        <MessageSquare className="h-12 w-12 text-primary-600" />
                      </div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">
                        Start Your Conversation
                      </h3>
                      <p className="text-sm text-gray-600 mb-4">
                        Ask questions about your uploaded documents and get instant AI-powered answers
                      </p>
                      <div className="grid grid-cols-1 gap-2 text-left">
                        <div className="bg-white border border-gray-200 rounded-lg p-3 hover:border-primary-300 transition-colors">
                          <p className="text-xs font-medium text-gray-700">💡 Example:</p>
                          <p className="text-xs text-gray-600 mt-1">"Summarize the main points"</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-3 hover:border-primary-300 transition-colors">
                          <p className="text-xs font-medium text-gray-700">💡 Example:</p>
                          <p className="text-xs text-gray-600 mt-1">"What are the key findings?"</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
                      >
                        <div className={`flex items-start space-x-2 max-w-[85%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                          {/* Avatar */}
                          <div className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                            message.role === 'user'
                              ? 'bg-gradient-to-br from-primary-600 to-indigo-600'
                              : 'bg-gradient-to-br from-gray-600 to-gray-800'
                          } shadow-md`}>
                            {message.role === 'user' ? (
                              <User className="h-4 w-4 text-white" />
                            ) : (
                              <Bot className="h-4 w-4 text-white" />
                            )}
                          </div>
                          
                          {/* Message Content */}
                          <div className={`flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div
                              className={`rounded-2xl px-4 py-3 shadow-md ${
                                message.role === 'user'
                                  ? 'bg-gradient-to-r from-primary-600 to-indigo-600 text-white rounded-tr-sm'
                                  : 'bg-white text-gray-900 border border-gray-200 rounded-tl-sm'
                              }`}
                            >
                              <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                              {message.sources && message.sources.length > 0 && (
                                <div className={`mt-3 pt-3 border-t ${message.role === 'user' ? 'border-white/20' : 'border-gray-200'}`}>
                                  <p className="text-xs font-semibold mb-2 flex items-center space-x-1">
                                    <FileText className="h-3 w-3" />
                                    <span>Sources:</span>
                                  </p>
                                  <div className="space-y-1">
                                    {message.sources.map((source, idx) => (
                                      <div key={idx} className={`text-xs flex items-start space-x-2 ${message.role === 'user' ? 'opacity-90' : 'opacity-75'}`}>
                                        <span className="flex-shrink-0">•</span>
                                        <span className="flex-1">
                                          {typeof source === 'object' 
                                            ? `${source.filename || 'Unknown'}${source.page ? ` (Page ${source.page})` : ''}`
                                            : source}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                            <span className="text-xs text-gray-400 mt-1 px-1">
                              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                    {loading && (
                      <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2 duration-300">
                        <div className="flex items-start space-x-2">
                          <div className="flex-shrink-0 h-8 w-8 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center shadow-md">
                            <Bot className="h-4 w-4 text-white" />
                          </div>
                          <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-md">
                            <div className="flex items-center space-x-2">
                              <Loader2 className="h-4 w-4 animate-spin text-primary-600" />
                              <span className="text-sm text-gray-600">Thinking...</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-200 bg-white p-4">
                <form onSubmit={handleQuerySubmit} className="flex space-x-3">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder={documents.length === 0 ? "Upload a document first..." : "Type your question here..."}
                      disabled={loading || documents.length === 0}
                      className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-50 disabled:cursor-not-allowed transition-all duration-200"
                    />
                    {question.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setQuestion('')}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                  <button
                    type="submit"
                    disabled={loading || !question.trim() || documents.length === 0}
                    className="bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-700 hover:to-indigo-700 text-white px-6 py-3 rounded-xl flex items-center space-x-2 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-md"
                  >
                    {loading ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Send className="h-5 w-5" />
                    )}
                    <span className="font-medium">Send</span>
                  </button>
                </form>

                {documents.length === 0 && (
                  <div className="mt-3 text-center">
                    <p className="text-xs text-gray-500 flex items-center justify-center space-x-1">
                      <AlertCircle className="h-3 w-3" />
                      <span>Upload documents to start chatting with your AI assistant</span>
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
