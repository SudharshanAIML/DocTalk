import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Sidebar from './Sidebar';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import { queryAPI, documentsAPI } from '../api/index';
import ThemeToggle from './ThemeToggle';
import { Brain } from 'lucide-react';

const ChatLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [searchMode, setSearchMode] = useState('hybrid'); // docs_only, hybrid, web_only
  const [conversationId, setConversationId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [activeFilters, setActiveFilters] = useState(null);

  // Fetch user documents for filter panel
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await documentsAPI.list();
        // Backend returns a plain array directly
        const docs = Array.isArray(response.data) ? response.data : (response.data?.files || []);
        setDocuments(docs);
      } catch (error) {
        console.error('Failed to fetch documents:', error);
      }
    };
    fetchDocuments();
  }, []);

  const handleSendMessage = async (message, files, filters) => {
    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      files: files
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);

    try {
      // Prepare search context based on search mode
      const searchContext = {
        enable_web_search: searchMode === 'hybrid' || searchMode === 'web_only',
        search_type: searchMode,
        max_web_results: 5
      };

      // Use filters passed from ChatInput (from FilterPanel)
      const queryFilters = filters || activeFilters;

      // Call the query API with search context and filters
      const response = await queryAPI.query(
        message,
        queryFilters,   // pass real filters here
        conversationId,
        searchContext
      );

      // Update conversation id if new
      if (!conversationId) {
        setConversationId(response.data.conversation_id);
      }

      // Add AI response
      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date(),
        sources: response.data.sources || [],
        searchUsed: response.data.web_search_used
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleSearchModeChange = (newMode) => {
    setSearchMode(newMode);
  };

  return (
    <div className="flex h-screen bg-[var(--bg-primary)] font-sans text-[var(--text-primary)]">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Chat Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <div className="flex h-[60px] items-center gap-4 border-b border-[var(--border-light)] bg-[var(--bg-secondary)] px-6 py-4 md:px-4 md:py-2">
          <button 
            className="flex h-9 w-9 items-center justify-center rounded-md text-[var(--text-secondary)] transition-colors duration-200 hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M3 12h18M3 6h18M3 18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
          <h1 className="m-0 text-lg font-semibold text-[var(--text-primary)]">DocTalk</h1>
          
          {/* Deep Search Button */}
          <Link
            to="/deep-search"
            className="ml-4 flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium text-white transition-all hover:opacity-90"
            style={{
              background: "linear-gradient(135deg, var(--accent-primary), var(--accent-hover))",
            }}
          >
            <Brain className="h-4 w-4" />
            <span>Deep Search</span>
          </Link>
          
          <div className="ml-auto">
            <ThemeToggle variant="header" />
          </div>
        </div>

        {/* Messages Container */}
        <div className="custom-scrollbar flex flex-1 justify-center overflow-y-auto px-6 py-8 md:px-4 md:py-4">
          <div className="w-full max-w-[768px]">
            {messages.length === 0 ? (
              <div className="flex min-h-[400px] flex-col items-center justify-center px-8 py-16 text-center">
                <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-xl bg-[var(--bg-tertiary)] text-[var(--text-tertiary)]">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" 
                      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <h2 className="mb-2 mt-0 text-2xl font-semibold text-[var(--text-primary)]">Start a conversation</h2>
                <p className="m-0 text-base text-[var(--text-secondary)]">Upload documents and ask questions about them</p>
              </div>
            ) : (
              <ChatMessages messages={messages} isStreaming={isStreaming} />
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-[var(--border-light)] bg-[var(--bg-secondary)] p-6 md:p-4">
          <div className="mx-auto max-w-[768px]">
            <ChatInput 
              onSend={handleSendMessage}
              disabled={isStreaming}
              searchMode={searchMode}
              onSearchModeChange={handleSearchModeChange}
              documents={documents}
              activeFilters={activeFilters}
              onFiltersChange={setActiveFilters}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatLayout;
