import React, { useState, useRef, useEffect } from 'react';
import { SearchModeDropdown } from './SearchMode';
import FilterPanel from './FilterPanel';

const ChatInput = ({ 
  onSend, 
  disabled, 
  searchMode = 'docs_only',
  onSearchModeChange = () => {},
  documents = [],
  activeFilters = null,
  onFiltersChange = () => {}
}) => {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState([]);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!message.trim() && files.length === 0) return;
    if (disabled) return;

    // Pass message, files, AND the current active filters
    onSend(message, files, activeFilters);
    setMessage('');
    setFiles([]);
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Count how many documents are currently filtered
  const filteredDocCount = activeFilters?.document_ids?.length || 0;

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 rounded-lg bg-[var(--bg-tertiary)] p-2">
          {files.map((file, index) => (
            <div key={index} className="flex items-center gap-1 rounded-md border border-[var(--border-light)] bg-[var(--bg-secondary)] px-2 py-1 text-sm text-[var(--text-secondary)]">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" 
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M13 2v7h7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span className="max-w-[200px] truncate">{file.name}</span>
              <button
                type="button"
                onClick={() => removeFile(index)}
                className="flex h-5 w-5 items-center justify-center rounded-sm text-[var(--text-tertiary)] transition-colors duration-200 hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                aria-label="Remove file"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Active filter badge */}
      {filteredDocCount > 0 && (
        <div className="flex items-center gap-2 rounded-lg bg-[var(--bg-tertiary)] px-3 py-1.5">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" className="text-[var(--accent-primary)]">
            <path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span className="text-xs font-medium text-[var(--accent-primary)]">
            Searching {filteredDocCount} selected document{filteredDocCount > 1 ? 's' : ''} only
          </span>
          <button
            type="button"
            onClick={() => onFiltersChange(null)}
            className="ml-auto text-xs text-[var(--text-tertiary)] hover:text-[var(--error)] transition-colors"
          >
            Clear
          </button>
        </div>
      )}

      <div className="flex items-end gap-2 rounded-xl border-2 border-[var(--border-light)] bg-[var(--bg-secondary)] p-4 transition-colors duration-200 focus-within:border-[var(--accent-primary)]">
        <SearchModeDropdown 
          searchMode={searchMode}
          onModeChange={onSearchModeChange}
          disabled={disabled}
        />
        
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-transparent text-[var(--text-secondary)] transition-colors duration-200 hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] disabled:cursor-not-allowed disabled:opacity-50"
          disabled={disabled}
          aria-label="Attach file"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" 
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          onChange={handleFileSelect}
          className="hidden"
        />

        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={filteredDocCount > 0 ? `Ask about ${filteredDocCount} selected document${filteredDocCount > 1 ? 's' : ''}...` : "Ask about your documents..."}
          className="custom-scrollbar max-h-[200px] flex-1 resize-none overflow-y-auto border-none bg-transparent text-base leading-6 text-[var(--text-primary)] outline-none placeholder:text-[var(--text-placeholder)] disabled:cursor-not-allowed disabled:opacity-60"
          disabled={disabled}
          rows={1}
        />

        {/* Filter Panel button */}
        <FilterPanel
          documents={documents}
          onFiltersChange={onFiltersChange}
          activeFilters={activeFilters}
        />

        <button
          type="submit"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-transparent text-[var(--text-secondary)] transition-colors duration-200 hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] disabled:cursor-not-allowed disabled:opacity-50 enabled:bg-[var(--accent-primary)] enabled:text-white enabled:hover:bg-[var(--accent-hover)]"
          disabled={disabled || (!message.trim() && files.length === 0)}
          aria-label="Send message"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" 
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      <div className="px-2 text-center text-xs text-[var(--text-tertiary)]">
        <span>Press Enter to send, Shift+Enter for new line</span>
      </div>
    </form>
  );
};

export default ChatInput;
