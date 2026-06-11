import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth APIs
export const authAPI = {
  register: (email, password) => api.post('/auth/register', { email, password }),
  login: (email, password) => api.post('/auth/login', { email, password }),
  demoLogin: () => api.post('/auth/demo'),
  logout: () => api.post('/auth/logout'),  // New: calls backend to cleanup session
};

// Documents APIs
export const documentsAPI = {
  upload: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress && onProgress(percentCompleted);
      },
    });
  },
  
  list: () => api.get('/documents/'),
  delete: (fileId) => api.delete(`/documents/${fileId}`),
  getById: (fileId) => api.get(`/documents/${fileId}`),
};

// Query APIs
export const queryAPI = {
  query: (question, filters = null, conversationId = null, searchContext = null) => 
    api.post('/query/', { 
      question, 
      filters, 
      conversation_id: conversationId,
      search_context: searchContext
    }),
  queryStream: async (question, onChunk, onComplete, onError) => {
    try {
      const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              onComplete && onComplete();
            } else {
              try {
                const parsed = JSON.parse(data);
                onChunk && onChunk(parsed);
              } catch (e) {
                console.error('Error parsing chunk:', e);
              }
            }
          }
        }
      }
    } catch (error) {
      onError && onError(error);
    }
  },
};

// Search APIs (MCP + Web Search)
export const searchAPI = {
  hybridSearch: (query, options = {}) =>
    api.post('/search/hybrid', { query, ...options }),
  
  webSearch: (query, numResults = 5) =>
    api.post('/search/hybrid', { 
      query, 
      search_type: 'web_only', 
      top_k_web: numResults 
    }),
  
  getMCPStatus: () =>
    api.get('/search/mcp/health'),
  
  getCachedResults: (query, limit = 10) =>
    api.get(`/search/cache?query=${encodeURIComponent(query)}&limit=${limit}`),
  
  clearSearchCache: (queryHash) =>
    api.delete(`/search/cache/${queryHash}`),
};

// Deep Search APIs
export const deepSearchAPI = {
  // Main deep search endpoint
  search: (query, options = {}) =>
    api.post('/search/deep', {
      query,
      depth: options.depth || 'moderate',
      include_reasoning: options.includeReasoning !== false,
      cross_document: options.crossDocument !== false,
      context_limit: options.contextLimit || 20000,
      conversation_id: options.conversationId,
    }),
  
  // Get detailed reasoning steps
  getReasoning: (query, depth = 'moderate') =>
    api.post('/search/deep/reasoning', { query, depth }),
  
  // Get document/concept relationships
  getRelationships: (options = {}) => {
    const params = new URLSearchParams();
    if (options.documentId) params.append('document_id', options.documentId);
    if (options.concept) params.append('concept', options.concept);
    if (options.limit) params.append('limit', options.limit);
    return api.get(`/search/relationships?${params.toString()}`);
  },
  
  // Cross-document search
  crossDocumentSearch: (query, documentIds = null) =>
    api.post('/search/cross-document', {
      query,
      document_ids: documentIds,
    }),
  
  // Health check
  getHealth: () =>
    api.get('/search/deep/health'),
  
  // History
  getHistory: (limit = 20) =>
    api.get(`/search/deep/history?limit=${limit}`),
  
  getSearchById: (searchId) =>
    api.get(`/search/deep/history/${searchId}`),
  
  clearHistory: () =>
    api.delete('/search/deep/history'),
};

// Chat History APIs
export const chatAPI = {
  getConversations: (limit = 20, offset = 0) =>
    api.get(`/chat/history?limit=${limit}&offset=${offset}`),
  
  getConversation: (conversationId) =>
    api.get(`/chat/history/${conversationId}`),
  
  newConversation: (title = null) =>
    api.post('/chat/new-conversation', { title }),
  
  renameConversation: (conversationId, title) =>
    api.put(`/chat/history/${conversationId}`, { title }),
  
  deleteConversation: (conversationId) =>
    api.delete(`/chat/history/${conversationId}`),
  
  deleteMessage: (conversationId, messageId) =>
    api.delete(`/chat/history/${conversationId}/${messageId}`),
  
  clearAllHistory: () =>
    api.delete('/chat/history?confirm=true'),
  
  searchHistory: (query, limit = 20) =>
    api.get(`/chat/history/search?query=${encodeURIComponent(query)}&limit=${limit}`),
};

// Perplexica AI Search APIs
export const perplexicaAPI = {
  // Web search with focus modes
  search: (query, options = {}) =>
    api.post('/perplexica/search', {
      query,
      focus_mode: options.focusMode || 'webSearch',
      use_cache: options.useCache !== false,
      history: options.history || []
    }),
  
  // Hybrid search (documents + web)
  hybridSearch: (query, options = {}) =>
    api.post('/perplexica/hybrid', {
      query,
      search_documents: options.searchDocuments !== false,
      search_web: options.searchWeb !== false,
      focus_mode: options.focusMode || 'webSearch',
      doc_top_k: options.docTopK || 5,
      web_top_k: options.webTopK || 5,
      auto_web_threshold: options.autoWebThreshold || 0.6,
      use_cache: options.useCache !== false
    }),
  
  // Get service health
  getHealth: () =>
    api.get('/perplexica/health'),
  
  // Get available focus modes
  getFocusModes: () =>
    api.get('/perplexica/focus-modes'),
  
  // Clear search cache
  clearCache: () =>
    api.delete('/perplexica/cache'),
};

export default api;
