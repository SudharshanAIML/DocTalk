import { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    if (token) {
      setUser({ token });
    }
    setLoading(false);
  }, [token]);

  const login = async (email, password, isDemo = false) => {
    let response;
    if (isDemo) {
      response = await authAPI.demoLogin();
    } else {
      response = await authAPI.login(email, password);
    }
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    setToken(access_token);
    setUser({ token: access_token });
    return response.data;
  };

  const register = async (email, password) => {
    const response = await authAPI.register(email, password);
    return response.data;
  };

  const logout = async () => {
    try {
      // Call backend to cleanup session (delete FAISS, docs, etc.)
      await authAPI.logout();
    } catch (error) {
      console.error('Logout cleanup error:', error);
      // Continue with local logout even if backend fails
    }
    
    // Clear local storage and state
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
