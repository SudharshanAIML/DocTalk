import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LogIn, Mail, Lock, FileText, Zap } from 'lucide-react';
import ThemeToggle from '../components/ThemeToggle';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await login(null, null, true); // true = demo mode
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Demo login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[var(--bg-primary)] px-4 py-6 text-[var(--text-primary)]">
      <div className="absolute right-4 top-4 z-20">
        <ThemeToggle variant="header" />
      </div>

      {/* Background Orbs */}
      <div className="bg-orb w-48 sm:w-[400px] h-48 sm:h-[400px] bg-primary-600 top-[-10%] left-[-5%]" />
      <div className="bg-orb w-40 sm:w-[350px] h-40 sm:h-[350px] bg-accent-600 bottom-[-10%] right-[-5%]" style={{ animationDelay: '-10s' }} />

      <div className="max-w-md w-full space-y-6 sm:space-y-8 relative z-10 animate-fade-in">
        {/* Logo */}
        <div className="text-center">
          <div className="mx-auto h-12 w-12 sm:h-14 sm:w-14 rounded-2xl flex items-center justify-center shadow-glow mb-4 sm:mb-6"
            style={{ background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-hover))' }}>
            <FileText className="h-6 w-6 sm:h-7 sm:w-7 text-white" />
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-[var(--text-primary)]">Welcome back</h2>
          <p className="mt-1 text-xs text-[var(--text-secondary)] sm:mt-1.5 sm:text-sm">Sign in to your DocTalk account</p>
        </div>

        {/* Login Form */}
        <div className="card animate-slide-up">
          <form className="space-y-4 sm:space-y-5" onSubmit={handleSubmit}>
            {error && (
              <div
                className="rounded-xl px-3 py-2.5 text-xs sm:px-4 sm:py-3 sm:text-sm"
                style={{
                  background: 'color-mix(in srgb, var(--error) 14%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--error) 28%, transparent)',
                  color: 'var(--error)',
                }}
              >
                {error}
              </div>
            )}

            <div>
              <label htmlFor="email" className="mb-1 block text-xs font-medium text-[var(--text-secondary)] sm:mb-1.5 sm:text-sm">
                Email address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 sm:pl-3.5 flex items-center pointer-events-none">
                  <Mail className="h-4 w-4 text-[var(--text-tertiary)]" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-field"
                  style={{ paddingLeft: '2.5rem' }}
                  placeholder="you@example.com"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="mb-1 block text-xs font-medium text-[var(--text-secondary)] sm:mb-1.5 sm:text-sm">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 sm:pl-3.5 flex items-center pointer-events-none">
                  <Lock className="h-4 w-4 text-[var(--text-tertiary)]" />
                </div>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field"
                  style={{ paddingLeft: '2.5rem' }}
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 sm:py-3 rounded-xl flex items-center justify-center space-x-2 font-semibold text-xs sm:text-sm text-white transition-all duration-300 disabled:opacity-40 hover:shadow-glow"
              style={{ background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-hover))' }}
            >
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
              ) : (
                <>
                  <LogIn className="h-4 w-4" />
                  <span>Sign in</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleDemoLogin}
              disabled={loading}
              className="w-full py-2.5 sm:py-3 rounded-xl flex items-center justify-center space-x-2 font-semibold text-xs sm:text-sm transition-all duration-300 disabled:opacity-40 border border-[var(--border-light)] hover:border-[var(--accent-primary)] text-[var(--text-primary)]"
            >
              <Zap className="h-4 w-4 text-yellow-500" />
              <span>Try Demo Account</span>
            </button>
          </form>

          <div className="mt-5 sm:mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-[var(--border-light)]" />
              </div>
              <div className="relative flex justify-center text-[10px] sm:text-xs">
                <span className="rounded bg-[var(--bg-secondary)] px-3 text-[var(--text-tertiary)]">
                  Don't have an account?
                </span>
              </div>
            </div>
            <div className="mt-4 sm:mt-5">
              <Link to="/register" className="w-full btn-secondary flex items-center justify-center text-xs sm:text-sm">
                Create new account
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
