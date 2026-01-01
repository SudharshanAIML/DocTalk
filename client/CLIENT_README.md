# DocTalk Frontend

A modern, responsive React + TailwindCSS frontend for the DocTalk RAG application.

## Features

- 🎨 Modern and responsive UI with TailwindCSS
- 🔐 Authentication (Login/Register)
- 📄 Document upload and management
- 💬 Real-time chat interface with streaming responses
- 📱 Mobile-friendly design
- ⚡ Fast and optimized with Vite

## Tech Stack

- **React 18** - UI framework
- **React Router** - Navigation
- **TailwindCSS** - Styling
- **Axios** - API requests
- **Lucide React** - Icons
- **Vite** - Build tool

## Getting Started

### Prerequisites

- Node.js 16+ installed
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── api/           # API client and endpoints
├── components/    # Reusable components
├── context/       # React context (Auth)
├── pages/         # Page components
│   ├── Landing.jsx
│   ├── Login.jsx
│   ├── Register.jsx
│   └── Dashboard.jsx
├── App.jsx        # Main app component
└── main.jsx       # Entry point
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Environment Variables

The app connects to the backend at `http://localhost:8000` by default. You can modify this in `src/api/index.js` if needed.

## Key Features

### Authentication
- User registration with email/password
- Secure login with JWT tokens
- Protected routes for authenticated users

### Document Management
- Upload documents (PDF, DOCX, TXT)
- View uploaded documents
- Delete documents
- Real-time upload progress

### Chat Interface
- Ask questions about uploaded documents
- Streaming responses for better UX
- Source citations
- Message history

## Responsive Design

The app is fully responsive and works on:
- 📱 Mobile devices (320px+)
- 📱 Tablets (768px+)
- 💻 Desktop (1024px+)
- 🖥️ Large screens (1280px+)

## License

MIT
