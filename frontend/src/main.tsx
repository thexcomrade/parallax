import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/base.css';
import './styles/bento.css';
import { App } from './App';

const rootElement = document.getElementById('root');
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
