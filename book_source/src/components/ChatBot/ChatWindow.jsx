import React, { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import { useChat } from './useChat';
import '@site/src/css/chatbot.css';

const ChatWindow = ({ isVisible, onClose }) => {
  const {
    messages,
    input,
    mode,
    selectedText,
    isLoading,
    handleInputChange,
    handleSubmit,
    setMode,
  } = useChat();

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (!isVisible) return null;

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h2>Physical AI Assistant</h2>
        <button onClick={onClose} className="close-btn">&times;</button>
      </div>
      <div className="chat-modes">
        <button
          className={mode === 'book' ? 'active' : ''}
          onClick={() => setMode('book')}
        >
          Ask about book
        </button>
        <button
          className={mode === 'selectedText' ? 'active' : ''}
          onClick={() => setMode('selectedText')}
          disabled={!selectedText}
        >
          Ask about selected text
        </button>
      </div>
      {mode === 'selectedText' && selectedText && (
        <div className="selected-text-display">
          <p><strong>Selected Text:</strong> "{selectedText}"</p>
        </div>
      )}
      <div className="messages-container">
        {messages.map((msg, index) => (
          <MessageBubble key={index} message={msg} />
        ))}
        {isLoading && <div className="loading-indicator">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
};

export default ChatWindow;
