import React, { useState, useEffect } from 'react';
import ChatWindow from './ChatWindow';
import { useChat } from './useChat';
import '@site/src/css/chatbot.css';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { handleSelectedText } = useChat();

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  useEffect(() => {
    const handleMouseUp = () => {
      const text = window.getSelection().toString();
      if (text) {
        handleSelectedText(text);
        setIsOpen(true);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleSelectedText]);

  return (
    <div className="chat-widget-container">
      <button className="chat-widget-button" onClick={toggleChat}>
        🤖
      </button>
      <ChatWindow isVisible={isOpen} onClose={toggleChat} />
    </div>
  );
};

export default ChatWidget;
