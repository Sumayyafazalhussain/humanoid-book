import { useState, useCallback } from 'react';
import { query, querySelectedText } from './apiService';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('book'); // 'book' or 'selectedText'
  const [selectedText, setSelectedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let response;
      if (mode === 'selectedText' && selectedText) {
        response = await querySelectedText(input, selectedText);
      } else {
        response = await query(input);
      }
      
      const botMessage = {
        text: response.answer,
        sender: 'bot',
        sources: response.sources,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error querying API:', error);
      // The apiService now throws with a message that includes the backend's helpful fallback
      const errorMessage = {
        text: error.message || 'An unexpected error occurred. Please try again.',
        sender: 'bot',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectedText = useCallback((text) => {
    if (text) {
      setSelectedText(text);
      setMode('selectedText');
    }
  }, []);

  return {
    messages,
    input,
    mode,
    selectedText,
    isLoading,
    handleInputChange,
    handleSubmit,
    handleSelectedText,
    setMode,
  };
};
