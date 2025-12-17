import React from 'react';
import '@site/src/css/chatbot.css';

const MessageBubble = ({ message }) => {
  const { text, sender, sources } = message;
  const isBot = sender === 'bot';

  return (
    <div className={`message-bubble ${isBot ? 'bot' : 'user'}`}>
      <div className="message-text">{text}</div>
      {isBot && sources && (
        <div className="message-sources">
          <strong>Sources:</strong>
          <ul>
            {sources.map((source, index) => (
              <li key={index}>{source}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
